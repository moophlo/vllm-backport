# SPDX-License-Identifier: Apache-2.0
"""Compile the port's Triton functions for an explicit Ampere target without a GPU."""
import ast
import importlib.util
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

root = Path(sys.argv[1])
kernels = {
    "vllm/v1/attention/ops/common.py": {"f32_to_e4m3_uint8", "e4m3_uint8_to_f32", "pack_ue8m0_x4"},
    "vllm/v1/attention/ops/mqa_logits_triton.py": {"_fp8_mqa_logits_kernel", "_fp8_paged_mqa_logits_kernel"},
    "vllm/v1/attention/ops/triton_mla_sparse.py": {"_triton_mla_sparse_fwd_kernel", "_triton_mla_sparse_splitk_kernel", "_triton_mla_sparse_reduce_kernel"},
    "vllm/models/glm5next/nvidia/ops/kpool_compress.py": {"_fp8_store_value", "_hadamard128_stage", "_hadamard128", "_fwht_stage", "_fwht_quant_kernel", "_kpool_softmax_rotate_write_cache_kernel", "_kpool_decode_update_batched_kernel"},
}
def compile_kernel(fn, pointers, constants, warps=4):
    signature = {}
    for name in fn.arg_names:
        if name in constants:
            signature[name] = "constexpr"
        elif name in pointers:
            signature[name] = "*" + pointers[name]
        elif name == "sm_scale_log2":
            signature[name] = "fp32"
        elif name.startswith("stride_") or name == "kv_block_stride_bytes":
            signature[name] = "i64"
        else:
            assert not name.endswith("_ptr"), (fn.__name__, name)
            signature[name] = "i32"
    obj = triton.compile(ASTSource(fn, signature, constexprs=constants),
                         target=GPUTarget("cuda", 80, 32), options={"num_warps": warps})
    assert obj.asm["cubin"] and ".target sm_80" in obj.asm["ptx"]
    print("sm_80 cubin", fn.__name__, len(obj.asm["cubin"]), constants, flush=True)

with TemporaryDirectory() as tmp:
    module = Path(tmp) / "sm80_kernel_sources.py"
    segments = ["import triton", "import triton.language as tl"]
    for path, names in kernels.items():
        source = (root / path).read_text()
        lines = source.splitlines()
        functions = {f.name: f for f in ast.parse(source).body if isinstance(f, ast.FunctionDef)}
        for f in functions.values():
            if f.name in names:
                segments.extend(["", *lines[f.decorator_list[0].lineno - 1:f.end_lineno]])
        assert names <= functions.keys(), names - functions.keys()
    module.write_text("\n".join(segments) + "\n")
    spec = importlib.util.spec_from_file_location(module.stem, module)
    k = importlib.util.module_from_spec(spec)
    sys.modules[module.stem] = k
    spec.loader.exec_module(k)
    compile_kernel(k._fwht_quant_kernel,
                   dict(q_ptr="bf16", qout_ptr="u8", sout_ptr="fp32"), dict(BLOCK_R=32), 2)
    pool = dict(PAGE_SIZE=32, BUF_NUMEL_PER_PAGE=4224, POOL_SIZE=4, HEAD_DIM=128,
                S_OFFSET_NBYTES_IN_PAGE=4096, ROUND_SCALE=True, MX_SCALE=False, BLOCK_D=128)
    compile_kernel(k._kpool_softmax_rotate_write_cache_kernel,
                   dict(buf_fp8_ptr="u8", buf_fp32_ptr="fp32", slot_k_ptr="bf16",
                        slot_score_ptr="bf16", ape_ptr="fp32", loc_ptr="i64",
                        write_mask_ptr="i1", compressed_k_ptr="u8", compressed_scale_ptr="fp32"),
                   dict(pool, HAS_WRITE_MASK=True, RETURN_COMPRESSED=True, WRITE_CACHE=True))
    compile_kernel(k._kpool_decode_update_batched_kernel,
                   dict(buf_fp8_ptr="u8", buf_fp32_ptr="fp32", tail_kv_ptr="bf16",
                        tail_slot_mapping_ptr="i64", key_ptr="bf16", slot_score_ptr="bf16",
                        ape_ptr="fp32", slot_mapping_ptr="i64", positions_ptr="i64"),
                   dict(pool, RING=8, TAIL_BLOCK_ELEMS=2048, KPOOL_HEAD=1024))
    mqa = dict(NUM_HEADS=32, HEAD_DIM=128, BLOCK_H=32, BLOCK_D=128, BLOCK_N=32, EMULATE_FP8=True)
    compile_kernel(k._fp8_mqa_logits_kernel,
                   dict(q_ptr="u8", k_ptr="u8", k_scale_ptr="fp32", weights_ptr="fp32",
                        cu_ks_ptr="i32", cu_ke_ptr="i32", out_ptr="fp32"), mqa)
    for width in (1,4):
        compile_kernel(k._fp8_paged_mqa_logits_kernel,
                       dict(q_ptr="u8", kv_u8_ptr="u8", weights_ptr="fp32", ctx_lens_ptr="i32",
                            block_tables_ptr="i32", indices_ptr="i32", out_ptr="fp32"),
                       dict(mqa, BLOCK_SIZE=32, NEXT_N=width, CTX_2D=True, HAS_INDICES=width==4))
    ptr = dict(q_ptr="bf16", kv_ptr="bf16", indices_ptr="i32", out_ptr="bf16", lse_ptr="fp32",
               partial_out_ptr="fp32", partial_lse_ptr="fp32")
    for rope in (0,64):
        mla = dict(NUM_HEADS=32, KV_LORA_RANK=512, ROPE_DIM=rope, BLOCK_H=16, BLOCK_N=32)
        compile_kernel(k._triton_mla_sparse_fwd_kernel, ptr,
                       dict(mla, RETURN_LSE=True, LOGE2=math.log(2)))
        compile_kernel(k._triton_mla_sparse_splitk_kernel, ptr, mla)
    compile_kernel(k._triton_mla_sparse_reduce_kernel, ptr,
                   dict(NUM_KV_SPLITS=4, KV_LORA_RANK=512, RETURN_LSE=True, LOGE2=math.log(2)))
