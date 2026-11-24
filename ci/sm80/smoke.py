# SPDX-License-Identifier: Apache-2.0
"""Run inside the built image on one compute-capability 8.0 device."""

from tempfile import TemporaryDirectory

import lmcache
import torch
from transformers import LlamaConfig

import vllm._C  # Load the native extension from this image.
from vllm import LLM, SamplingParams
from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut


def check_glm_kpool():
    """Compare GLM pool writes and MTP tail updates with an independent CPU reference."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        fwht128_quant_fp8,
        kpool_compress_and_write_cache,
        kpool_decode_update_and_maybe_write_cache_batched,
    )

    batch, pool, width, page = 2, 4, 128, 32
    hadamard = torch.ones(1, 1)
    for _ in range(7):
        hadamard = torch.cat(
            (torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1))
        )

    def reference(rows):
        rotated = ((rows.float() @ hadamard) * (width ** -0.5)).bfloat16().float()
        scale = torch.exp2(torch.ceil(torch.log2(rotated.abs().amax(-1).clamp_min(1e-4) / 448)))
        encoded = (rotated / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return encoded.view(torch.uint8), scale

    tokens = torch.arange(batch * pool * width).reshape(batch, pool, width)
    keys = ((tokens * 7) % 31 - 15).bfloat16()
    keys[0, 0].zero_()
    query = keys.flatten(0, 1)
    expected_q, expected_q_scale = reference(query)
    actual_q, actual_q_scale = fwht128_quant_fp8(query.cuda())
    torch.testing.assert_close(actual_q.view(torch.uint8).cpu(), expected_q)
    torch.testing.assert_close(actual_q_scale.flatten().cpu(), expected_q_scale, rtol=0, atol=0)

    # Equal scores give an exact uniform softmax over the four distinct keys.
    expected_k, expected_scale = reference(keys.float().mean(1).bfloat16())
    # Pad each physical page. Comparing the full backing storage also checks
    # both the row stride and untouched locations, including the middle page.
    stride = page * (width + 4) + 32
    expected_cache = torch.full((3, stride), 165, dtype=torch.uint8)
    locations = [2 * page + 3, 1]
    for row, location in enumerate(locations):
        block, offset = divmod(location, page)
        expected_cache[block, offset * width : (offset + 1) * width] = expected_k[row]
        scale_start = page * width + 4 * offset
        expected_cache[block, scale_start : scale_start + 4] = expected_scale[row : row + 1].view(torch.uint8)

    backing = torch.full_like(expected_cache, 165, device="cuda")
    cache = backing[:, :page * (width + 4)].view(3, page, width + 4)
    k = keys.cuda()
    scores = torch.zeros_like(k)
    ape = torch.zeros(pool, width, device="cuda")
    loc = torch.tensor(locations, device="cuda")
    packed, scale = kpool_compress_and_write_cache(
        cache, k, scores, ape, loc, pool, return_compressed=True
    )
    torch.testing.assert_close(packed.view(torch.uint8).cpu(), expected_k)
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(backing.cpu(), expected_cache)

    ring = 2 * pool
    positions = torch.arange(pool, dtype=torch.int32, device="cuda").repeat(batch, 1)
    # Simulate two unrelated requests with different physical tail blocks.
    tail_blocks = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    tail_loc = tail_blocks[:, None] * ring + positions
    pool_loc = torch.full_like(positions, -1)
    pool_loc[:, -1] = loc.int()
    for next_n in (1, pool):
        backing.fill_(165)
        tail = torch.full((3, 2, ring, width), -37, dtype=k.dtype, device="cuda")
        for begin in range(0, pool, next_n):
            rows = slice(begin, begin + next_n)
            kpool_decode_update_and_maybe_write_cache_batched(
                cache, tail, tail_loc[:, rows], k[:, rows], scores[:, rows], ape,
                pool_loc[:, rows], positions[:, rows], pool,
            )
        torch.testing.assert_close(backing.cpu(), expected_cache)
        torch.testing.assert_close(tail[tail_blocks.long(), 0, :pool].cpu(), keys)
    print("GLM kpool FP8 queries, paged prefill, decode and four-token MTP parity passed.", flush=True)


def main():
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        raise RuntimeError(f"Expected SM80, got {capability}")
    print(torch.cuda.get_device_name(0), "LMCache:", lmcache.__file__, flush=True)

    # Exercise the fork's pre-SM89 FP8 lookup path against a CPU reference.
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    actual = get_e4m3fn_bf16_lut(torch.device("cuda", 0)).cpu()
    torch.testing.assert_close(actual, values.to(torch.bfloat16), rtol=0, atol=0,
                               equal_nan=True)

    check_glm_kpool()

    with TemporaryDirectory() as model_dir:
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"], vocab_size=256, hidden_size=128,
            intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128,
        )
        config.save_pretrained(model_dir)
        engine = LLM(model=model_dir, load_format="dummy", dtype="bfloat16",
                     skip_tokenizer_init=True, enforce_eager=True, max_model_len=128,
                     max_num_seqs=1, gpu_memory_utilization=0.35)
        sampling = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True,
                                  detokenize=False)
        result = engine.generate([{"prompt_token_ids": [1, 12, 23]}], sampling)
        ids = result[0].outputs[0].token_ids
        if len(ids) != 2 or any(t < 0 or t >= config.vocab_size for t in ids):
            raise RuntimeError(f"Unexpected generation: {ids}")
    print("SM80 FP8 lookup, prefill and decode passed.", flush=True)


if __name__ == "__main__":
    main()
