# SPDX-License-Identifier: Apache-2.0
"""Run inside the built image on one compute-capability 8.0 device."""

from tempfile import TemporaryDirectory

import lmcache
import torch
from transformers import LlamaConfig

import vllm._C  # Load the native extension from this image.
from vllm import LLM, SamplingParams
from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut


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
