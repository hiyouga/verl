import json

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from verl.third_party.vllm import LLM


@hydra.main(config_path=None, version_base=None)
def main(config):
    print(json.dumps(OmegaConf.to_container(config), indent=2, ensure_ascii=False))
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    vllm_engine = LLM(
        model,
        tokenizer=tokenizer,
        model_hf_config=model.config,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        enforce_eager=True,
        skip_tokenizer_init=False,
        max_model_len=8192,
        load_format="dummy_dtensor",
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        skip_special_tokens=False,
    )
    message = [{"role": "user", "content": "What is 45+87?"}]
    token_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True)
    inputs = [{"prompt_token_ids": token_ids}] * 8
    results = vllm_engine.generate(inputs, sampling_params)
    for result in results:
        print(result.outputs[0].text)


if __name__ == "__main__":
    main()
