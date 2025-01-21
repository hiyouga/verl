import json
import readline  # noqa: F401

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


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
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("Tips: use `clear` to remove the history, use `exit` to exit the conversation.")

    messages = []
    while True:
        query = input("\nUser: ")

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        gen_kwargs = {
            "do_sample": True,
            "max_new_tokens": 1024,
            "streamer": streamer,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        print("Assistant: ", end="", flush=True)
        generated_tokens = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
        response = tokenizer.decode(generated_tokens[0, len(input_ids[0]) :], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
