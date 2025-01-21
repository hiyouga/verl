# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Literal

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class GSM8KDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int, split: Literal["train", "test"] = "train"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._data = load_dataset("openai/gsm8k", "main", split=split)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        messages = [
            {"role": "user", "content": self._data["question"][index]},
            {"role": "assistant", "content": self._data["answer"][index]},
        ]
        prompt_str = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        response_str = self.tokenizer.apply_chat_template(messages, tokenize=False)[len(prompt_str) :]
        if not response_str.rstrip().endswith(self.tokenizer.eos_token):
            response_str += self.tokenizer.eos_token

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_str, add_special_tokens=False)
        input_ids = torch.tensor(prompt_ids + response_ids)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.size(0))
        loss_mask = torch.tensor([0] * len(prompt_ids) + [1] * len(response_ids))
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        if len(input_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(input_ids)
            return {k: F.pad(v, (0, pad_length)) for k, v in model_inputs.items()}
        else:
            return {k: v[-self.max_seq_len :] for k, v in model_inputs.items()}
