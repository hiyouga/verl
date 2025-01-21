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
"""Utils for tokenization."""

from transformers import PreTrainedTokenizer


def build_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer.

    Args:
        tokenizer_path (str): The name of the tokenizer.
        **kwargs: The keyword arguments for the tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
