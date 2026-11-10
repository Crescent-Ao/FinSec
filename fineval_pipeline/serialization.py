"""Normalize only the empty thinking scaffold used by the Qwen chat template."""
import re


def strip_empty_thinking_prefix(text):
    if not isinstance(text, str):
        raise TypeError('Model completion must be text')
    return re.sub(r'\A(?:\s*<think>\s*</think>)+\s*', '', text).strip()
