"""Deterministic max-min allocation of a bounded evidence content budget."""

from functools import lru_cache
import logging


class ContextBudgetExceeded(ValueError):
    """The complete outgoing answer request exceeds the application budget."""


def check_context_budget(messages: list[dict], context_window: int, output_tokens: int) -> int:
    """Check all text, framing allowance and output using the shared estimator.

    Not a model-specific tokenizer guarantee; never silently truncate evidence
    or instructions to make a request fit.
    """
    if context_window <= 0 or output_tokens < 0:
        raise ValueError("Invalid context window or output reserve")
    tokenizer = evidence_tokenizer()
    texts = [str(value) for message in messages for value in message.values()]
    count = sum(len(tokenizer.encode(text, disallowed_special=())) if tokenizer is not None
                else len(text.encode("utf-8")) for text in texts)
    estimate = count + 16 * len(messages) + 256
    if estimate + output_tokens > context_window:
        raise ContextBudgetExceeded(
            f"回答上下文超出当前预算：输入估算 {estimate}，输出预留 {output_tokens}，"
            f"总预算 {context_window}。请减少检索数量、缩短问题或历史，"
            "或确认模型支持后调整上下文预算；本次超限请求未发送至回答模型。"
        )
    return estimate


@lru_cache(maxsize=1)
def evidence_tokenizer():
    # An explicit estimator, not a claim to match every hosted model tokenizer.
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logging.getLogger(__name__).warning("Evidence tokenizer unavailable; using conservative UTF-8 byte budget")
        return None


def bound_evidence_tokens(sources: list[str], budget: int = 8000) -> list[str]:
    tokenizer = evidence_tokenizer()
    if tokenizer is None:
        encoded_bytes = [source.encode("utf-8") for source in sources]
        allocations = allocate_evidence_budget([len(value) for value in encoded_bytes], budget)
        return [value[:size].decode("utf-8", errors="ignore") for value, size in zip(encoded_bytes, allocations)]
    encoded = [tokenizer.encode(source, disallowed_special=()) for source in sources]
    allocations = allocate_evidence_budget([len(tokens) for tokens in encoded], budget)
    # A token boundary can split a UTF-8 codepoint; remove an incomplete suffix,
    # rather than introducing replacement characters into quoted evidence.
    return [tokenizer.decode_bytes(tokens[:size]).decode("utf-8", errors="ignore")
            for tokens, size in zip(encoded, allocations)]


def allocate_evidence_budget(lengths: list[int], budget: int) -> list[int]:
    if budget < 0 or any(length < 0 for length in lengths):
        raise ValueError("Evidence lengths and budget must be nonnegative")
    allocation = [0] * len(lengths)
    remaining = budget
    for position, index in enumerate(sorted(range(len(lengths)), key=lambda i: (lengths[i], i))):
        share = remaining // (len(lengths) - position)
        allocation[index] = min(lengths[index], share)
        remaining -= allocation[index]
    return allocation
