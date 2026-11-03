"""Small, testable helpers for supervised fine-tuning data preparation."""


def render_sft_conversation(tokenizer, conversation, row_capacity):
    """Render one conversation so it can always fit in an SFT packing row.

    Returns ``None`` when truncation leaves no assistant target. Feeding a batch
    whose targets are all ignored to mean cross-entropy produces NaN, so callers
    must skip such examples rather than adding them to the packing buffer.
    """
    if not isinstance(row_capacity, int) or isinstance(row_capacity, bool) or row_capacity < 2:
        raise ValueError("row_capacity must be an integer of at least 2")

    ids, mask = tokenizer.render_conversation(
        conversation, max_tokens=row_capacity
    )
    if len(ids) != len(mask):
        raise ValueError("render_conversation returned ids and mask with different lengths")
    if len(ids) > row_capacity:
        raise ValueError("render_conversation exceeded the requested SFT row capacity")
    if len(ids) < 2 or not any(mask[1:]):
        return None
    return ids, mask


def build_packed_sample_ids(conversation_lengths, row_capacity):
    """Return compact sample IDs for one packed row.

    Every conversation receives a distinct non-negative ID. Remaining padding
    positions receive -1. Models expand these IDs into block-diagonal causal
    attention masks only when packed-sample isolation is enabled.
    """
    if not isinstance(row_capacity, int) or isinstance(row_capacity, bool) or row_capacity < 1:
        raise ValueError("row_capacity must be a positive integer")
    if any(
        not isinstance(length, int) or isinstance(length, bool) or length <= 0
        for length in conversation_lengths
    ):
        raise ValueError("conversation lengths must be positive integers")
    content_length = sum(conversation_lengths)
    if content_length > row_capacity:
        raise ValueError("packed conversations exceed row capacity")
    sample_ids = []
    for sample_id, length in enumerate(conversation_lengths):
        sample_ids.extend([sample_id] * length)
    sample_ids.extend([-1] * (row_capacity - content_length))
    return sample_ids


__all__ = ["build_packed_sample_ids", "render_sft_conversation"]
