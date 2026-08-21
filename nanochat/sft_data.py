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


__all__ = ["render_sft_conversation"]
