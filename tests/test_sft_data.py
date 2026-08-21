from nanochat.sft_data import render_sft_conversation


class FakeTokenizer:
    def __init__(self, ids, mask):
        self.ids = ids
        self.mask = mask
        self.requested_max_tokens = None

    def render_conversation(self, conversation, max_tokens=2048):
        self.requested_max_tokens = max_tokens
        return self.ids[:max_tokens], self.mask[:max_tokens]


def test_render_sft_conversation_caps_examples_at_row_capacity():
    tokenizer = FakeTokenizer(list(range(20)), [0, 0, 0] + [1] * 17)
    ids, mask = render_sft_conversation(tokenizer, {}, row_capacity=8)

    assert tokenizer.requested_max_tokens == 8
    assert len(ids) == len(mask) == 8
    assert any(mask[1:])


def test_render_sft_conversation_skips_truncation_without_targets():
    tokenizer = FakeTokenizer(list(range(20)), [0] * 10 + [1] * 10)

    assert render_sft_conversation(tokenizer, {}, row_capacity=8) is None
