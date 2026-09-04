from caption_server import compose_bilingual_caption


PROMPT = 'Return exactly EN: ... followed by a blank line and ZH-CN: ...'


def test_existing_bilingual_caption_is_canonicalized_without_translation():
    calls = []
    caption = 'EN: A red phone rests on a table.\n\nZH-CN: 一部红色手机放在桌上。'

    result, composed = compose_bilingual_caption(caption, PROMPT, calls.append)

    assert result == caption
    assert composed is False
    assert calls == []


def test_english_only_caption_is_translated_and_wrapped():
    calls = []

    def translate(text):
        calls.append(text)
        return '一部带有深色保护壳的手机放在桌上。'

    result, composed = compose_bilingual_caption(
        'EN: A phone with a dark protective case rests on a table.',
        PROMPT,
        translate,
    )

    assert result == (
        'EN: A phone with a dark protective case rests on a table.\n\n'
        'ZH-CN: 一部带有深色保护壳的手机放在桌上。'
    )
    assert composed is True
    assert calls == ['A phone with a dark protective case rests on a table.']


def test_non_bilingual_request_is_unchanged():
    calls = []

    result, composed = compose_bilingual_caption('A concise caption.', 'Describe this image.', calls.append)

    assert result == 'A concise caption.'
    assert composed is False
    assert calls == []
