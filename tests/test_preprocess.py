from src.preprocess import clean_text


def test_clean_text_normalizes_html_urls_case_and_punctuation():
    text = "NASA <b>LAUNCHES</b> https://example.com/Mission! It's 2026."
    assert clean_text(text) == "nasa launches"


def test_clean_text_returns_empty_string_for_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(123) == ""
