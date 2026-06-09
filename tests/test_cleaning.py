from app.cleaning import clean_text, extract_keywords


def test_clean_text_handles_vietnamese() -> None:
    result = clean_text("Giảng viên dạy rất dễ hiểu và nhiệt tình 😊")
    assert result
    assert "giảng" in result
    assert "nhiệt" in result
    assert "😊" not in result


def test_clean_text_handles_non_string_and_noise() -> None:
    assert clean_text(None) == ""
    assert clean_text("https://example.com @user !!!") == ""


def test_extract_keywords_returns_short_list() -> None:
    keywords = extract_keywords("phòng máy quá cũ và mạng rất yếu", limit=4)
    assert 1 <= len(keywords) <= 4
