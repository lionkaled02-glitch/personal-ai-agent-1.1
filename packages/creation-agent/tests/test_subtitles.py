from creation_agent import make_srt


def test_srt_is_deterministic_and_valid() -> None:
    value = make_srt("One two three four five six", 6)
    assert value == make_srt("One two three four five six", 6)
    assert "00:00:00,000 -->" in value
    assert value.startswith("1\n")
