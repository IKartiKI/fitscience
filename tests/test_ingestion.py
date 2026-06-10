from src.ingestion_pipeline import chunk_text, parse_extraction_json


def test_chunk_text_splits_long_text():
    text = "Muscle hypertrophy. " * 500  # ~10,000 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)


def test_chunk_text_short_text_is_one_chunk():
    chunks = chunk_text("Short text about squats.")
    assert chunks == ["Short text about squats."]


def test_parse_extraction_json_strips_markdown_fences():
    raw = '```json\n{"title": "T", "year": 2020, "authors": [], "claims": ["c1"], "exercises": [], "muscle_groups": []}\n```'
    data = parse_extraction_json(raw)
    assert data["title"] == "T"
    assert data["claims"] == ["c1"]


def test_parse_extraction_json_handles_plain_json():
    raw = '{"title": null, "year": null, "authors": [], "claims": [], "exercises": [], "muscle_groups": []}'
    assert parse_extraction_json(raw)["title"] is None
