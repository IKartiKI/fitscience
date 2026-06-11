from src.ingestion_pipeline import normalize_verdict


def test_normalize_verdict_accepts_valid_values():
    assert normalize_verdict("CONTRADICT") == "CONTRADICT"
    assert normalize_verdict(" agree \n") == "AGREE"
    assert normalize_verdict("Unrelated") == "UNRELATED"


def test_normalize_verdict_defaults_to_unrelated_on_garbage():
    assert normalize_verdict("These claims seem to contradict") == "UNRELATED"
    assert normalize_verdict("") == "UNRELATED"
    assert normalize_verdict(None) == "UNRELATED"
