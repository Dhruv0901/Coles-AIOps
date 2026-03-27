def test_import_pipeline() -> None:
    from src.pipeline import run_pipeline

    assert callable(run_pipeline)
