from typing import Any

from src.data_ingestion import DatasetRecord


class Preprocessor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def transform(self, records: list[DatasetRecord]) -> list[dict[str, Any]]:
        resize = self.config.get("resize", [224, 224])
        normalize = bool(self.config.get("normalize", True))
        return [
            {
                "frame_id": record.frame_id,
                "source_uri": record.source_uri,
                "metadata": record.metadata,
                "preprocess": {"resize": resize, "normalize": normalize},
            }
            for record in records
        ]
