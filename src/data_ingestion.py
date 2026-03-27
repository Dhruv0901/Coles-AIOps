from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetRecord:
    frame_id: str
    source_uri: str
    metadata: dict[str, Any]


class DataIngestion:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def load(self) -> list[DatasetRecord]:
        source = self.config.get("source", {})
        sample_count = int(source.get("sample_count", 16))
        source_uri = str(source.get("path", "data/placeholder"))
        return [
            DatasetRecord(
                frame_id=f"frame_{idx:05d}",
                source_uri=str(Path(source_uri) / f"{idx:05d}.jpg"),
                metadata={"index": idx, "stream": source.get("stream", "default")},
            )
            for idx in range(sample_count)
        ]
