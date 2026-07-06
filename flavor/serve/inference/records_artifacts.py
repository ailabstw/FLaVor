import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULT_RECORDS_ARTIFACT_HREF_PREFIX = "/invocations/artifacts"
RECORDS_ARTIFACT_MEDIA_TYPE = "application/x-ndjson"
RECORDS_ARTIFACT_NAME_RE = re.compile(r"^records_[A-Za-z0-9_-]{21}\.jsonl$")


def _normalize_href_prefix(href_prefix: str) -> str:
    href_prefix = href_prefix.strip() or DEFAULT_RECORDS_ARTIFACT_HREF_PREFIX
    return "/" + href_prefix.strip("/")


def validate_artifact_name(artifact_name: str) -> str:
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ValueError("Invalid records artifact name.")
    if not RECORDS_ARTIFACT_NAME_RE.fullmatch(artifact_name):
        raise ValueError("Invalid records artifact name.")
    return artifact_name


class TabularRecordsArtifactStore:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        href_prefix: Optional[str] = None,
    ):
        self.output_dir = Path(
            output_dir
            or os.environ.get("AICOCO_RECORDS_OUTPUT_DIR")
            or tempfile.gettempdir()
        )
        self.href_prefix = _normalize_href_prefix(
            href_prefix
            or os.environ.get("AICOCO_RECORDS_HREF_PREFIX")
            or DEFAULT_RECORDS_ARTIFACT_HREF_PREFIX
        )

    def href_for(self, artifact_name: str) -> str:
        artifact_name = validate_artifact_name(artifact_name)
        return f"{self.href_prefix}/{artifact_name}"

    def resolve_path(self, artifact_name: str) -> Path:
        artifact_name = validate_artifact_name(artifact_name)
        return self.output_dir / artifact_name

    def write_jsonl(
        self, records: Iterable[Dict[str, Any]], artifact_name: str
    ) -> Dict[str, Any]:
        output_path = self.resolve_path(artifact_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rows = 0

        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                    f.write("\n")
                    rows += 1
            tmp_path.replace(output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return {
            "format": "jsonl",
            "href": self.href_for(artifact_name),
            "rows": rows,
            "bytes": output_path.stat().st_size,
        }
