"""Sample resumes and job descriptions shipped with the app (all fictional).

Each .txt file starts with header lines, then the document:
    # title: Junior Data Analyst
    # role: Data Analyst
    # level: fresher
The web demo only ever analyses these samples, never an uploaded resume.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.services.catalog import DEFAULT_DATA_DIR

Kind = Literal["resumes", "jds"]


class Sample(BaseModel):
    id: str
    title: str
    meta: dict[str, str]
    text: str


def load_samples(kind: Kind, data_dir: Path | None = None) -> list[Sample]:
    folder = (data_dir or DEFAULT_DATA_DIR) / f"sample_{kind}"
    out = []
    for path in sorted(folder.glob("*.txt")):
        header: dict[str, str] = {}
        body: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not body and line.startswith("#") and ":" in line:
                key, value = line[1:].split(":", 1)
                header[key.strip().lower()] = value.strip()
            else:
                body.append(line)
        if "title" not in header:
            raise ValueError(f"{path.name}: add a first line like '# title: AI Engineer'")
        out.append(Sample(id=path.stem, title=header.pop("title"), meta=header, text="\n".join(body).strip()))
    return out


def get_sample(kind: Kind, sample_id: str) -> Sample | None:
    return next((s for s in load_samples(kind) if s.id == sample_id), None)
