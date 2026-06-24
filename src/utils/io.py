from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_bilingual_readme(
    directory: str | Path,
    *,
    title: str,
    english: str,
    chinese: str,
    preserve_existing: bool = False,
) -> None:
    destination = Path(directory) / "README.md"
    existing = destination.read_text(encoding="utf-8") if preserve_existing and destination.exists() else ""
    marker = "<!-- codex-bilingual-readme -->"
    if marker in existing:
        existing = existing.split("<!-- /codex-bilingual-readme -->", 1)[-1].lstrip("\n")
    lines = [
        marker,
        f"# {title}",
        "",
        "## English",
        english,
        "",
        "## 中文",
        chinese,
        "<!-- /codex-bilingual-readme -->",
    ]
    if existing.strip():
        lines.extend(["", existing.rstrip()])
    ensure_dir(destination.parent)
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
