from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supabase_store import get_supabase_client

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "assistant_knowledge"


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        block = para.strip()
        if not block:
            continue
        if size + len(block) > max_chars and current:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block)
    if current:
        parts.append("\n\n".join(current))
    return parts


def main() -> int:
    client = get_supabase_client()
    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not files:
        print("No knowledge files found.")
        return 1
    for path in files:
        content = path.read_text(encoding="utf-8")
        title = content.splitlines()[0].lstrip("# ").strip() if content.splitlines() else path.stem
        category = path.stem
        doc = {
            "title": title,
            "category": category,
            "source_path": str(path.relative_to(ROOT)),
            "content": content,
            "metadata": {"seeded_by": "seed_knowledge_base.py"},
        }
        try:
            existing = (
                client.table("app_knowledge_base")
                .select("document_id")
                .eq("source_path", doc["source_path"])
                .limit(1)
                .execute()
            )
            if existing.data:
                document_id = existing.data[0]["document_id"]
                client.table("app_knowledge_base").update(doc).eq("document_id", document_id).execute()
                client.table("app_knowledge_chunks").delete().eq("document_id", document_id).execute()
            else:
                resp = client.table("app_knowledge_base").insert(doc).execute()
                document_id = resp.data[0]["document_id"]
            chunks = [
                {
                    "document_id": document_id,
                    "chunk_index": idx,
                    "content": chunk,
                    "metadata": {"source_path": doc["source_path"], "title": title, "category": category},
                }
                for idx, chunk in enumerate(chunk_text(content))
            ]
            if chunks:
                client.table("app_knowledge_chunks").insert(chunks).execute()
            print(f"seeded {path.name}: {len(chunks)} chunks")
        except Exception as exc:
            print(f"failed {path.name}: {type(exc).__name__}: {str(exc)[:300]}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
