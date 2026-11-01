"""Lesson 1: list the evidence files. Uses only Python's standard library."""
from pathlib import Path

project_folder = Path(__file__).resolve().parent
corpus_folder = project_folder / "corpus"
files = sorted(path for path in corpus_folder.rglob("*") if path.is_file())

print(f"Found {len(files)} corpus files:\n")
for path in files:
    content = path.read_text(encoding="utf-8")
    relative_name = path.relative_to(corpus_folder)
    print(f"{str(relative_name):45} {len(content.splitlines()):3} lines")

print("\nThese are source documents. Evaluation answers are kept outside corpus/.")
