"""
Convert document to Markdown text with page chunks and metadata.

Execute as "python test.py input.pdf"

The output will be a file named "input.pdf.json"
"""

import pymupdf4llm
import pathlib
import sys
import json


def make_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    else:
        return str(obj)


filename = sys.argv[1]
outname = filename.replace(".pdf", ".json")

# Try with force_text=True to extract text from images
md_data = pymupdf4llm.to_markdown(
    filename, 
    page_chunks=True,
)

# Output as JSON to preserve both text and metadata
with open(outname, 'w', encoding='utf-8') as f:
    json.dump(md_data, f, indent=2, ensure_ascii=False, default=make_serializable)

print(f"Converted {filename} to {outname}")
print(f"Total pages: {len(md_data) if isinstance(md_data, list) else 1}")