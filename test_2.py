import pymupdf4llm
import pathlib
import sys
 
 
filename = sys.argv[1]  # read filename from command line
outname = filename.replace(".pdf", ".md")
md_text = pymupdf4llm.to_markdown(filename, page_chunks=True)

# Extract text from list of dictionaries and join into a single string
if isinstance(md_text, list):
    # Each item in the list is a dictionary with text content
    md_text = "\n\n".join([chunk.get('text', '') for chunk in md_text])

pathlib.Path(outname).write_bytes(md_text.encode())