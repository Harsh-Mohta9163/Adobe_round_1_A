"""
Convert document to Markdown text.

Execute as "python markdowntext.py input.pdf"

The output will be a file named "input.pdf-markdown.md"
"""

import pymupdf4llm
import pathlib

def pdf_to_markdown(input_pdf_path, output_md_path):
    """
    Convert a PDF file to Markdown and save it to the specified output file.
    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_md_path (str): Path to the output Markdown file.
    """
    md_text = pymupdf4llm.to_markdown(input_pdf_path)
    pathlib.Path(output_md_path).write_bytes(md_text.encode())