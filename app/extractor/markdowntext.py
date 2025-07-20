"""
Convert document to Markdown text.

Execute as "python markdowntext.py input.pdf"

The output will be a file named "input.pdf-markdown.md"
"""

import pymupdf4llm
import pathlib

def pdf_to_markdown(pdf_name):
    """
    Convert a PDF file to Markdown and save it to the specified output file.
    Args:
        pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
    """
    # Extract base filename without extension
    base_filename = pdf_name.replace('.pdf', '')
    
    # Define paths
    input_pdf_path = f"../../data/input/{pdf_name}"
    output_md_path = f"../../data/md_files/{base_filename}.md"
    
    # Convert PDF to Markdown
    md_text = pymupdf4llm.to_markdown(input_pdf_path)
    pathlib.Path(output_md_path).write_bytes(md_text.encode())