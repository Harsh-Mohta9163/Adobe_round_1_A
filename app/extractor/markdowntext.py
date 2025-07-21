# """
# Convert document to Markdown text.

# Execute as "python markdowntext.py input.pdf"

# The output will be a file named "input.pdf-markdown.md"
# """

# import pymupdf4llm
# import pathlib

# def pdf_to_markdown(pdf_name):
#     """
#     Convert a PDF file to Markdown and save it to the specified output file.
#     Args:
#         pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
#     """
#     # Extract base filename without extension
#     base_filename = pdf_name.replace('.pdf', '')
    
#     # Define paths
#     input_pdf_path = f"../../data/input/{pdf_name}"
#     output_md_path = f"../../data/md_files/{base_filename}.md"
    
#     # Convert PDF to Markdown
#     md_text = pymupdf4llm.to_markdown(input_pdf_path)
#     pathlib.Path(output_md_path).write_bytes(md_text.encode())











"""
Convert document to Markdown text with page chunks and output as JSON.

Execute as "python markdowntext.py input.pdf"

The output will be a JSON file with metadata and line-level page information.
"""

import pymupdf4llm
import pathlib
import json
import sys


def make_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    else:
        return str(obj)


def pdf_to_markdown(pdf_name):
    """
    Convert a PDF file to Markdown JSON and save it to the specified output file.
    Args:
        pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
    """
    # Extract base filename without extension
    base_filename = pdf_name.replace('.pdf', '')
    
    # Define paths - keeping same structure as before
    input_pdf_path = f"../../data/input/{pdf_name}"
    output_json_path = f"../../data/md_files/{base_filename}.json"  # Changed to .json
    
    print(f"Processing {pdf_name}...")
    
    # Convert PDF to Markdown with page chunks
    md_data = pymupdf4llm.to_markdown(input_pdf_path, page_chunks=True)
    
    # Extract metadata from the first page
    metadata = {}
    if md_data and len(md_data) > 0:
        first_page = md_data[0]
        if 'metadata' in first_page:
            metadata = first_page['metadata']
    
    # Process each page and extract lines
    lines_data = []
    line_number = 1
    
    for page_data in md_data:
        page_num = page_data.get('metadata', {}).get('page', 1)
        page_text = page_data.get('text', '')
        
        # Split text into lines and process each line
        lines = page_text.split('\n')
        
        for line in lines:
            # Skip empty lines
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            lines_data.append({
                "line_number": line_number,
                "page_number": page_num,
                "text": line_stripped
            })
            line_number += 1
    
    # Create final JSON structure
    final_output = {
        "metadata": metadata,
        "total_lines": len(lines_data),
        "total_pages": len(md_data),
        "lines": lines_data
    }
    
    # Save to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False, default=make_serializable)
    
    print(f"Converted {pdf_name} to {output_json_path}")
    print(f"Total lines: {len(lines_data)}")
    print(f"Total pages: {len(md_data)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python markdowntext.py input.pdf")
        sys.exit(1)
    
    pdf_name = sys.argv[1]
    pdf_to_markdown(pdf_name)