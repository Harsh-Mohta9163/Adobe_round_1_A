
"""
Fixed version of markdowntext.py that handles pymupdf4llm text detection issues
"""

import pymupdf4llm
import pymupdf
import pathlib
import json
import sys
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple


def make_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    else:
        return str(obj)


def extract_text_fallback(pdf_path):
    """
    Fallback text extraction using direct PyMuPDF when pymupdf4llm fails
    """
    print("  Using fallback text extraction...")
    
    doc = pymupdf.open(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(doc, 1):
        # Get all text from the page
        text = page.get_text()
        
        if text.strip():
            pages_data.append({
                'text': text,
                'metadata': {'page': page_num}
            })
        else:
            # If no text, try with different extraction methods
            text_dict = page.get_text("dict")
            lines = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        if line_text.strip():
                            lines.append(line_text.strip())
            
            if lines:
                pages_data.append({
                    'text': '\n'.join(lines),
                    'metadata': {'page': page_num}
                })
            else:
                print(f"    Warning: No text found on page {page_num}")
                pages_data.append({
                    'text': '',
                    'metadata': {'page': page_num}
                })
    
    doc.close()
    return pages_data


def normalize_for_pattern_detection(line: str) -> str:
    """
    Normalize line for pattern detection - only handle page numbers since other elements remain constant.
    """
    normalized = line.strip()
    
    if len(normalized) < 2:
        return ""
    
    # Replace page numbers in various formats
    normalized = re.sub(r'\bpage\s+\d+\s+of\s+\d+\b', 'page X of Y', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\b\d+\s+of\s+\d+\b', 'X of Y', normalized)
    normalized = re.sub(r'\bpage\s+\d+\b', 'page X', normalized, flags=re.IGNORECASE)
    
    # Replace standalone numbers (common page numbers in footers)
    normalized = re.sub(r'^\s*\d+\s*$', 'X', normalized)
    
    # Normalize whitespace
    normalized = ' '.join(normalized.split()).lower()
    
    return normalized


def extract_page_lines(page_data: Dict, page_index: int, max_lines: int = 5) -> Dict:
    """
    Extract lines from a single page and identify top/bottom lines for header/footer detection.
    """
    page_text = page_data.get('text', '')
    page_num = page_data.get('metadata', {}).get('page', page_index + 1)
    
    # Split into lines and filter out empty ones
    all_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
    
    if not all_lines:
        return {
            'page_num': page_num,
            'all_lines': [],
            'top_lines': [],
            'bottom_lines': [],
            'line_count': 0
        }
    
    # Extract top and bottom lines (up to max_lines each)
    top_lines = all_lines[:min(max_lines, len(all_lines))]
    bottom_lines = all_lines[-min(max_lines, len(all_lines)):] if len(all_lines) > max_lines else []
    
    return {
        'page_num': page_num,
        'all_lines': all_lines,
        'top_lines': top_lines,
        'bottom_lines': bottom_lines,
        'line_count': len(all_lines)
    }


def identify_header_footer_patterns(page_analyses: List[Dict], min_frequency: float = 0.99) -> Tuple[Set[str], Set[str]]:
    """
    Identify header and footer patterns that appear consistently across pages.
    Only checks first 5 and last 5 lines of each page for safety.
    """
    pages_with_content = [p for p in page_analyses if p['line_count'] > 0]
    total_pages = len(pages_with_content)
    
    if total_pages <= 2:
        return set(), set()
    
    # Require high frequency - must appear on most pages to be considered header/footer
    min_occurrences = max(2, int(total_pages * min_frequency))
    
    print(f"  Analyzing {total_pages} pages for repeating patterns...")
    print(f"  Minimum occurrences required: {min_occurrences} pages ({min_frequency*100}%)")
    print(f"  Checking only first 5 and last 5 lines of each page")
    
    # Collect patterns only from header/footer areas
    all_patterns = defaultdict(lambda: {'pages': set(), 'positions': [], 'examples': []})
    
    for page_analysis in pages_with_content:
        page_num = page_analysis['page_num']
        all_lines = page_analysis['all_lines']
        
        if len(all_lines) < 10:
            # For short pages, check all lines but mark position
            lines_to_check = [(i, line) for i, line in enumerate(all_lines)]
        else:
            # For longer pages, only check first 5 and last 5 lines
            lines_to_check = []
            # First 5 lines
            for i in range(min(5, len(all_lines))):
                lines_to_check.append((i, all_lines[i]))
            # Last 5 lines
            for i in range(max(5, len(all_lines) - 5), len(all_lines)):
                lines_to_check.append((i, all_lines[i]))
        
        for i, line in lines_to_check:
            normalized = normalize_for_pattern_detection(line)
            all_patterns[normalized]['pages'].add(page_num)
            # Calculate relative position (0 = very top, 1 = very bottom)
            relative_pos = i / (len(all_lines) - 1) if len(all_lines) > 1 else 0.5
            all_patterns[normalized]['positions'].append(relative_pos)
            if len(all_patterns[normalized]['examples']) < 3:
                all_patterns[normalized]['examples'].append(line)
    
    # Identify patterns that appear frequently
    header_patterns = set()
    footer_patterns = set()
    
    for pattern, info in all_patterns.items():
        if len(info['pages']) >= min_occurrences:
            # Calculate average position to determine if it's header or footer
            avg_position = sum(info['positions']) / len(info['positions'])
            
            # Much stricter position thresholds: first 10% = header, last 10% = footer
            if avg_position <= 0.1:
                header_patterns.add(pattern)
                print(f"  Header pattern: '{pattern}' (appears on {len(info['pages'])} pages, avg pos: {avg_position:.3f})")
                print(f"    Examples: {info['examples'][:2]}")
            elif avg_position >= 0.9:
                footer_patterns.add(pattern)
                print(f"  Footer pattern: '{pattern}' (appears on {len(info['pages'])} pages, avg pos: {avg_position:.3f})")
                print(f"    Examples: {info['examples'][:2]}")
    
    return header_patterns, footer_patterns


def filter_page_lines(page_analysis: Dict, header_patterns: Set[str], footer_patterns: Set[str]) -> List[str]:
    """
    Filter out header and footer lines from a single page.
    """
    if page_analysis['line_count'] == 0:
        return []
    
    filtered_lines = []
    
    for line in page_analysis['all_lines']:
        normalized = normalize_for_pattern_detection(line)
        
        # Skip if this line matches any header or footer pattern
        if normalized in header_patterns or normalized in footer_patterns:
            continue
            
        filtered_lines.append(line)
    
    return filtered_lines


def pdf_to_markdown(pdf_name):
    """
    Convert a PDF file to Markdown JSON with improved text extraction.
    Args:
        pdf_name (str): Name of the PDF file (e.g., 'file04.pdf')
    """
    # Extract base filename without extension
    base_filename = pdf_name.replace('.pdf', '')
    
    # Define paths
    input_pdf_path = f"../../data/input/{pdf_name}"
    output_json_path = f"../../data/md_files/{base_filename}.json"
    
    print(f"Processing {pdf_name}...")
    
    # Try pymupdf4llm first with different parameters
    print("  Attempting extraction with pymupdf4llm...")
    
    try:
        # Try with force_text=True and write_images=False (most aggressive text extraction)
        md_data = pymupdf4llm.to_markdown(
            input_pdf_path, 
            page_chunks=True,
            ignore_images=True,
            ignore_graphics=True,
            # Additional parameters to force text extraction
            dpi=150,  # Higher DPI for better text detection
        )
        
        # Check if we got meaningful content
        total_text = sum(len(page.get('text', '').strip()) for page in md_data)
        
        if total_text < 100:  # If very little text extracted
            print(f"  pymupdf4llm extracted only {total_text} characters, trying fallback...")
            md_data = extract_text_fallback(input_pdf_path)
        else:
            print(f"  pymupdf4llm successfully extracted {total_text} characters")
            
    except Exception as e:
        print(f"  pymupdf4llm failed: {e}")
        print("  Using fallback extraction...")
        md_data = extract_text_fallback(input_pdf_path)
    
    # Extract metadata from the first page
    metadata = {}
    if md_data and len(md_data) > 0:
        first_page = md_data[0]
        if 'metadata' in first_page:
            metadata = first_page['metadata']
        else:
            # Create basic metadata
            metadata = {
                "file_path": input_pdf_path,
                "page_count": len(md_data),
                "extraction_method": "fallback" if 'fallback' in str(type(md_data)) else "pymupdf4llm"
            }
    
    print(f"  Extracted {len(md_data)} page chunks")
    
    # Step 1: Parallel analysis of each page to extract lines
    print("  Analyzing pages for header/footer detection...")
    page_analyses = []
    
    # Determine number of lines to check (adaptive based on document size)
    max_check_lines = min(6, max(3, len(md_data) // 8))  # 3-6 lines based on doc size
    
    with ThreadPoolExecutor(max_workers=min(8, len(md_data))) as executor:
        future_to_index = {
            executor.submit(extract_page_lines, page_data, i, max_check_lines): i 
            for i, page_data in enumerate(md_data)
        }
        
        for future in as_completed(future_to_index):
            page_analysis = future.result()
            page_analyses.append(page_analysis)
    
    # Sort by page number to maintain order
    page_analyses.sort(key=lambda x: x['page_num'])
    
    # Step 2: Identify header and footer patterns
    header_patterns, footer_patterns = identify_header_footer_patterns(page_analyses, min_frequency=0.9)
    
    # Step 3: Parallel filtering of lines for each page
    if header_patterns or footer_patterns:
        print(f"  Filtering out {len(header_patterns)} header and {len(footer_patterns)} footer patterns...")
        
        filtered_pages = []
        
        with ThreadPoolExecutor(max_workers=min(8, len(page_analyses))) as executor:
            future_to_page = {
                executor.submit(filter_page_lines, page_analysis, header_patterns, footer_patterns): page_analysis
                for page_analysis in page_analyses
            }
            
            for future in as_completed(future_to_page):
                page_analysis = future_to_page[future]
                filtered_lines = future.result()
                filtered_pages.append({
                    'page_num': page_analysis['page_num'],
                    'filtered_lines': filtered_lines
                })
        
        # Sort by page number to maintain order
        filtered_pages.sort(key=lambda x: x['page_num'])
    else:
        print("  No repetitive headers/footers detected")
        filtered_pages = [{'page_num': p['page_num'], 'filtered_lines': p['all_lines']} for p in page_analyses]
    
    # Step 4: Create final line-by-line structure
    print("  Creating final JSON structure...")
    lines_data = []
    line_number = 1
    
    for page_info in filtered_pages:
        page_num = page_info['page_num']
        
        for line in page_info['filtered_lines']:
            lines_data.append({
                "line_number": line_number,
                "page_number": page_num,
                "text": line
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
    
    # Calculate and display statistics
    original_line_count = sum(len(page['all_lines']) for page in page_analyses)
    removed_lines = original_line_count - len(lines_data)
    
    print(f"Converted {pdf_name} to {output_json_path}")
    print(f"  Original lines: {original_line_count}")
    print(f"  Final lines: {len(lines_data)}")
    if removed_lines > 0:
        print(f"  Lines removed: {removed_lines} ({removed_lines/original_line_count*100:.1f}%)")
        print(f"  Header patterns removed: {len(header_patterns)}")
        print(f"  Footer patterns removed: {len(footer_patterns)}")
    else:
        print(f"  No headers/footers detected for removal")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python markdowntext_fixed.py input.pdf")
        sys.exit(1)
    
    pdf_name = sys.argv[1]
    pdf_to_markdown(pdf_name)



































