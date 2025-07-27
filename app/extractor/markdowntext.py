import pymupdf4llm
import pymupdf
import pathlib
import json
import sys
import re
import os # <-- Added for os.path.basename
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple

# ... (all your helper functions like make_serializable, extract_text_fallback, etc. are unchanged) ...
def make_serializable(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    else:
        return str(obj)

def extract_text_fallback(pdf_path):
    print("  Using fallback text extraction...")
    doc = pymupdf.open(pdf_path)
    pages_data = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages_data.append({'text': text, 'metadata': {'page': page_num}})
        else:
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
                pages_data.append({'text': '\n'.join(lines), 'metadata': {'page': page_num}})
            else:
                print(f"    Warning: No text found on page {page_num}")
                pages_data.append({'text': '', 'metadata': {'page': page_num}})
    doc.close()
    return pages_data

def normalize_for_pattern_detection(line: str) -> str:
    normalized = line.strip()
    if len(normalized) < 2: return ""
    normalized = re.sub(r'\bpage\s+\d+\s+of\s+\d+\b', 'page X of Y', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\b\d+\s+of\s+\d+\b', 'X of Y', normalized)
    normalized = re.sub(r'\bpage\s+\d+\b', 'page X', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^\s*\d+\s*$', 'X', normalized)
    normalized = ' '.join(normalized.split()).lower()
    return normalized

def extract_page_lines(page_data: Dict, page_index: int, max_lines: int = 5) -> Dict:
    page_text = page_data.get('text', '')
    page_num = page_data.get('metadata', {}).get('page', page_index + 1)
    all_lines = [line.strip() for line in page_text.split('\n') if line.strip()]
    if not all_lines:
        return {'page_num': page_num, 'all_lines': [], 'top_lines': [], 'bottom_lines': [], 'line_count': 0}
    top_lines = all_lines[:min(max_lines, len(all_lines))]
    bottom_lines = all_lines[-min(max_lines, len(all_lines)):] if len(all_lines) > max_lines else []
    return {'page_num': page_num, 'all_lines': all_lines, 'top_lines': top_lines, 'bottom_lines': bottom_lines, 'line_count': len(all_lines)}

def identify_header_footer_patterns(page_analyses: List[Dict], min_frequency: float = 0.8) -> Tuple[Set[str], Set[str]]:
    pages_with_content = [p for p in page_analyses if p['line_count'] > 0]
    total_pages = len(pages_with_content)
    if total_pages <= 2: return set(), set()
    min_occurrences = max(2, int(total_pages * min_frequency))
    print(f"  Analyzing {total_pages} pages for repeating patterns...")
    print(f"  Minimum occurrences required: {min_occurrences} pages ({min_frequency*100}%)")
    print(f"  Checking only first 5 and last 5 lines of each page")
    all_patterns = defaultdict(lambda: {'pages': set(), 'positions': [], 'examples': []})
    for page_analysis in pages_with_content:
        page_num = page_analysis['page_num']
        all_lines = page_analysis['all_lines']
        if len(all_lines) < 10:
            lines_to_check = [(i, line) for i, line in enumerate(all_lines)]
        else:
            lines_to_check = []
            for i in range(min(6, len(all_lines))): lines_to_check.append((i, all_lines[i]))
            for i in range(max(6, len(all_lines) - 5), len(all_lines)): lines_to_check.append((i, all_lines[i]))
        for i, line in lines_to_check:
            normalized = normalize_for_pattern_detection(line)
            all_patterns[normalized]['pages'].add(page_num)
            relative_pos = i / (len(all_lines) - 1) if len(all_lines) > 1 else 0.5
            all_patterns[normalized]['positions'].append(relative_pos)
            if len(all_patterns[normalized]['examples']) < 3: all_patterns[normalized]['examples'].append(line)
    header_patterns = set()
    footer_patterns = set()
    for pattern, info in all_patterns.items():
        if len(info['pages']) >= min_occurrences:
            avg_position = sum(info['positions']) / len(info['positions'])
            if avg_position <= 0.2:
                header_patterns.add(pattern)
                print(f"  Header pattern: '{pattern}' (appears on {len(info['pages'])} pages, avg pos: {avg_position:.3f})")
                print(f"    Examples: {info['examples'][:2]}")
            elif avg_position >= 0.8:
                footer_patterns.add(pattern)
                print(f"  Footer pattern: '{pattern}' (appears on {len(info['pages'])} pages, avg pos: {avg_position:.3f})")
                print(f"    Examples: {info['examples'][:2]}")
    return header_patterns, footer_patterns

def filter_page_lines(page_analysis: Dict, header_patterns: Set[str], footer_patterns: Set[str]) -> List[str]:
    if page_analysis['line_count'] == 0: return []
    filtered_lines = []
    for line in page_analysis['all_lines']:
        normalized = normalize_for_pattern_detection(line)
        if normalized in header_patterns or normalized in footer_patterns:
            continue
        filtered_lines.append(line)
    return filtered_lines


# CHANGED: The function now accepts full paths as arguments
def pdf_to_markdown(input_pdf_path, output_json_path):
    """
    Convert a PDF file to Markdown JSON with improved text extraction.
    Accepts full input and output paths.
    """
    pdf_name = os.path.basename(input_pdf_path)
    print(f"Processing {pdf_name}...")
    
    print("  Attempting extraction with pymupdf4llm...")
    try:
        md_data = pymupdf4llm.to_markdown(
            input_pdf_path, 
            page_chunks=True,
            ignore_images=True,
            ignore_graphics=True,
            dpi=150,
        )
        total_text = sum(len(page.get('text', '').strip()) for page in md_data)
        if total_text < 100:
            print(f"  pymupdf4llm extracted only {total_text} characters, trying fallback...")
            md_data = extract_text_fallback(input_pdf_path)
        else:
            print(f"  pymupdf4llm successfully extracted {total_text} characters")
    except Exception as e:
        print(f"  pymupdf4llm failed: {e}")
        print("  Using fallback extraction...")
        md_data = extract_text_fallback(input_pdf_path)
    
    metadata = {}
    if md_data and len(md_data) > 0:
        first_page = md_data[0]
        if 'metadata' in first_page:
            metadata = first_page['metadata']
        else:
            metadata = {
                "file_path": input_pdf_path,
                "page_count": len(md_data),
                "extraction_method": "fallback" if 'fallback' in str(type(md_data)) else "pymupdf4llm"
            }
    
    print(f"  Extracted {len(md_data)} page chunks")
    
    print("  Analyzing pages for header/footer detection...")
    page_analyses = []
    max_check_lines = min(6, max(3, len(md_data) // 8))
    with ThreadPoolExecutor(max_workers=min(8, len(md_data))) as executor:
        future_to_index = {
            executor.submit(extract_page_lines, page_data, i, max_check_lines): i 
            for i, page_data in enumerate(md_data)
        }
        for future in as_completed(future_to_index):
            page_analysis = future.result()
            page_analyses.append(page_analysis)
    page_analyses.sort(key=lambda x: x['page_num'])
    
    header_patterns, footer_patterns = identify_header_footer_patterns(page_analyses, min_frequency=0.9)
    
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
                filtered_pages.append({'page_num': page_analysis['page_num'], 'filtered_lines': filtered_lines})
        filtered_pages.sort(key=lambda x: x['page_num'])
    else:
        print("  No repetitive headers/footers detected")
        filtered_pages = [{'page_num': p['page_num'], 'filtered_lines': p['all_lines']} for p in page_analyses]
    
    print("  Creating final JSON structure...")
    lines_data = []
    line_number = 1
    for page_info in filtered_pages:
        page_num = page_info['page_num']
        for line in page_info['filtered_lines']:
            lines_data.append({"line_number": line_number, "page_number": page_num, "text": line})
            line_number += 1
    
    final_output = {
        "metadata": metadata,
        "total_lines": len(lines_data),
        "total_pages": len(md_data),
        "lines": lines_data
    }
    
    # CHANGED: Use the provided output_json_path argument
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False, default=make_serializable)
    
    original_line_count = sum(len(page['all_lines']) for page in page_analyses)
    removed_lines = original_line_count - len(lines_data)
    
    print(f"Converted {pdf_name} to {os.path.basename(output_json_path)}")
    print(f"  Original lines: {original_line_count}")
    print(f"  Final lines: {len(lines_data)}")
    if removed_lines > 0:
        print(f"  Lines removed: {removed_lines} ({removed_lines/original_line_count*100:.1f}%)")
    
    return True # Indicate success

# This block allows you to test this script by itself
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python markdowntext.py <path_to_input.pdf> <path_to_output.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pdf_to_markdown(input_path, output_path)