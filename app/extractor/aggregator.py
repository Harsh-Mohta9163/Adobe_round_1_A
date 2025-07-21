import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

def load_json_data(json_file: str) -> Tuple[List[Dict], Dict[int, List[int]]]:
    """Load spans and create page index for faster lookup"""
    with open(json_file, 'r', encoding='utf-8') as f:
        spans = json.load(f)
    
    # Sort by page_num, then by column, then by bbox[1] (y-coordinate)
    spans.sort(key=lambda x: (x['page_num'], x['column'], x['bbox'][1] if len(x['bbox']) >= 2 else 0))
    
    # Create page index: page_num -> list of span indices
    page_index = {}
    for i, span in enumerate(spans):
        page_num = span.get('page_num', 1)
        if page_num not in page_index:
            page_index[page_num] = []
        page_index[page_num].append(i)
    
    return spans, page_index

def load_md_file(md_file: str) -> List[str]:
    """Load markdown file and return lines"""
    with open(md_file, 'r', encoding='utf-8') as f:
        return f.readlines()

def is_table_content(md_text: str) -> bool:
    """
    Determine if the markdown text is part of a table.
    Tables in markdown are identified by the presence of | characters.
    """
    # Remove leading/trailing whitespace
    text = md_text.strip()
    
    # Check if the line starts and ends with | (typical table row)
    if text.startswith('|') and text.endswith('|'):
        return True
    
    # Check if the line contains multiple | characters (likely a table row)
    if text.count('|') >= 2:
        return True
    
    return False

def clean_table_text(text: str) -> str:
    """
    Clean table text for better matching by removing table-specific characters
    and handling multi-line content.
    """
    if not text:
        return ""
    
    # Remove | characters used for table formatting
    cleaned = text.replace('|', '')
    
    # Handle <br> tags in table cells by replacing with spaces
    cleaned = re.sub(r'<br\s*/?>', ' ', cleaned)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()

def extract_table_cell_content(md_text: str) -> List[str]:
    """
    Extract individual cell contents from a table row.
    """
    if not is_table_content(md_text):
        return []
    
    # Remove leading and trailing |
    content = md_text.strip()
    if content.startswith('|'):
        content = content[1:]
    if content.endswith('|'):
        content = content[:-1]
    
    # Split by | to get individual cells
    cells = [cell.strip() for cell in content.split('|')]
    
    # Clean each cell
    cleaned_cells = []
    for cell in cells:
        # Remove markdown formatting like **text**
        cell_cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cell)
        # Handle <br> tags
        cell_cleaned = re.sub(r'<br\s*/?>', ' ', cell_cleaned)
        # Clean up whitespace
        cell_cleaned = ' '.join(cell_cleaned.split())
        if cell_cleaned:  # Only add non-empty cells
            cleaned_cells.append(cell_cleaned)
    
    return cleaned_cells

def clean_text(text: str, is_table: bool = False) -> str:
    """Clean text for comparison by removing extra whitespace and special characters"""
    # Remove extra whitespace, newlines, and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # For table content, remove | characters and handle <br> tags
    if is_table:
        text = clean_table_text(text)
    
    # Remove markdown formatting for comparison
    text = re.sub(r'[*_#`\[\]()]+', '', text)
    return text.lower()

def clean_md_line(md_line: str) -> str:
    """Clean markdown line by removing markdown formatting"""
    # Remove leading/trailing whitespace
    cleaned = md_line.strip()
    
    # Handle table content first
    if is_table_content(cleaned):
        cleaned = clean_table_text(cleaned)
    
    # Remove markdown headers (# ## ###)
    cleaned = re.sub(r'^#{1,6}\s*', '', cleaned)
    
    # Remove bold/italic markers (**text** or *text*)
    cleaned = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', cleaned)
    
    cleaned = re.sub(r'^[-+=@]+\s*', '', cleaned)
    
    
    # Remove other markdown formatting
    cleaned = re.sub(r'[_`\[\]()]+', '', cleaned)
    
    return cleaned.strip()

def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings"""
    return SequenceMatcher(None, clean_text(text1), clean_text(text2)).ratio()

def find_best_matching_span_optimized(md_line: str, spans: List[Dict], page_index: Dict[int, List[int]], 
                                     used_spans: set, last_matched_page: int, min_similarity: float = 0.6) -> Tuple[Optional[Dict], int]:
    """Find best matching span with page-based optimization and enhanced table support"""
    if not md_line.strip():
        return None, last_matched_page
    
    # Check if this is table content
    is_table = is_table_content(md_line)
    
    # Clean the markdown line
    cleaned_md_line = clean_md_line(md_line)
    
    if len(clean_text(cleaned_md_line, is_table)) < 3:  # Skip very short lines
        return None, last_matched_page
    
    # Determine which pages to search
    pages_to_search = []
    if last_matched_page > 0:
        # Search current page (p) and next page (p+1)
        pages_to_search = [last_matched_page, last_matched_page + 1]
    else:
        # First line - search page 1 and 2
        pages_to_search = [1, 2]
    
    # Filter to only existing pages
    pages_to_search = [p for p in pages_to_search if p in page_index]
    
    best_span = None
    best_score = 0
    best_index = -1
    matched_page = last_matched_page
    
    # Prepare search texts
    search_texts = []
    if is_table:
        # For table content, extract individual cells
        table_cells = extract_table_cell_content(md_line)
        search_texts.extend(table_cells)
        # Also include the full cleaned line
        search_texts.append(cleaned_md_line)
    else:
        search_texts = [cleaned_md_line]
    
    # Search only spans from specified pages
    for page_num in pages_to_search:
        for span_idx in page_index[page_num]:
            if span_idx in used_spans:
                continue
            
            span = spans[span_idx]
            span_text = span['text'].strip()
            
            if len(clean_text(span_text, is_table)) < 3:
                continue
            
            # Try matching against all search texts
            for search_text in search_texts:
                if not search_text:
                    continue
                
                # Calculate similarity score
                score = similarity_score(search_text, span_text)
                
                # Clean versions for containment check
                cleaned_search = clean_text(search_text, is_table)
                cleaned_span = clean_text(span_text, is_table)
                
                # Boost score if search text contains span text or vice versa
                if cleaned_search in cleaned_span or cleaned_span in cleaned_search:
                    score += 0.2
                
                # Boost score for exact matches after cleaning
                if cleaned_search == cleaned_span:
                    score = 1.0
                
                # For table cells, boost score if span text is contained in the cell
                if is_table and len(search_text) > 20:  # Longer table cell content
                    if cleaned_span in cleaned_search:
                        score = max(score, 0.8)
                
                if score > best_score and score >= min_similarity:
                    best_score = score
                    best_span = span
                    best_index = span_idx
                    matched_page = page_num
            
            # Early termination for high-confidence matches
            if best_score >= 0.9:
                break
        
        # If we found a very good match, don't search other pages
        if best_score >= 0.9:
            break
    
    if best_span and best_index != -1:
        used_spans.add(best_index)
        return best_span, matched_page
    
    return None, last_matched_page

def decode_font_flags(flags):
    """Decode PyMuPDF font flags and return both styles list and individual boolean flags"""
    styles = []
    is_bold = bool(flags & 16)        # FONTFLAG_BOLD = 16
    is_italic = bool(flags & 2)       # FONTFLAG_ITALIC = 2  
    is_monospace = bool(flags & 8)    # FONTFLAG_MONOSPACE = 8
    is_serifed = bool(flags & 4)      # FONTFLAG_SERIFED = 4
    is_superscript = bool(flags & 1)  # FONTFLAG_SUPERSCRIPT = 1
    
    if is_superscript: styles.append("superscript")
    if is_italic:      styles.append("italic") 
    if is_serifed:     styles.append("serifed")
    if is_monospace:   styles.append("monospace")
    if is_bold:        styles.append("bold")
    
    return styles if styles else ["normal"], is_bold, is_italic, is_monospace

def aggregate_md_to_spans(pdf_name: str):
    """
    Main function to aggregate MD lines with corresponding spans for a given PDF
    Args:
        pdf_name (str): Name of the PDF file (e.g., 'file06.pdf')
    """
    
    # Extract base filename without extension
    base_filename = pdf_name.replace('.pdf', '')
    
    # Define file paths
    md_file = f"../../data/md_files/{base_filename}.md"
    json_file = f"../../data/spans_output/spans_{pdf_name}.json"
    output_file = f"../../data/aggregator_output/aggregated_{pdf_name}.json"
    
    start_time = time.time()
    
    print(f"Loading data from {json_file}...")
    load_start = time.time()
    spans, page_index = load_json_data(json_file)
    load_end = time.time()
    print(f"  Data loaded in {load_end - load_start:.2f} seconds")
    print(f"  Loaded {len(spans)} spans across {len(page_index)} pages")
    
    print(f"Loading markdown from {md_file}...")
    md_start = time.time()
    md_lines = load_md_file(md_file)
    md_end = time.time()
    print(f"  Markdown loaded in {md_end - md_start:.2f} seconds")
    
    print(f"Processing {len(md_lines)} lines from markdown file...")
    process_start = time.time()
    
    aggregated_data = []
    used_spans = set()
    unmatched_lines = []
    last_matched_page = 0  # Track page of last successful match
    total_spans_searched = 0
    table_lines_processed = 0
    table_lines_matched = 0
    
    for line_num, md_line in enumerate(md_lines, 1):
        # Show progress for every 100 lines
        if line_num % 100 == 0:
            elapsed = time.time() - process_start
            rate = line_num / elapsed if elapsed > 0 else 0
            avg_spans_per_line = total_spans_searched / line_num if line_num > 0 else 0
            print(f"  Processed {line_num}/{len(md_lines)} lines ({rate:.1f} lines/sec, avg {avg_spans_per_line:.1f} spans/line)")
        
        md_line_clean = md_line.strip()
        
        # Skip empty lines - don't record them
        if not md_line_clean:
            continue
        
        # Check if this line is part of a table
        is_table = is_table_content(md_line_clean)
        if is_table:
            table_lines_processed += 1
        
        # Find best matching span with page optimization
        matching_span, last_matched_page = find_best_matching_span_optimized(
            md_line_clean, spans, page_index, used_spans, last_matched_page
        )
        
        # Count spans searched for performance monitoring
        pages_searched = []
        if last_matched_page > 0:
            pages_searched = [p for p in [last_matched_page, last_matched_page + 1] if p in page_index]
        else:
            pages_searched = [p for p in [1, 2] if p in page_index]
        
        spans_in_pages = sum(len(page_index[p]) for p in pages_searched)
        total_spans_searched += spans_in_pages
        
        if matching_span:
            if is_table:
                table_lines_matched += 1
                
            # Extract features from the matching span, handling fonts array
            fonts = matching_span.get("fonts", [])
            first_font = fonts[0] if fonts else {}
            
            # Decode font styles from font_flags
            font_flags = first_font.get("font_flags", 0)
            font_styles, is_bold, is_italic, is_monospace = decode_font_flags(font_flags)
            
            features = {
                "page_num": matching_span.get("page_num"),
                "column": matching_span.get("column"),
                "bbox": matching_span.get("bbox"),
                "font_name": first_font.get("font_name", ""),
                "font_size": first_font.get("font_size", 0),
                "font_styles": font_styles,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "is_monospace": is_monospace,
                "color": first_font.get("color", 0)
            }
            
            # Calculate confidence score
            confidence = similarity_score(clean_md_line(md_line_clean), matching_span["text"])
            
            aggregated_data.append({
                "line_number": line_num,
                "page_number": matching_span.get("page_num"),  # Added page_number field
                "is_in_table": is_table,  # Added is_in_table field
                "md_text_original": md_line_clean,
                "md_text_cleaned": clean_md_line(md_line_clean),
                "span_text": matching_span["text"],
                "matched_page": last_matched_page,
                "span_match": True,
                "features": features,
                "match_confidence": round(confidence, 3)
            })
        else:
            # No matching span found
            unmatched_lines.append({
                "line_number": line_num,
                "page_number": None,  # Added page_number field
                "is_in_table": is_table,  # Added is_in_table field
                "text_original": md_line_clean,
                "text_cleaned": clean_md_line(md_line_clean),
                "search_pages": pages_searched
            })
            
            aggregated_data.append({
                "line_number": line_num,
                "page_number": None,  # Added page_number field
                "is_in_table": is_table,  # Added is_in_table field
                "md_text_original": md_line_clean,
                "md_text_cleaned": clean_md_line(md_line_clean),
                "span_text": None,
                "matched_page": None,
                "span_match": False,
                "features": None,
                "match_confidence": None
            })
    
    process_end = time.time()
    processing_time = process_end - process_start
    
    # Create summary statistics
    total_lines = len([line for line in md_lines if line.strip()])  # Only count non-empty lines
    matched_lines = sum(1 for item in aggregated_data if item["span_match"])
    avg_spans_per_line = total_spans_searched / total_lines if total_lines > 0 else 0
    total_spans_available = len(spans)
    optimization_ratio = total_spans_available / avg_spans_per_line if avg_spans_per_line > 0 else 1
    
    summary = {
        "total_md_lines": total_lines,
        "matched_lines": matched_lines,
        "unmatched_lines": len(unmatched_lines),
        "table_lines_processed": table_lines_processed,
        "table_lines_matched": table_lines_matched,
        "table_match_percentage": round((table_lines_matched / table_lines_processed) * 100, 2) if table_lines_processed > 0 else 0,
        "match_percentage": round((matched_lines / total_lines) * 100, 2) if total_lines > 0 else 0,
        "total_spans_available": total_spans_available,
        "spans_used": len(used_spans),
        "processing_time_seconds": round(processing_time, 2),
        "lines_per_second": round(total_lines / processing_time, 2) if processing_time > 0 else 0,
        "avg_spans_searched_per_line": round(avg_spans_per_line, 1),
        "optimization_speedup": f"{optimization_ratio:.1f}x faster than full search",
        "pages_available": len(page_index)
    }
    
    # Prepare final output
    final_output = {
        "summary": summary,
        "aggregated_data": aggregated_data,
        "unmatched_lines": unmatched_lines
    }
    
    # Save to output file
    save_start = time.time()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    save_end = time.time()
    
    total_time = time.time() - start_time
    
    print(f"\nOptimized Aggregation completed!")
    print(f"Performance:")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Processing rate: {summary['lines_per_second']:.1f} lines/second")
    print(f"  Average spans searched per line: {summary['avg_spans_searched_per_line']}")
    print(f"  Optimization: {summary['optimization_speedup']}")
    
    print(f"\nResults:")
    print(f"  Total MD lines (non-empty): {summary['total_md_lines']}")
    print(f"  Matched lines: {summary['matched_lines']}")
    print(f"  Unmatched lines: {summary['unmatched_lines']}")
    print(f"  Match percentage: {summary['match_percentage']}%")
    print(f"  Table lines processed: {summary['table_lines_processed']}")
    print(f"  Table lines matched: {summary['table_lines_matched']} ({summary['table_match_percentage']}%)")
    print(f"  Spans used: {summary['spans_used']}/{summary['total_spans_available']}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"\nOutput saved to: {output_file}")
    
    if unmatched_lines:
        print(f"\nFirst 5 unmatched lines:")
        for i, line in enumerate(unmatched_lines[:5]):
            table_indicator = " (TABLE)" if line.get("is_in_table") else ""
            print(f"  Line {line['line_number']}{table_indicator}: {line['text_cleaned'][:100]}...")












