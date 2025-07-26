import pymupdf
import json
from multi_column import column_boxes

def extract_columns_and_split(pdf_name):
    """
    Extract columns and split lines from a PDF, saving JSON file:
    - ../../data/spans_output/spans_{pdf_name}.json
    Args:
        pdf_name (str): Name of the PDF file (e.g., 'file03.pdf')
    """
    doc = pymupdf.open(f"../../data/input/{pdf_name}")
    all_output = []

    for page_num, page in enumerate(doc, 1):
        bboxes = column_boxes(page, footer_margin=0, header_margin=0, no_image_text=False)
        
        # Detect tables on this page
        tables = page.find_tables()
        table_bboxes = []
        
        # Extract table bounding boxes
        for table in tables:
            table_bbox = table.bbox  # (x0, y0, x1, y1)
            table_bboxes.append(table_bbox)
        
        print(f"Page {page_num}: Found {len(table_bboxes)} tables")

        for col_idx, rect in enumerate(bboxes):
            text_dict = page.get_text("dict", clip=rect)
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_fonts = []
                        line_bbox = None
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            line_fonts.append({
                                "font_name": span.get("font", ""),
                                "font_size": span.get("size", 0),
                                "font_flags": span.get("flags", 0)
                            })
                            
                            # Get the actual line bbox from the first span
                            if line_bbox is None:
                                line_bbox = span["bbox"]  # [x0, y0, x1, y1]
                        
                        if line_text.strip() and line_bbox:
                            # Check if this line is inside any table
                            is_in_table = is_bbox_in_tables(line_bbox, table_bboxes)
                            
                            all_output.append({
                                "page_num": page_num,
                                "column": col_idx,
                                "bbox": list(line_bbox),
                                "text": line_text.strip(),
                                "fonts": line_fonts,
                                "is_in_table": is_in_table  # ← New field
                            })
    
    doc.close()
    
    # Save all accumulated data
    with open(f"../../data/spans_output/spans_{pdf_name}.json", "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    
    return all_output

def is_bbox_in_tables(line_bbox, table_bboxes, overlap_threshold=0.5):
    """
    Check if a line bbox overlaps with any table bbox
    
    Args:
        line_bbox: [x0, y0, x1, y1] of the text line
        table_bboxes: List of table bboxes [(x0, y0, x1, y1), ...]
        overlap_threshold: Minimum overlap ratio to consider as "in table"
    
    Returns:
        bool: True if line is inside a table
    """
    if not table_bboxes:
        return False
    
    line_x0, line_y0, line_x1, line_y1 = line_bbox
    line_area = (line_x1 - line_x0) * (line_y1 - line_y0)
    
    if line_area <= 0:
        return False
    
    for table_bbox in table_bboxes:
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        
        # Calculate intersection
        intersect_x0 = max(line_x0, table_x0)
        intersect_y0 = max(line_y0, table_y0)
        intersect_x1 = min(line_x1, table_x1)
        intersect_y1 = min(line_y1, table_y1)
        
        # Check if there's an intersection
        if intersect_x0 < intersect_x1 and intersect_y0 < intersect_y1:
            intersect_area = (intersect_x1 - intersect_x0) * (intersect_y1 - intersect_y0)
            overlap_ratio = intersect_area / line_area
            
            if overlap_ratio >= overlap_threshold:
                return True
    
    return False