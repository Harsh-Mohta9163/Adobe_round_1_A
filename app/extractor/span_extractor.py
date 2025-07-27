import pymupdf
import json
import os # <-- Added for os.path.basename
from multi_column import column_boxes

# CHANGED: The function now accepts full paths as arguments
def extract_columns_and_split(input_pdf_path, output_json_path):
    """
    Extract columns and split lines from a PDF, saving a JSON file.
    Accepts full input and output paths.
    """
    # CHANGED: Use the provided input_pdf_path argument
    doc = pymupdf.open(input_pdf_path)
    all_output = []
    
    # This is used for your print statements
    pdf_name = os.path.basename(input_pdf_path)

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
                            
                            if line_bbox is None:
                                line_bbox = span["bbox"]
                        
                        if line_text.strip() and line_bbox:
                            is_in_table = is_bbox_in_tables(line_bbox, table_bboxes)
                            
                            all_output.append({
                                "page_num": page_num,
                                "column": col_idx,
                                "bbox": list(line_bbox),
                                "text": line_text.strip(),
                                "fonts": line_fonts,
                                "is_in_table": is_in_table
                            })
    
    doc.close()
    
    # CHANGED: Use the provided output_json_path argument
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    
    return True # Indicate success

def is_bbox_in_tables(line_bbox, table_bboxes, overlap_threshold=0.5):
    # ... (this function is unchanged) ...
    if not table_bboxes:
        return False
    
    line_x0, line_y0, line_x1, line_y1 = line_bbox
    line_area = (line_x1 - line_x0) * (line_y1 - line_y0)
    
    if line_area <= 0:
        return False
    
    for table_bbox in table_bboxes:
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        intersect_x0 = max(line_x0, table_x0)
        intersect_y0 = max(line_y0, table_y0)
        intersect_x1 = min(line_x1, table_x1)
        intersect_y1 = min(line_y1, table_y1)
        
        if intersect_x0 < intersect_x1 and intersect_y0 < intersect_y1:
            intersect_area = (intersect_x1 - intersect_x0) * (intersect_y1 - intersect_y0)
            overlap_ratio = intersect_area / line_area
            
            if overlap_ratio >= overlap_threshold:
                return True
    
    return False

# This block allows you to test this script by itself
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python span_extractor.py <path_to_input.pdf> <path_to_output.json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    extract_columns_and_split(input_path, output_path)