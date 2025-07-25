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
    all_output = []  # ← Renamed for clarity

    for page_num, page in enumerate(doc, 1):
        bboxes = column_boxes(page, footer_margin=0, header_margin=0, no_image_text=False)

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
                            all_output.append({  # ← Add to main list
                                "page_num": page_num,
                                "column": col_idx,
                                "bbox": list(line_bbox),
                                "text": line_text.strip(),
                                "fonts": line_fonts
                            })
    
    doc.close()  # ← Good practice
    
    # Save all accumulated data
    with open(f"../../data/spans_output/spans_{pdf_name}.json", "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    
    return all_output  # ← Return for potential use