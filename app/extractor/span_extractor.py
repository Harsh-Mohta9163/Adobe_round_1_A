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
    output = []

    for page_num, page in enumerate(doc, 1):
        bboxes = column_boxes(page, footer_margin=0, no_image_text=True)
        page_columns = []
        for rect in bboxes:
            text_dict = page.get_text("dict", clip=rect)
            lines = []
            fonts = []
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_fonts = []
                        for span in line["spans"]:
                            line_text += span["text"]
                            line_fonts.append({
                                "font_name": span.get("font", ""),
                                "font_size": span.get("size", 0),
                                "font_flags": span.get("flags", 0)
                            })
                        if line_text.strip():
                            lines.append(line_text.strip())
                            fonts.append(line_fonts)
            page_columns.append({
                "bbox": list(rect),
                "text": "\n".join(lines),
                "fonts": fonts  # list of font info for each line
            })
        output.append({
            "page_num": page_num,
            "columns": page_columns
        })

    # Create split output directly without intermediate file
    split_output = []
    for page in output:
        page_num = page["page_num"]
        for col_idx, col in enumerate(page["columns"]):
            bbox = col["bbox"]
            lines = col["text"].split('\n')
            fonts = col.get("fonts", [])
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    font_info = fonts[i] if i < len(fonts) else []
                    split_output.append({
                        "page_num": page_num,
                        "column": col_idx,
                        "bbox": bbox,
                        "text": line,
                        "fonts": font_info
                    })

    with open(f"../../data/spans_output/spans_{pdf_name}.json", "w", encoding="utf-8") as f:
        json.dump(split_output, f, ensure_ascii=False, indent=2)