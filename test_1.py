"""
Convert document to Markdown with OCR support for image-based text.

First install: pip install pytesseract pillow
Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
"""

import pymupdf4llm
import pytesseract
from PIL import Image
import os
import json
import sys
import re


def extract_text_from_images(image_folder):
    """Extract text from saved images using OCR."""
    ocr_results = {}
    
    if not os.path.exists(image_folder):
        return ocr_results
        
    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            try:
                # Use Tesseract to extract text
                text = pytesseract.image_to_string(
                    Image.open(image_path),
                    config='--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
                )
                if text.strip():
                    ocr_results[filename] = text.strip()
                    print(f"Extracted text from {filename}: {len(text.strip())} characters")
            except Exception as e:
                print(f"OCR failed for {filename}: {e}")
    
    return ocr_results


def combine_text_and_images(md_data, ocr_results):
    """Combine markdown data with OCR results."""
    for page in md_data:
        if 'text' in page:
            # Find image references in the text
            image_refs = re.findall(r'!\[\]\(([^)]+)\)', page['text'])
            
            # Replace image references with OCR text
            enhanced_text = page['text']
            for img_path in image_refs:
                img_filename = os.path.basename(img_path)
                if img_filename in ocr_results:
                    ocr_text = ocr_results[img_filename]
                    # Replace image reference with OCR text
                    enhanced_text = enhanced_text.replace(
                        f"![]({img_path})", 
                        f"\n## Extracted Text from {img_filename}\n\n{ocr_text}\n"
                    )
            
            page['enhanced_text'] = enhanced_text
            page['ocr_extractions'] = {
                img: ocr_results[img] 
                for img in ocr_results 
                if any(img in ref for ref in image_refs)
            }
    
    return md_data


def make_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    else:
        return str(obj)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_with_ocr.py input.pdf")
        return
    
    filename = sys.argv[1]
    outname = filename.replace(".pdf", "_with_ocr.json")
    image_folder = "./images"
    
    print(f"Processing {filename}...")
    
    # Extract using pymupdf4llm
    md_data = pymupdf4llm.to_markdown(
        filename, 
        page_chunks=True,
        write_images=True,
        force_text=True,
        image_path=image_folder,
        dpi=300
    )
    
    print(f"Extracted {len(md_data)} pages")
    
    # Perform OCR on saved images
    print("Performing OCR on extracted images...")
    ocr_results = extract_text_from_images(image_folder)
    
    # Combine results
    enhanced_data = combine_text_and_images(md_data, ocr_results)
    
    # Save results
    result = {
        "original_extraction": md_data,
        "ocr_results": ocr_results,
        "enhanced_pages": enhanced_data
    }
    
    with open(outname, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=make_serializable)
    
    print(f"Results saved to {outname}")
    print(f"OCR extracted text from {len(ocr_results)} images")


if __name__ == "__main__":
    main()