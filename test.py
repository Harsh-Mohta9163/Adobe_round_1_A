"""
TOC Generator using pypdfium2 to extract content based on PDF bookmarks.
This extracts the actual Table of Contents from PDF bookmarks/outline.
"""

import pypdfium2
import json
import os
from typing import List, Dict, Any


# ========================= CONFIGURATION =========================
# Just change the PDF names here to test different files
PDF_FILES_TO_TEST = [
    "file01.pdf",
    "file02.pdf", 
    "file03.pdf",
    "file04.pdf",
    "file05.pdf",
    "security.pdf",
    "file02_1.pdf",
]
# ================================================================


def extract_toc_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract Table of Contents from PDF using bookmarks/outline.
    """
    print(f"Extracting TOC from: {pdf_path}")
    
    try:
        # Open PDF document
        pdf = pypdfium2.PdfDocument(pdf_path)
        
        # Get total pages first before closing
        total_pages = len(pdf)
        
        # Get bookmarks/outline
        toc_items = []
        for bookmark in pdf.get_toc():
            toc_items.append(bookmark)
        
        if not toc_items:
            print("  No bookmarks/TOC found in PDF")
            pdf.close()
            return {
                'has_toc': False,
                'toc_items': [],
                'total_pages': total_pages,
                'message': 'No bookmarks found in PDF'
            }
        
        print(f"  Found {len(toc_items)} TOC items")
        
        # Extract TOC structure and content
        toc_data = []
        
        for i, bookmark in enumerate(toc_items):
            print(f"  Processing bookmark {i+1}/{len(toc_items)}: '{bookmark.title}'")
            
            # Get content for this bookmark
            content = get_bookmark_content(pdf, toc_items, i)
            
            toc_entry = {
                'index': i,
                'level': bookmark.level,
                'title': bookmark.title,
                'page_index': bookmark.page_index,
                'page_number': bookmark.page_index + 1,  # 1-based page numbering
                'is_closed': bookmark.is_closed,
                'n_kids': bookmark.n_kids,
                'view_mode': bookmark.view_mode,
                'view_pos': bookmark.view_pos,
                'content_preview': content[:500] + "..." if len(content) > 500 else content,
                'content_length': len(content),
                'full_content': content
            }
            
            toc_data.append(toc_entry)
        
        pdf.close()
        
        result = {
            'has_toc': True,
            'toc_items': toc_data,
            'total_toc_items': len(toc_data),
            'total_pages': total_pages,
            'pdf_path': pdf_path
        }
        
        return result
        
    except Exception as e:
        print(f"  Error extracting TOC: {str(e)}")
        return {
            'has_toc': False,
            'error': str(e),
            'pdf_path': pdf_path
        }


def get_bookmark_content(pdf: pypdfium2.PdfDocument, bookmarks: List, index: int) -> str:
    """
    Extract content for a specific bookmark.
    """
    current_bookmark = bookmarks[index]
    next_index = index + 1
    
    # Determine the end boundary
    if next_index >= len(bookmarks):
        # Last bookmark - goes to end of document
        last_page = len(pdf) - 1
        next_bookmark = None
    else:
        # Next bookmark determines the end
        next_bookmark = bookmarks[next_index]
        last_page = next_bookmark.page_index
    
    first_page = current_bookmark.page_index
    content = []
    
    try:
        # Ensure we don't go beyond document bounds
        max_pages = len(pdf)
        last_page = min(last_page, max_pages - 1)
        
        for page_index in range(first_page, last_page + 1):
            if page_index >= max_pages:
                break
                
            page = pdf[page_index]
            text_page = page.get_textpage()
            
            # For now, just get all text from the page
            # TODO: Could add boundary detection later
            text = text_page.get_text_range()
            
            if text.strip():
                content.append(text.strip())
                
    except Exception as e:
        print(f"    Warning: Error extracting content for '{current_bookmark.title}': {str(e)}")
        content.append(f"[Error extracting content: {str(e)}]")
    
    return '\n'.join(content)


def analyze_toc_structure(toc_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the TOC structure for insights.
    """
    if not toc_data:
        return {'analysis': 'No TOC data to analyze'}
    
    levels = [item['level'] for item in toc_data]
    content_lengths = [item['content_length'] for item in toc_data if item['content_length'] > 0]
    
    analysis = {
        'total_sections': len(toc_data),
        'min_level': min(levels),
        'max_level': max(levels),
        'level_distribution': {level: levels.count(level) for level in set(levels)},
        'avg_content_length': sum(content_lengths) / len(content_lengths) if content_lengths else 0,
        'sections_with_content': len(content_lengths),
        'sections_without_content': len(toc_data) - len(content_lengths),
        'page_span': {
            'first_page': min(item['page_number'] for item in toc_data),
            'last_page': max(item['page_number'] for item in toc_data)
        }
    }
    
    return analysis


def save_toc_data(toc_result: Dict, output_path: str):
    """
    Save TOC data to JSON file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a version without full content for the main file (too large)
    toc_summary = toc_result.copy()
    if 'toc_items' in toc_summary:
        for item in toc_summary['toc_items']:
            if 'full_content' in item:
                del item['full_content']  # Remove full content to keep file manageable
    
    # Add analysis
    if toc_result.get('has_toc') and toc_result.get('toc_items'):
        toc_summary['structure_analysis'] = analyze_toc_structure(toc_result['toc_items'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(toc_summary, f, indent=2, ensure_ascii=False)
    
    print(f"  TOC data saved to: {output_path}")


def test_toc_extractor(pdf_name: str):
    """
    Test the TOC extractor with a PDF from the input folder.
    """
    # Define paths
    input_pdf_path = f"data/input/{pdf_name}"
    base_filename = pdf_name.replace('.pdf', '')
    output_json_path = f"data/toc_output/toc_{base_filename}.json"
    
    if not os.path.exists(input_pdf_path):
        print(f"❌ Error: PDF file not found: {input_pdf_path}")
        return
    
    print(f"Testing TOC extraction for: {pdf_name}")
    print("=" * 50)
    
    # Extract TOC
    toc_result = extract_toc_from_pdf(input_pdf_path)
    
    # Display results
    if toc_result.get('has_toc'):
        print(f"\n✅ TOC extraction successful!")
        print(f"  Total TOC items: {toc_result['total_toc_items']}")
        print(f"  Total pages: {toc_result['total_pages']}")
        
        print(f"\n📋 TOC Structure:")
        for item in toc_result['toc_items']:
            indent = "  " * (item['level'] - 1)
            print(f"{indent}📄 Level {item['level']}: '{item['title']}'")
            print(f"{indent}   📍 Page {item['page_number']} | Content: {item['content_length']} chars")
            if item['content_preview']:
                preview = item['content_preview'].replace('\n', ' ')[:100]
                print(f"{indent}   💬 Preview: {preview}...")
            print()
        
        # Show analysis
        analysis = analyze_toc_structure(toc_result['toc_items'])
        print(f"📊 Structure Analysis:")
        print(f"  • Total sections: {analysis['total_sections']}")
        print(f"  • Level range: {analysis['min_level']} to {analysis['max_level']}")
        print(f"  • Sections with content: {analysis['sections_with_content']}")
        print(f"  • Average content length: {analysis.get('avg_content_length', 0):.0f} characters")
        print(f"  • Page span: {analysis['page_span']['first_page']} to {analysis['page_span']['last_page']}")
        
    else:
        print(f"\n❌ No TOC found in PDF")
        if 'error' in toc_result:
            print(f"  Error: {toc_result['error']}")
        else:
            print(f"  Reason: {toc_result.get('message', 'Unknown')}")
    
    # Save results
    save_toc_data(toc_result, output_json_path)
    
    print(f"\n📁 Results saved to: {output_json_path}")


if __name__ == "__main__":
    print("🔍 Testing TOC Extractor with PDF Bookmarks")
    print("=" * 60)
    
    # Test all PDFs in the configuration array
    for pdf_name in PDF_FILES_TO_TEST:
        print(f"\n{'='*60}")
        test_toc_extractor(pdf_name)
        print(f"{'='*60}")
    
    print(f"\n🎉 Completed testing {len(PDF_FILES_TO_TEST)} PDF files!")