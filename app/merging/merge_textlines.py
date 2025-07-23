import pandas as pd
import ast
import os
import glob
from textblob import TextBlob
import re

def finalize_block(block_parts):
    """
    Processes a list of block parts to create a single, finalized textblock dictionary.
    """
    if not block_parts:
        return None

    full_text = ' '.join(part['text'] for part in block_parts).strip()
    all_bboxes = [part.get('bbox', [0, 0, 0, 0]) for part in block_parts]
    all_font_sizes = [part.get('font_size', 12.0) for part in block_parts]

    avg_font_size = round(sum(all_font_sizes) / len(all_font_sizes), 2) if all_font_sizes else 12.0
    
    min_x0 = min(b[0] for b in all_bboxes) if all_bboxes else 0
    min_y0 = min(b[1] for b in all_bboxes) if all_bboxes else 0
    max_x1 = max(b[2] for b in all_bboxes) if all_bboxes else 0
    max_y1 = max(b[3] for b in all_bboxes) if all_bboxes else 0
    union_bbox = [min_x0, min_y0, max_x1, max_y1]
    
    word_count = len(full_text.split())
    is_all_caps = 1 if full_text.isupper() and any(c.isalpha() for c in full_text) else 0
    char_density = len(full_text.replace(" ", "")) / ((max_x1 - min_x0) * (max_y1 - min_y0)) if (max_x1 > min_x0 and max_y1 > min_y0) else 0.1
    
    # Base block data
    block_data = {
        'text': full_text,
        'bbox': union_bbox,
        'page_number': block_parts[0].get('pagenum', -1),
        'avg_font_size': avg_font_size,
        'word_count': word_count,
        'is_all_caps': is_all_caps,
        'char_density': char_density,
    }
    
    block_data['title_label'] = ''
    return block_data

def detect_column_names(df):
    """Detect the correct column names for text and other features."""
    columns = df.columns.tolist()
    detected_cols = {
        'text_a': None, 'text_b': None, 'page_number_a': None, 'page_number_b': None
    }
    for col in columns:
        col_lower = col.lower()
        if 'text_a' in col_lower and 'span' not in col_lower: detected_cols['text_a'] = col
        elif 'text_b' in col_lower and 'span' not in col_lower: detected_cols['text_b'] = col
        elif 'page_number_a' in col_lower: detected_cols['page_number_a'] = col
        elif 'page_number_b' in col_lower: detected_cols['page_number_b'] = col
    return detected_cols

def merge_textlines(input_csv_path: str, output_csv_path: str):
    """
    Merges text lines based on pairwise labels and page numbers from a CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"❌ Error loading {input_csv_path}: {e}")
        return False

    if df.empty:
        print(f"⚠️ Warning: Input CSV is empty.")
        return False

    cols = detect_column_names(df)
    for col_name, col_key in cols.items():
        if col_key is None:
            print(f"❌ Required column '{col_name}' not found in the CSV.")
            return False

    # --- Create a map of all unique lines and their properties ---
    lines_map = {}
    for _, row in df.iterrows():
        for line_type in ['a', 'b']:
            text = str(row[cols[f'text_{line_type}']])
            if text not in lines_map:
                lines_map[text] = {
                    'text': text,
                    'pagenum': row[cols[f'page_number_{line_type}']],
                }

    # --- New Merging Logic: Iterate through pairs ---
    text_blocks = []
    if df.empty:
        return True

    # Start the first block with the first line of the first pair
    first_line_text = str(df.iloc[0][cols['text_a']])
    current_block_parts = [lines_map[first_line_text]]

    for _, row in df.iterrows():
        line_a_text = str(row[cols['text_a']])
        line_b_text = str(row[cols['text_b']])
        
        # Ensure the sequence is maintained
        if not current_block_parts or current_block_parts[-1]['text'] != line_a_text:
            if current_block_parts:
                text_blocks.append(finalize_block(current_block_parts))
            current_block_parts = [lines_map.get(line_a_text)]
            if not current_block_parts[0]: # handle case where line_a_text may not be in map
                continue

        # Decide whether to merge line_b
        should_merge = (row['label'] == 1) and (row[cols['page_number_a']] == row[cols['page_number_b']])

        if should_merge:
            current_block_parts.append(lines_map.get(line_b_text))
        else:
            text_blocks.append(finalize_block(current_block_parts))
            current_block_parts = [lines_map.get(line_b_text)]

        # Clean Nones from current_block_parts
        current_block_parts = [part for part in current_block_parts if part is not None]

    # Add the final block
    if current_block_parts:
        text_blocks.append(finalize_block(current_block_parts))

    # --- Create and save the final DataFrame ---
    # **FIXED**: Directly convert the list of blocks without removing duplicates
    output_df = pd.DataFrame([block for block in text_blocks if block])
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Success! Merged into {len(output_df)} text blocks.")
    print(f"📁 Output saved to: '{output_csv_path}'")
    
    return True

if __name__ == '__main__':
    # --- CONFIGURATION ---
    INPUT_FOLDER = '../../data/labelled_textlines'
    OUTPUT_FOLDER = '../../data/merged_textblocks_gt'
    # -------------------

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    csv_pattern = os.path.join(INPUT_FOLDER, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{INPUT_FOLDER}'")
        exit()
    
    print(f"🔍 Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    successful_files = 0
    failed_files = []
    
    for input_csv in csv_files:
        filename = os.path.basename(input_csv)
        
        if 'textlines_ground_truth_' in filename:
            output_filename = filename.replace('textlines_ground_truth_', 'merged_textblocks_')
        else:
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"merged_textblocks_{name_without_ext}.csv"
        
        output_csv = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"\n📄 Processing: {filename}")
        print(f"📤 Output: {output_filename}")
        print("-" * 60)
        
        success = merge_textlines(input_csv, output_csv)
        
        if success:
            successful_files += 1
        else:
            failed_files.append(filename)
    
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_files}/{len(csv_files)} files")
    
    if failed_files:
        print(f"❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")