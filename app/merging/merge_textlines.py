import pandas as pd
import ast
import os
import glob
from textblob import TextBlob
import re

def calculate_verb_ratio(text):
    """Calculates the ratio of verbs to total words in a text string."""
    if not text:
        return 0.0
    try:
        blob = TextBlob(text)
        words = blob.words
        if not words:
            return 0.0
        verbs = [word for word, tag in blob.tags if tag.startswith('VB')]
        return len(verbs) / len(words)
    except:
        return 0.0

def calculate_capitalized_ratio(text):
    """Calculates the ratio of words starting with a capital letter."""
    if not text:
        return 0.0
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    capitalized_words = [word for word in words if word[0].isupper()]
    return len(capitalized_words) / len(words)

def get_feature_value(row, feature_name, default_value=0):
    """Safely get feature value from row with fallback to default."""
    try:
        if feature_name in row and pd.notna(row[feature_name]):
            return row[feature_name]
        else:
            return default_value
    except:
        return default_value

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
    
    # Calculate new features
    ratio_of_verbs = calculate_verb_ratio(full_text)
    ratio_capitalized = calculate_capitalized_ratio(full_text)
    
    # Check if textblock ends with colon
    ends_with_colon = 1 if full_text.strip().endswith(':') else 0
    
    # Check if textblock is bold (majority rule)
    bold_count = sum(1 for part in block_parts if part.get('is_bold', 0))
    is_bold = 1 if bold_count > len(block_parts) / 2 else 0
    
    # Calculate normalized vertical gap (average gap between parts)
    normalized_vertical_gap = 0.0
    if len(block_parts) > 1:
        gaps = []
        for i in range(len(block_parts) - 1):
            gap = block_parts[i+1]['bbox'][1] - block_parts[i]['bbox'][3]  # y_start_next - y_end_current
            gaps.append(gap)
        normalized_vertical_gap = sum(gaps) / len(gaps) if gaps else 0.0
    
    # Calculate indentation change
    indentation_change = 0.0
    if len(block_parts) > 1:
        first_indent = block_parts[0]['bbox'][0]  # x0 of first part
        last_indent = block_parts[-1]['bbox'][0]  # x0 of last part
        indentation_change = abs(last_indent - first_indent)
    
    # Check if all parts have same alignment (x0 coordinates)
    same_alignment = 1 if len(set(part['bbox'][0] for part in block_parts)) == 1 else 0
    
    # Check if text is centered (simplified check)
    is_centered_A = 1 if len(block_parts) == 1 and (union_bbox[2] - union_bbox[0]) < 200 else 0
    
    # Font size difference between first and last part
    font_size_diff = 0.0
    if len(block_parts) > 1:
        font_size_diff = abs(block_parts[-1]['font_size'] - block_parts[0]['font_size'])
    
    # Check if all parts have same font size
    same_font = 1 if len(set(part['font_size'] for part in block_parts)) == 1 else 0
    
    # Check formatting properties (using first part for _A features)
    is_bold_A = block_parts[0].get('is_bold', 0) if block_parts else 0
    is_italic_A = block_parts[0].get('is_italic', 0) if block_parts else 0
    is_monospace_A = block_parts[0].get('is_monospace', 0) if block_parts else 0
    
    # Check if all parts have same formatting
    same_bold = 1 if len(set(part.get('is_bold', 0) for part in block_parts)) == 1 else 0
    same_italic = 1 if len(set(part.get('is_italic', 0) for part in block_parts)) == 1 else 0
    same_monospace = 1 if len(set(part.get('is_monospace', 0) for part in block_parts)) == 1 else 0
    
    # Check line ending and starting patterns
    line_a_ends_punctuation = 1 if block_parts and block_parts[0]['text'].strip() and block_parts[0]['text'].strip()[-1] in '.!?:;,' else 0
    line_b_starts_lowercase = 1 if len(block_parts) > 1 and block_parts[1]['text'].strip() and block_parts[1]['text'].strip()[0].islower() else 0
    
    # Check if lines are in rectangles/tables (assuming these are in additional_features)
    is_linea_in_rectangle = block_parts[0].get('in_rectangle', 0) if block_parts else 0
    is_lineb_in_rectangle = block_parts[1].get('in_rectangle', 0) if len(block_parts) > 1 else 0
    both_in_table = 1 if all(part.get('in_table', 0) for part in block_parts) else 0
    neither_in_table = 1 if not any(part.get('in_table', 0) for part in block_parts) else 0
    
    # Check if lines are hashed (assuming these are in additional_features)
    is_linea_hashed = block_parts[0].get('is_hashed', 0) if block_parts else 0
    is_lineb_hashed = block_parts[1].get('is_hashed', 0) if len(block_parts) > 1 else 0
    both_hashed = 1 if all(part.get('is_hashed', 0) for part in block_parts) else 0
    neither_hashed = 1 if not any(part.get('is_hashed', 0) for part in block_parts) else 0
    
    # Base block data
    block_data = {
        'text': full_text,
        'bbox': union_bbox,
        'page_number': block_parts[0].get('pagenum', -1),
        'avg_font_size': avg_font_size,
        'word_count': word_count,
        'is_all_caps': is_all_caps,
        'char_density': char_density,
        'ratio_of_verbs': ratio_of_verbs,
        'ratio_capitalized': ratio_capitalized,
        'ends_with_colon': ends_with_colon,
        'is_bold': is_bold,
        'normalized_vertical_gap': normalized_vertical_gap,
        'indentation_change': indentation_change,
        'same_alignment': same_alignment,
        'is_centered_A': is_centered_A,
        'font_size_diff': font_size_diff,
        'same_font': same_font,
        'is_bold_A': is_bold_A,
        'is_italic_A': is_italic_A,
        'is_monospace_A': is_monospace_A,
        'same_bold': same_bold,
        'same_italic': same_italic,
        'same_monospace': same_monospace,
        'line_a_ends_punctuation': line_a_ends_punctuation,
        'line_b_starts_lowercase': line_b_starts_lowercase,
        'is_linea_in_rectangle': is_linea_in_rectangle,
        'is_lineb_in_rectangle': is_lineb_in_rectangle,
        'both_in_table': both_in_table,
        'neither_in_table': neither_in_table,
        'is_linea_hashed': is_linea_hashed,
        'is_lineb_hashed': is_lineb_hashed,
        'both_hashed': both_hashed,
        'neither_hashed': neither_hashed,
        'title_label': ''
    }
    
    return block_data

def detect_column_names(df):
    """Detect the correct column names for text and other features."""
    columns = df.columns.tolist()
    detected_cols = {
        'text_a': None, 'text_b': None, 'page_number_a': None, 'page_number_b': None,
        'font_size_a': None, 'font_size_b': None, 'bbox_a': None, 'bbox_b': None
    }
    for col in columns:
        col_lower = col.lower()
        if 'text_a' in col_lower and 'span' not in col_lower: detected_cols['text_a'] = col
        elif 'text_b' in col_lower and 'span' not in col_lower: detected_cols['text_b'] = col
        elif 'page_number_a' in col_lower: detected_cols['page_number_a'] = col
        elif 'page_number_b' in col_lower: detected_cols['page_number_b'] = col
        elif 'font_size_a' in col_lower: detected_cols['font_size_a'] = col
        elif 'font_size_b' in col_lower: detected_cols['font_size_b'] = col
        elif 'bbox_a' in col_lower: detected_cols['bbox_a'] = col
        elif 'bbox_b' in col_lower: detected_cols['bbox_b'] = col
    return detected_cols

def parse_bbox(bbox_str):
    """Parse bbox string to list of coordinates."""
    try:
        if isinstance(bbox_str, str):
            # Remove brackets and split by comma
            bbox_str = bbox_str.strip('[](){}')
            coords = [float(x.strip()) for x in bbox_str.split(',')]
            return coords if len(coords) == 4 else [0, 0, 100, 20]
        elif isinstance(bbox_str, list):
            return bbox_str
        else:
            return [0, 0, 100, 20]
    except:
        return [0, 0, 100, 20]

def merge_textlines(input_csv_path: str, output_csv_path: str):
    """
    Merges text lines based on pairwise labels and page numbers from a CSV file.
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("Could not decode file with any supported encoding")
            
    except Exception as e:
        print(f"❌ Error loading {input_csv_path}: {e}")
        return False

    if df.empty:
        print(f"⚠️ Warning: Input CSV is empty.")
        return False

    cols = detect_column_names(df)
    # Check only required columns
    required_cols = ['text_a', 'text_b', 'page_number_a', 'page_number_b']
    for col_name in required_cols:
        if cols[col_name] is None:
            print(f"❌ Required column '{col_name}' not found in the CSV.")
            return False

    # --- Create a map of all unique lines and their properties ---
    lines_map = {}
    for _, row in df.iterrows():
        for line_type in ['a', 'b']:
            text = str(row[cols[f'text_{line_type}']])
            if text not in lines_map:
                # Parse bbox if available
                bbox = [0, 0, 100, 20]  # default
                if cols[f'bbox_{line_type}'] and cols[f'bbox_{line_type}'] in row:
                    bbox = parse_bbox(row[cols[f'bbox_{line_type}']])
                
                # Get font size if available
                font_size = 12.0  # default
                if cols[f'font_size_{line_type}'] and cols[f'font_size_{line_type}'] in row:
                    try:
                        font_size = float(row[cols[f'font_size_{line_type}']])
                    except:
                        font_size = 12.0
                
                lines_map[text] = {
                    'text': text,
                    'pagenum': row[cols[f'page_number_{line_type}']],
                    'bbox': bbox,
                    'font_size': font_size,
                    # Add additional features if they exist in the row
                    'is_bold': get_feature_value(row, f'is_bold_{line_type}', 0),
                    'is_italic': get_feature_value(row, f'is_italic_{line_type}', 0),
                    'is_monospace': get_feature_value(row, f'is_monospace_{line_type}', 0),
                    'in_rectangle': get_feature_value(row, f'in_rectangle_{line_type}', 0),
                    'in_table': get_feature_value(row, f'in_table_{line_type}', 0),
                    'is_hashed': get_feature_value(row, f'is_hashed_{line_type}', 0),
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