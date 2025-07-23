import pandas as pd
import ast
import os
import glob
from textblob import TextBlob
import re

def calculate_verb_ratio(text):
    """Calculates the ratio of verbs to total words in a text string."""
    if not text:
        return 0
    try:
        blob = TextBlob(text)
        words = blob.words
        if not words:
            return 0
        verbs = [word for word, tag in blob.tags if tag.startswith('VB')]
        return len(verbs) / len(words)
    except:
        return 0

def calculate_capitalized_ratio(text):
    """Calculates the ratio of words starting with a capital letter."""
    if not text:
        return 0
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0
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
    ratio_of_verbs = calculate_verb_ratio(full_text)
    ratio_capitalized = calculate_capitalized_ratio(full_text)
    
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
        'ratio_capitalized': ratio_capitalized
    }
    
    # Add additional features from the source file
    if block_parts[0].get('additional_features'):
        first_part_features = block_parts[0]['additional_features']
        for feature_name in first_part_features:
            # For numeric features, calculate the average across all parts in the block
            if isinstance(first_part_features[feature_name], (int, float)):
                all_values = [part['additional_features'][feature_name] for part in block_parts if feature_name in part.get('additional_features', {})]
                block_data[feature_name] = round(sum(all_values) / len(all_values), 4) if all_values else 0
            else:
                # For non-numeric features, take the value from the first part
                block_data[feature_name] = first_part_features[feature_name]

    block_data['title_label'] = ''
    return block_data

def detect_column_names(df):
    """Detect the correct column names for text and other features."""
    columns = df.columns.tolist()
    detected_cols = {
        'text_a': None, 'text_b': None, 'page_number_a': None, 'page_number_b': None,
        'font_size_a': None, 'font_size_b': None
    }
    for col in columns:
        col_lower = col.lower()
        if 'text_a' in col_lower and 'span' not in col_lower: detected_cols['text_a'] = col
        elif 'text_b' in col_lower and 'span' not in col_lower: detected_cols['text_b'] = col
        elif 'page_number_a' in col_lower: detected_cols['page_number_a'] = col
        elif 'page_number_b' in col_lower: detected_cols['page_number_b'] = col
        elif 'font_size_a' in col_lower: detected_cols['font_size_a'] = col
        elif 'font_size_b' in col_lower: detected_cols['font_size_b'] = col
    return detected_cols

def get_additional_features(df_columns):
    """Get additional feature columns to be carried over."""
    # Define columns that are handled specially or should be excluded
    exclude_list = [
        'text_a', 'span_text_a', 'text_b', 'span_text_b', 'label',
        'page_number_a', 'page_number_b', 'font_size_a', 'font_size_b'
    ]
    
    additional_features = []
    for col in df_columns:
        if col not in exclude_list and not col.endswith(('_b', '_B')):
            additional_features.append(col)
    return additional_features

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

    additional_feature_cols = get_additional_features(df.columns)

    # --- Create a map of all unique lines and their properties ---
    lines_map = {}
    for _, row in df.iterrows():
        for line_type in ['a', 'b']:
            text = str(row[cols[f'text_{line_type}']])
            if text not in lines_map:
                lines_map[text] = {
                    'text': text,
                    'pagenum': row[cols[f'page_number_{line_type}']],
                    'font_size': row[cols[f'font_size_{line_type}']],
                    'bbox': [0,0,100,20], # Placeholder, recalculated in finalize
                    'additional_features': {f: get_feature_value(row, f) for f in additional_feature_cols}
                }

    # --- New Merging Logic: Iterate through pairs ---
    text_blocks = []
    if df.empty:
        return True

    first_line_text = str(df.iloc[0][cols['text_a']])
    current_block_parts = [lines_map[first_line_text]]

    for _, row in df.iterrows():
        line_a_text = str(row[cols['text_a']])
        line_b_text = str(row[cols['text_b']])
        
        if not current_block_parts or current_block_parts[-1]['text'] != line_a_text:
            if current_block_parts:
                text_blocks.append(finalize_block(current_block_parts))
            current_block_parts = [lines_map[line_a_text]]
            
        should_merge = (row['label'] == 1) and (row[cols['page_number_a']] == row[cols['page_number_b']])

        if should_merge:
            current_block_parts.append(lines_map[line_b_text])
        else:
            text_blocks.append(finalize_block(current_block_parts))
            current_block_parts = [lines_map[line_b_text]]

    if current_block_parts:
        text_blocks.append(finalize_block(current_block_parts))

    # --- Create and save the final DataFrame ---
    unique_blocks = {}
    for block in text_blocks:
        if block and block['text'] not in unique_blocks:
            unique_blocks[block['text']] = block
            
    output_df = pd.DataFrame(list(unique_blocks.values()))
    
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
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")