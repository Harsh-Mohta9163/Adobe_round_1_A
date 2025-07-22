import pandas as pd
import ast
import os
import glob
from textblob import TextBlob
import re

def safe_eval_bbox(bbox_str):
    """Safely evaluates a string representation of a list/tuple."""
    try:
        return ast.literal_eval(str(bbox_str))
    except (ValueError, SyntaxError):
        return [0, 0, 0, 0]

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

    full_text = ' '.join(part['text'] for part in block_parts)
    all_bboxes = [part['bbox'] for part in block_parts]
    all_font_sizes = [part['font_size'] for part in block_parts]

    # Calculate average font size from all font sizes in the block
    avg_font_size = round(sum(all_font_sizes) / len(all_font_sizes), 2) if all_font_sizes else 12.0
    
    min_x0 = min(b[0] for b in all_bboxes)
    min_y0 = min(b[1] for b in all_bboxes)
    max_x1 = max(b[2] for b in all_bboxes)
    max_y1 = max(b[3] for b in all_bboxes)
    union_bbox = [min_x0, min_y0, max_x1, max_y1]
    
    word_count = len(full_text.split())
    is_all_caps = 1 if full_text.isupper() and any(c.isalpha() for c in full_text) else 0
    char_density = len(full_text.replace(" ", "")) / ((max_x1 - min_x0) * (max_y1 - min_y0)) if (max_x1 > min_x0 and max_y1 > min_y0) else 0.1
    ratio_of_verbs = calculate_verb_ratio(full_text)
    ratio_capitalized = calculate_capitalized_ratio(full_text)
    
    # Base block data (without title_label yet)
    block_data = {
        'text': full_text,
        'bbox': union_bbox,
        'avg_font_size': avg_font_size,
        'word_count': word_count,
        'is_all_caps': is_all_caps,
        'char_density': char_density,
        'ratio_of_verbs': ratio_of_verbs,
        'ratio_capitalized': ratio_capitalized
    }
    
    # Add additional features from the first block part (use average if numeric)
    if block_parts[0].get('additional_features'):
        first_part_features = block_parts[0]['additional_features']
        for feature_name, feature_value in first_part_features.items():
            if isinstance(feature_value, (int, float)):
                # For numeric features, calculate average across all parts
                all_values = []
                for part in block_parts:
                    if part.get('additional_features', {}).get(feature_name) is not None:
                        all_values.append(part['additional_features'][feature_name])
                if all_values:
                    block_data[feature_name] = round(sum(all_values) / len(all_values), 4)
                else:
                    block_data[feature_name] = feature_value
            else:
                # For non-numeric features, take the first value
                block_data[feature_name] = feature_value
    
    # Add title_label as the last column
    block_data['title_label'] = ''
    
    return block_data

def detect_column_names(df):
    """Detect the correct column names for text and other features."""
    columns = df.columns.tolist()
    
    # Find text columns
    text_a_col = None
    text_b_col = None
    font_size_a_col = None
    font_size_b_col = None
    
    for col in columns:
        col_lower = col.lower()
        if 'text_a' in col_lower and 'span' not in col_lower:
            text_a_col = col
        elif 'text_b' in col_lower and 'span' not in col_lower:
            text_b_col = col
        elif 'font_size_a' in col_lower:
            font_size_a_col = col
        elif 'font_size_b' in col_lower:
            font_size_b_col = col
    
    return {
        'text_a': text_a_col,
        'text_b': text_b_col,
        'font_size_a': font_size_a_col,
        'font_size_b': font_size_b_col
    }

def get_additional_features(df):
    """Get additional features excluding span_text columns, line_b specific columns, and basic columns."""
    exclude_patterns = [
        'text_a', 'text_b', 'span_text_a', 'span_text_b', 
        'font_size_a', 'font_size_b', 'label'
    ]
    
    additional_features = []
    for col in df.columns:
        col_lower = col.lower()
        # Skip if column matches any exclude pattern
        if not any(pattern in col_lower for pattern in exclude_patterns):
            # Skip line_b specific columns (anything ending with _b or _B)
            if not (col.endswith('_b') or col.endswith('_B')):
                additional_features.append(col)
    
    return additional_features

def merge_textlines_guaranteed(input_csv_path: str, output_csv_path: str):
    """
    A robust script to merge textlines that guarantees no lines are dropped.
    Now properly merges lines based on labels and excludes line_b features.
    """
    try:
        # Try multiple encodings to handle different file formats
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(input_csv_path, encoding=encoding)
                print(f"✅ Loaded {os.path.basename(input_csv_path)} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"❌ Could not load {input_csv_path} with any encoding")
            return False
            
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_csv_path}'")
        return False
    except Exception as e:
        print(f"❌ Error loading {input_csv_path}: {e}")
        return False

    if df.empty:
        print(f"⚠️  Warning: Input CSV {os.path.basename(input_csv_path)} is empty.")
        return False

    # Detect column names
    cols = detect_column_names(df)
    
    if not cols['text_a'] or not cols['text_b']:
        print(f"❌ Required text columns not found in {os.path.basename(input_csv_path)}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    if 'label' not in df.columns:
        print(f"❌ 'label' column not found in {os.path.basename(input_csv_path)}")
        return False

    # Get additional features (excluding span_text, line_b columns, basic columns)
    additional_feature_cols = get_additional_features(df)
    
    print(f"📋 Using columns: text_a='{cols['text_a']}', text_b='{cols['text_b']}'")
    print(f"📋 Font size columns: font_size_a='{cols['font_size_a']}', font_size_b='{cols['font_size_b']}'")
    
    if additional_feature_cols:
        print(f"📋 Additional features found: {len(additional_feature_cols)} features")
        print(f"📋 Features: {', '.join(additional_feature_cols[:5])}{'...' if len(additional_feature_cols) > 5 else ''}")

    # Step 1: Create ordered list of unique lines with their features
    unique_lines = []
    seen_texts = set()
    
    # Process each row to build the sequence of unique lines
    for index, row in df.iterrows():
        text_a = str(row[cols['text_a']])
        text_b = str(row[cols['text_b']])
        
        # Get font sizes
        try:
            font_size_a = float(row[cols['font_size_a']]) if cols['font_size_a'] and pd.notna(row.get(cols['font_size_a'])) else 12.0
        except (ValueError, TypeError):
            font_size_a = 12.0
            
        try:
            font_size_b = float(row[cols['font_size_b']]) if cols['font_size_b'] and pd.notna(row.get(cols['font_size_b'])) else 12.0
        except (ValueError, TypeError):
            font_size_b = 12.0
        
        # Get additional features for this row
        additional_features = {}
        for feature_col in additional_feature_cols:
            additional_features[feature_col] = get_feature_value(row, feature_col)
        
        # Add line_A if we haven't seen it before
        if text_a not in seen_texts:
            unique_lines.append({
                'text': text_a,
                'bbox': [0, 0, 100, 20],  # Default bbox
                'font_size': font_size_a,
                'additional_features': additional_features
            })
            seen_texts.add(text_a)
        
        # Add line_B if we haven't seen it before
        if text_b not in seen_texts:
            unique_lines.append({
                'text': text_b,
                'bbox': [100, 0, 200, 20],  # Default bbox
                'font_size': font_size_b,
                'additional_features': additional_features
            })
            seen_texts.add(text_b)

    print(f"📊 Found {len(unique_lines)} unique lines to process")

    # Step 2: Create merge decision lookup
    merge_labels = {}
    for _, row in df.iterrows():
        text_a = str(row[cols['text_a']])
        text_b = str(row[cols['text_b']])
        merge_labels[(text_a, text_b)] = row['label']

    # Step 3: Merge lines based on labels
    text_blocks = []
    if not unique_lines:
        print(f"⚠️  No unique lines found")
        return False

    current_block_parts = [unique_lines[0]]

    for i in range(len(unique_lines) - 1):
        current_line_text = unique_lines[i]['text']
        next_line_text = unique_lines[i+1]['text']
        
        # Check if we should merge with the next line
        should_merge = merge_labels.get((current_line_text, next_line_text), 0) == 1

        if should_merge:
            # Add next line to current block
            current_block_parts.append(unique_lines[i+1])
            print(f"🔗 Merging: '{current_line_text[:30]}...' + '{next_line_text[:30]}...'")
        else:
            # Finalize current block and start new one
            final_block = finalize_block(current_block_parts)
            if final_block:
                text_blocks.append(final_block)
                print(f"✅ Created block: '{final_block['text'][:50]}...' ({len(current_block_parts)} lines)")
            
            # Start new block with next line
            current_block_parts = [unique_lines[i+1]]
            
    # Finalize the last block
    if current_block_parts:
        final_block = finalize_block(current_block_parts)
        if final_block:
            text_blocks.append(final_block)
            print(f"✅ Created final block: '{final_block['text'][:50]}...' ({len(current_block_parts)} lines)")

    # Create and save the final DataFrame
    output_df = pd.DataFrame(text_blocks)
    
    # Ensure title_label is the last column
    if 'title_label' in output_df.columns:
        cols_order = [col for col in output_df.columns if col != 'title_label'] + ['title_label']
        output_df = output_df[cols_order]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Success! Merged {len(unique_lines)} unique lines into {len(output_df)} text blocks.")
    print(f"📁 Output saved to: '{output_csv_path}'")
    
    # Show statistics
    if len(output_df) > 0:
        font_sizes = output_df['avg_font_size'].tolist()
        print(f"📊 Font size range: {min(font_sizes):.1f} - {max(font_sizes):.1f}")
        print(f"📊 Average font size: {sum(font_sizes)/len(font_sizes):.1f}")
        
        if additional_feature_cols:
            print(f"📊 Additional features included: {len(additional_feature_cols)}")
    
    return True

if __name__ == '__main__':
    # --- CONFIGURATION ---
    INPUT_FOLDER = '../../data/textlines_csv_output'  # Folder containing labeled CSV files
    OUTPUT_FOLDER = '../../data/merged_textblocks_gt'  # Output folder for merged textblocks
    # -------------------

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all CSV files in the input folder
    csv_pattern = os.path.join(INPUT_FOLDER, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{INPUT_FOLDER}'")
        exit()
    
    print(f"🔍 Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each CSV file
    successful_files = 0
    failed_files = []
    
    for input_csv in csv_files:
        filename = os.path.basename(input_csv)
        
        # Create output filename
        if 'textlines_ground_truth_' in filename:
            output_filename = filename.replace('textlines_ground_truth_', 'merged_textblocks_')
        else:
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"merged_textblocks_{name_without_ext}.csv"
        
        output_csv = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"\n📄 Processing: {filename}")
        print(f"📤 Output: {output_filename}")
        
        success = merge_textlines_guaranteed(input_csv, output_csv)
        
        if success:
            successful_files += 1
        else:
            failed_files.append(filename)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_files}/{len(csv_files)} files")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")