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

def finalize_block(block_parts):
    """
    Processes a list of block parts to create a single, finalized textblock dictionary.
    Now calculates average font size from all individual font sizes in the block.
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
    
    return {
        'text': full_text,
        'bbox': union_bbox,
        'avg_font_size': avg_font_size,  # This is now the average of all font sizes in the block
        'word_count': word_count,
        'is_all_caps': is_all_caps,
        'char_density': char_density,
        'ratio_of_verbs': ratio_of_verbs,
        'ratio_capitalized': ratio_capitalized,
        'title_label': ''
    }

def detect_column_names(df):
    """Detect the correct column names for text and other features."""
    columns = df.columns.tolist()
    
    # Find text columns
    text_a_col = None
    text_b_col = None
    bbox_a_col = None
    bbox_b_col = None
    font_size_a_col = None
    font_size_b_col = None
    
    for col in columns:
        col_lower = col.lower()
        if 'text_a' in col_lower or 'line_a_text' in col_lower:
            text_a_col = col
        elif 'text_b' in col_lower or 'line_b_text' in col_lower:
            text_b_col = col
        elif 'line_a_bbox' in col_lower or 'bbox_a' in col_lower:
            bbox_a_col = col
        elif 'line_b_bbox' in col_lower or 'bbox_b' in col_lower:
            bbox_b_col = col
        elif 'line_a_font_size' in col_lower or 'font_size_a' in col_lower:
            font_size_a_col = col
        elif 'line_b_font_size' in col_lower or 'font_size_b' in col_lower:
            font_size_b_col = col
    
    return {
        'text_a': text_a_col,
        'text_b': text_b_col,
        'bbox_a': bbox_a_col,
        'bbox_b': bbox_b_col,
        'font_size_a': font_size_a_col,
        'font_size_b': font_size_b_col
    }

def merge_textlines_guaranteed(input_csv_path: str, output_csv_path: str):
    """
    A robust script to merge textlines that guarantees no lines are dropped.
    Now properly uses font_size_a and font_size_b from CSV to calculate block averages.
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
        print(f"Looking for columns like: text_a, text_b, line_A_text, line_B_text")
        return False
    
    if 'label' not in df.columns:
        print(f"❌ 'label' column not found in {os.path.basename(input_csv_path)}")
        return False

    print(f"📋 Using columns: text_a='{cols['text_a']}', text_b='{cols['text_b']}'")
    if cols['font_size_a'] and cols['font_size_b']:
        print(f"📋 Font size columns: font_size_a='{cols['font_size_a']}', font_size_b='{cols['font_size_b']}'")
    else:
        print(f"⚠️  Font size columns not found, using defaults")

    # Step 1: Create a definitive, ordered list of unique lines with their font sizes
    unique_lines = []
    seen_texts = set()
    
    for index, row in df.iterrows():
        # Get text values
        text_a = str(row[cols['text_a']])
        text_b = str(row[cols['text_b']])
        
        # Get bbox values (with defaults)
        bbox_a = safe_eval_bbox(row.get(cols['bbox_a'], '[0,0,100,20]')) if cols['bbox_a'] else [0,0,100,20]
        bbox_b = safe_eval_bbox(row.get(cols['bbox_b'], '[100,0,200,20]')) if cols['bbox_b'] else [100,0,200,20]
        
        # Get font sizes from the CSV columns (this is the key change!)
        try:
            font_size_a = float(row[cols['font_size_a']]) if cols['font_size_a'] and pd.notna(row.get(cols['font_size_a'])) else 12.0
        except (ValueError, TypeError):
            font_size_a = 12.0
            
        try:
            font_size_b = float(row[cols['font_size_b']]) if cols['font_size_b'] and pd.notna(row.get(cols['font_size_b'])) else 12.0
        except (ValueError, TypeError):
            font_size_b = 12.0
        
        # Add line_A if we haven't seen it before
        if text_a not in seen_texts:
            unique_lines.append({
                'text': text_a,
                'bbox': bbox_a,
                'font_size': font_size_a  # Use actual font size from CSV
            })
            seen_texts.add(text_a)
        
        # Add line_B if we haven't seen it before
        if text_b not in seen_texts:
            unique_lines.append({
                'text': text_b,
                'bbox': bbox_b,
                'font_size': font_size_b  # Use actual font size from CSV
            })
            seen_texts.add(text_b)

    # Step 2: Iterate through the sequence of lines and group them
    text_blocks = []
    if not unique_lines:
        print(f"⚠️  No unique lines found in {os.path.basename(input_csv_path)}")
        return False

    # Create a lookup for merge decisions
    merge_labels = {}
    for _, row in df.iterrows():
        text_a = str(row[cols['text_a']])
        text_b = str(row[cols['text_b']])
        merge_labels[(text_a, text_b)] = row['label']

    current_block_parts = [unique_lines[0]]

    for i in range(len(unique_lines) - 1):
        current_line_text = unique_lines[i]['text']
        next_line_text = unique_lines[i+1]['text']
        
        # Decide whether to merge with the next line
        should_merge = merge_labels.get((current_line_text, next_line_text), 0) == 1

        if should_merge:
            current_block_parts.append(unique_lines[i+1])
        else:
            # Finalize the completed block (this will calculate avg font size from all parts)
            final_block = finalize_block(current_block_parts)
            if final_block:
                text_blocks.append(final_block)
            # Start a new block with the next line
            current_block_parts = [unique_lines[i+1]]
            
    # Finalize the very last block after the loop
    if current_block_parts:
        final_block = finalize_block(current_block_parts)
        if final_block:
            text_blocks.append(final_block)

    # Create and save the final DataFrame
    output_df = pd.DataFrame(text_blocks)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Success! All {len(unique_lines)} unique lines have been merged into {len(output_df)} blocks.")
    print(f"📁 Output saved to: '{output_csv_path}'")
    
    # Show font size statistics
    font_sizes = output_df['avg_font_size'].tolist()
    print(f"📊 Font size range: {min(font_sizes):.1f} - {max(font_sizes):.1f}")
    print(f"📊 Average font size across all blocks: {sum(font_sizes)/len(font_sizes):.1f}")
    
    return True

if __name__ == '__main__':
    # --- CONFIGURATION ---
    INPUT_FOLDER = '../data/merge_lines_gt'  # Folder containing labeled CSV files
    OUTPUT_FOLDER = '../data/merged_textblocks_gt'  # Output folder for merged textblocks
    # -------------------

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all CSV files in the input folder
    csv_pattern = os.path.join(INPUT_FOLDER, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{INPUT_FOLDER}'")
        print(f"Please make sure you have labeled CSV files in the folder.")
        print(f"Expected files like: merging_features_for_labeling_*.csv")
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
        # Change from 'merging_features_for_labeling_*.csv' to 'merged_textblocks_*.csv'
        if 'merging_features_for_labeling_' in filename:
            output_filename = filename.replace('merging_features_for_labeling_', 'merged_textblocks_')
        else:
            # Fallback: just add prefix
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
    print(f"📁 Input folder: {INPUT_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    if successful_files > 0:
        print(f"\n💡 Next steps:")
        print(f"   1. Review the generated CSV files in '{OUTPUT_FOLDER}'")
        print(f"   2. Manually label the 'title_label' column (1 for titles/headings, 0 for body text)")
        print(f"   3. Use the labeled data to train your title classifier")

    # Prerequisite reminder
    print(f"\n📋 Prerequisites (if not already done):")
    print(f"   pip install pandas textblob")
    print(f"   python -m textblob.download_corpora")