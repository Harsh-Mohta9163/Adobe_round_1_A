# run_hierarchy.py (Updated for new feature set)
#
# This script assigns a hierarchical level (H1, H2, etc.) to text blocks
# based on numbering and the provided stylistic features.

import pandas as pd
import numpy as np
import regex as re
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def get_style_clusters(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Clusters titles based on the available non-semantic features
    to group them into style families.
    """
    print("Step 1: Clustering titles by visual style...")
    print(f"Using features: {feature_columns}")
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only use columns that actually exist
    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        print("Warning: No valid features available for clustering.")
        df['style_cluster_id'] = 0
        return df
    
    df = df.dropna(subset=available_features)

    if df.empty:
        print("Warning: No valid data for clustering after handling non-numeric values.")
        df['style_cluster_id'] = -1
        return df

    style_features = df[available_features].values

    # Scale features
    scaler = StandardScaler()
    style_features_scaled = scaler.fit_transform(style_features)

    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(style_features_scaled)
    df['style_cluster_id'] = clustering.labels_
    print(f"✅ Done. Created {len(set(clustering.labels_))} style clusters.")
    return df

def parse_numbering(text: str) -> dict | None:
    """
    Parses title text for numbering patterns.
    """
    text = str(text).strip()
    
    # Check numeric patterns first (e.g., "1.", "1.1.", "1.1.1.")
    numeric_match = re.match(r'^(\d+(\.\d+)*)', text)
    if numeric_match:
        number_str = numeric_match.group(1)
        return {'type': 'numeric', 'depth': number_str.count('.') + 1}

    # Check Roman numeral patterns FIRST (e.g., "I.", "II.", "III.", "IV.", "V.")
    roman_match = re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.', text, re.IGNORECASE)
    if roman_match:
        return {'type': 'roman', 'depth': 1}
    
    # Check alphabetic patterns (e.g., "A.", "B.", "C.")
    alpha_match = re.match(r'^[A-Z]\.', text)
    if alpha_match:
        return {'type': 'alpha', 'depth': 2}

    # Check keyword patterns (e.g., "Appendix A:", "Figure 1:", "Table 2:")
    keyword_match = re.match(r'^(Appendix|Figure|Table)\s+[A-Z0-9]+:?', text, re.IGNORECASE)
    if keyword_match:
        return {'type': 'keyword', 'depth': 1}

    return None

def build_hierarchy(df: pd.DataFrame, font_size_col: str) -> list:
    """
    Applies the deterministic, stack-based algorithm to assign hierarchy levels.
    """
    print("\nStep 3: Building hierarchy with stateful algorithm...")
    if df.empty:
        return []

    hierarchy_levels = []
    hierarchy_stack = []

    # First title should be "Title" level (level 0)
    first_title = df.iloc[0]
    first_title_info = {
        'level': 0,
        'number_depth': first_title['numbering_info']['depth'] if first_title['numbering_info'] else -1,
        'style_cluster_id': first_title['style_cluster_id'],
        'font_size': first_title[font_size_col]
    }
    hierarchy_stack.append(first_title_info)
    hierarchy_levels.append('Title')

    for i in range(1, len(df)):
        current_title = df.iloc[i]
        previous_title_info = hierarchy_stack[-1]

        current_number_depth = current_title['numbering_info']['depth'] if current_title['numbering_info'] else -1
        current_style_id = current_title['style_cluster_id']
        current_font_size = current_title[font_size_col]

        # If both have numbering info, use numbering depth
        if current_number_depth > -1 and previous_title_info['number_depth'] > -1:
            if current_number_depth > previous_title_info['number_depth']:
                new_level = previous_title_info['level'] + 1
            elif current_number_depth == previous_title_info['number_depth']:
                while hierarchy_stack and hierarchy_stack[-1]['number_depth'] >= current_number_depth:
                    hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
            else:
                while hierarchy_stack and current_number_depth <= hierarchy_stack[-1]['number_depth']:
                    hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
        else:
            # No numbering info - use style and font size
            if current_style_id == previous_title_info['style_cluster_id']:
                hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
            elif current_font_size < previous_title_info['font_size']:
                new_level = previous_title_info['level'] + 1
            else:
                while hierarchy_stack and current_font_size >= hierarchy_stack[-1]['font_size']:
                    hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
        
        current_title_info = {
            'level': new_level,
            'number_depth': current_number_depth,
            'style_cluster_id': current_style_id,
            'font_size': current_font_size
        }
        hierarchy_stack.append(current_title_info)
        
        # Format the hierarchy level
        if new_level == 0:
            hierarchy_levels.append('Title')
        else:
            hierarchy_levels.append(f'H{new_level}')
        
    print("✅ Done.")
    return hierarchy_levels

def process_single_file(input_file: str, output_file: str, style_feature_cols: list, font_size_col: str) -> bool:
    """Process a single CSV file and generate hierarchy."""
    try:
        print(f"\n📄 Processing: {os.path.basename(input_file)}")
        df = pd.read_csv(input_file)
        
        # Check if model_labels column exists, fallback to title_label if not
        if 'model_labels' in df.columns:
            label_column = 'model_labels'
            print(f"Using model predictions from 'model_labels' column")
        elif 'title_label' in df.columns:
            label_column = 'title_label'
            print(f"Warning: 'model_labels' not found, using 'title_label' column")
        else:
            print("ERROR: Neither 'model_labels' nor 'title_label' column found.")
            return False

        # Filter for titles using the determined label column
        titles_df = df[df[label_column] == 1].copy().reset_index(drop=True)

        if titles_df.empty:
            print(f"No titles found ({label_column} == 1) in the input file.")
            return False

        print(f"Found {len(titles_df)} titles to process")

        # Run the full pipeline
        titles_df = get_style_clusters(titles_df, style_feature_cols)

        print("\nStep 2: Parsing titles for numbering patterns...")
        titles_df['numbering_info'] = titles_df['text'].apply(parse_numbering)
        print("✅ Done.")

        titles_df['hierarchy_level'] = build_hierarchy(titles_df, font_size_col)
        
        print("\n--- Final Document Hierarchy ---")
        final_columns = ['text', 'hierarchy_level'] + [col for col in style_feature_cols if col in titles_df.columns] + ['style_cluster_id']
        print(titles_df[final_columns].to_string())

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        titles_df.to_csv(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False

def main():
    """Main function to orchestrate the hierarchy building process for multiple files."""
    # --- CONFIGURATION ---
    INPUT_FOLDER = '../../data/test_labelled_merged_textblocks_gt'  # Folder containing prediction CSV files
    OUTPUT_FOLDER = '../../data/final_results'  # Output folder for hierarchy results
    
    font_size_col = 'avg_font_size'
    
    # Extended features for determining visual style of titles
    style_feature_cols = [
        font_size_col,
        'ratio_capitalized',
        'word_count',
        'is_all_caps',
        'char_density',
        'ratio_of_verbs',
        'is_hashed',  # NEW: Add is_hashed feature
        'normalized_vertical_gap',
        'indentation_change',
        'same_alignment',
        'is_centered_A',
        'font_size_diff',
        'same_font',
        'is_bold_A',
        'is_italic_A',
        'is_monospace_A',
        'same_bold',
        'same_italic',
        'same_monospace'
    ]

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all CSV files in the input folder
    csv_pattern = os.path.join(INPUT_FOLDER, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{INPUT_FOLDER}'")
        print("Make sure you have prediction CSV files from the model testing phase.")
        return
    
    print(f"🔍 Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each CSV file
    successful_files = 0
    failed_files = []
    
    for input_csv in csv_files:
        filename = os.path.basename(input_csv)
        
        # Create output filename
        # Change from 'predictions_*.csv' to 'hierarchy_*.csv'
        if filename.startswith('predictions_'):
            output_filename = filename.replace('predictions_', 'hierarchy_')
        else:
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"hierarchy_{name_without_ext}.csv"
        
        output_csv = os.path.join(OUTPUT_FOLDER, output_filename)
        
        success = process_single_file(input_csv, output_csv, style_feature_cols, font_size_col)
        
        if success:
            successful_files += 1
        else:
            failed_files.append(filename)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully processed: {successful_files}/{len(csv_files)} files")
    print(f"📁 Input folder: {INPUT_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    if successful_files > 0:
        print(f"\n💡 Results:")
        print(f"   - Hierarchy CSV files saved to '{OUTPUT_FOLDER}'")
        print(f"   - Each file contains title hierarchies (Title, H1, H2, H3, etc.)")
        print(f"   - Style clustering used {len(style_feature_cols)} features")

if __name__ == "__main__":
    main()