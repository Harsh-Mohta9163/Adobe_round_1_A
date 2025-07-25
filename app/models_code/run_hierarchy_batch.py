import pandas as pd
import numpy as np
import regex as re
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def get_style_clusters(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Clusters titles based on available features"""
    print("  Clustering titles by visual style...")
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only use columns that actually exist
    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        print("  Warning: No valid features available for clustering.")
        df['style_cluster_id'] = 0
        return df
    
    df = df.dropna(subset=available_features)

    if df.empty:
        print("  Warning: No valid data for clustering.")
        df['style_cluster_id'] = -1
        return df

    style_features = df[available_features].values

    # Scale features
    scaler = StandardScaler()
    style_features_scaled = scaler.fit_transform(style_features)

    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(style_features_scaled)
    df['style_cluster_id'] = clustering.labels_
    print(f"  ✅ Created {len(set(clustering.labels_))} style clusters")
    return df

def parse_numbering(text: str) -> dict | None:
    """Parse title text for numbering patterns"""
    text = str(text).strip()
    
    # Check numeric patterns first
    numeric_match = re.match(r'^(\d+(\.\d+)*)', text)
    if numeric_match:
        number_str = numeric_match.group(1)
        return {'type': 'numeric', 'depth': number_str.count('.') + 1}

    # Check Roman numeral patterns
    roman_match = re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.', text, re.IGNORECASE)
    if roman_match:
        return {'type': 'roman', 'depth': 1}
    
    # Check alphabetic patterns
    alpha_match = re.match(r'^[A-Z]\.', text)
    if alpha_match:
        return {'type': 'alpha', 'depth': 2}

    # Check keyword patterns
    keyword_match = re.match(r'^(Appendix|Figure|Table)\s+[A-Z0-9]+:?', text, re.IGNORECASE)
    if keyword_match:
        return {'type': 'keyword', 'depth': 1}

    return None

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def build_hierarchy(df: pd.DataFrame, font_size_col: str) -> list:
    """
    Assigns hierarchy levels by first globally clustering font sizes to establish
    style tiers, and then refines the hierarchy using numbering information.
    This approach is more robust against noisy or out-of-place titles.
    """
    print("\nStep 3: Building hierarchy with revised font-clustering algorithm...")
    if df.empty:
        return []

    hierarchy_levels = ["Title"]
    if len(df) <= 1:
        return hierarchy_levels

    titles_df = df.iloc[1:].copy()

    # --- 1. Globally Cluster Font Sizes to Find Hierarchy Tiers ---
    # Use Agglomerative Clustering to find natural groups of font sizes.
    # The distance_threshold is key: it groups fonts that are "close" together.
    # A threshold of 1.5 means fonts like 14.04 and 14.2 would be in the same cluster,
    # but 14.04 and 15.96 would be in different clusters.
    font_sizes = titles_df[[font_size_col]].values
    agg_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='single')
    font_clusters = agg_cluster.fit_predict(font_sizes)
    titles_df['font_cluster'] = font_clusters

    # --- 2. Rank Clusters to Create Font-Based Levels ---
    # The cluster with the highest average font size gets the highest rank (H1).
    cluster_ranks = titles_df.groupby('font_cluster')[font_size_col].mean().sort_values(ascending=False).index
    level_map = {cluster_id: i + 1 for i, cluster_id in enumerate(cluster_ranks)}
    titles_df['font_based_level'] = titles_df['font_cluster'].map(level_map)

    # --- 3. Determine Final Level using Numbering as the Primary Rule ---
    # The numbering depth (e.g., depth 2 for "2.1") is the most reliable signal.
    def get_final_level(row):
        num_info = row['numbering_info']
        if num_info and isinstance(num_info, dict):
            return num_info.get('depth', row['font_based_level'])
        return row['font_based_level']

    titles_df['final_level'] = titles_df.apply(get_final_level, axis=1)

    # --- 4. Ensure Logical Order (e.g., no H1 -> H3 jumps) ---
    # Use a simple stack to enforce a logical hierarchy sequence.
    level_stack = [0]  # Start with the 'Title' level
    for determined_level in titles_df['final_level']:
        # Pop from the stack until the parent level is found.
        # A parent level must be strictly smaller than the current level.
        while determined_level <= level_stack[-1]:
            level_stack.pop()
        
        # Correct the determined level if it creates an invalid jump (e.g., 0 -> 2)
        corrected_level = min(determined_level, level_stack[-1] + 1)
        level_stack.append(corrected_level)
        hierarchy_levels.append(f"H{corrected_level}")

    print("✅ Done.")
    return hierarchy_levels

def process_single_hierarchy_file(input_file: str, output_file: str, style_feature_cols: list, font_size_col: str) -> bool:
    """Process a single CSV file for hierarchy analysis"""
    try:
        print(f"📄 Processing: {os.path.basename(input_file)}")
        df = pd.read_csv(input_file)
        
        # Check for model_labels or title_label column
        if 'model_labels' in df.columns:
            label_column = 'model_labels'
        elif 'title_label' in df.columns:
            label_column = 'title_label'
        else:
            print("  ❌ No label column found")
            return False

        # Filter for titles
        titles_df = df[df[label_column] == 1].copy().reset_index(drop=True)

        if titles_df.empty:
            print(f"  ❌ No titles found")
            return False

        print(f"  Found {len(titles_df)} titles to process")

        # Run the hierarchy pipeline
        titles_df = get_style_clusters(titles_df, style_feature_cols)
        
        print("  Parsing numbering patterns...")
        titles_df['numbering_info'] = titles_df['text'].apply(parse_numbering)
        
        titles_df['hierarchy_level'] = build_hierarchy(titles_df, font_size_col)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        titles_df.to_csv(output_file, index=False)
        print(f"  ✅ Results saved to: {os.path.basename(output_file)}")
        
        # Show hierarchy summary
        hierarchy_counts = titles_df['hierarchy_level'].value_counts()
        print(f"  Hierarchy: {dict(hierarchy_counts)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {os.path.basename(input_file)}: {e}")
        return False

def process_all_hierarchy_files(input_folder, output_folder):
    """Process all files in the input folder for hierarchy analysis"""
    print(f"Processing hierarchy files from: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Feature columns for style clustering
    style_feature_cols = [
        'avg_font_size', 'ratio_capitalized', 'word_count', 'is_all_caps',
        'char_density', 'ratio_of_verbs', 'is_hashed', 'normalized_vertical_gap',
        'indentation_change', 'same_alignment', 'is_centered_A', 'font_size_diff',
        'same_font', 'is_bold_A', 'is_italic_A', 'is_monospace_A',
        'same_bold', 'same_italic', 'same_monospace'
    ]
    
    font_size_col = 'avg_font_size'
    
    # Find all CSV files
    csv_pattern = os.path.join(input_folder, '*.csv')
    input_files = glob.glob(csv_pattern)
    
    if not input_files:
        print(f"❌ No CSV files found in '{input_folder}'")
        return False
    
    print(f"Found {len(input_files)} CSV files to process:")
    for file in input_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    successful_files = 0
    
    for input_csv in input_files:
        filename = os.path.basename(input_csv)
        
        # Create output filename
        if filename.startswith('textblock_predictions_'):
            output_filename = filename.replace('textblock_predictions_', 'hierarchy_')
        else:
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"hierarchy_{name_without_ext}.csv"
        
        output_csv = os.path.join(output_folder, output_filename)
        
        success = process_single_hierarchy_file(input_csv, output_csv, style_feature_cols, font_size_col)
        
        if success:
            successful_files += 1
    
    print(f"\n✅ Successfully processed {successful_files}/{len(input_files)} hierarchy files")
    
    return successful_files > 0

if __name__ == "__main__":
    input_folder = '../../data/textblock_predictions'
    output_folder = '../../data/final_results'
    
    success = process_all_hierarchy_files(input_folder, output_folder)
    
    if success:
        print("✅ Batch hierarchy processing completed successfully")
    else:
        print("❌ Batch hierarchy processing failed")
        exit(1)