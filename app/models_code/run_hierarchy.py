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
# Import KMeans for better font clustering
from sklearn.cluster import KMeans

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

    # Check Roman numeral patterns (e.g., "I.", "II.", "III.", "IV.", "V.")
    # Made slightly more flexible by allowing missing period.
    roman_match = re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.?', text, re.IGNORECASE)
    if roman_match:
        return {'type': 'roman', 'depth': 1}
    
    # Check alphabetic patterns (e.g., "A.", "B.", "C.")
    alpha_match = re.match(r'^[A-Z]\.', text)
    if alpha_match:
        # This could be H2 or H3 depending on context, but assigning a fixed depth is a reasonable start.
        return {'type': 'alpha', 'depth': 2}

    # Check keyword patterns (e.g., "Appendix A:", "Figure 1:", "Table 2:")
    keyword_match = re.match(r'^(Appendix|Figure|Table|Chapter|Section)\s+[A-Z0-9]+:?', text, re.IGNORECASE)
    if keyword_match:
        return {'type': 'keyword', 'depth': 1}

    return None

def build_hierarchy(df: pd.DataFrame, font_size_col: str) -> list:
    """
    Assigns hierarchy levels by prioritizing numbering, then using KMeans font 
    clustering on the document body to robustly determine heading levels.
    """
    print("\nStep 3: Building hierarchy with KMeans font clustering...")
    if df.empty:
        return []

    # The first title in the CSV is always considered the main "Title".
    hierarchy_results = ["Title"]
    if len(df) <= 1:
        return hierarchy_results

    titles_df = df.iloc[1:].copy()

    # --- 1. Determine Level from Numbering (Primary Rule) ---
    titles_df['determined_level'] = titles_df['numbering_info'].apply(
        lambda info: info.get('depth') if isinstance(info, dict) else np.nan
    )

    # --- 2. For Un-numbered Titles, Cluster by Font Size ---
    unnumbered_mask = titles_df['determined_level'].isna()
    if unnumbered_mask.any():
        
        # KEY CHANGE: Exclude the first few titles (e.g., 4) from the training data 
        # for clustering, as they are often anomalous cover page titles.
        # This prevents their large fonts from skewing the hierarchy of the document body.
        OFFSET_FOR_COVER_PAGE = 4
        font_sizes_for_clustering = titles_df[unnumbered_mask].iloc[OFFSET_FOR_COVER_PAGE:][[font_size_col]].dropna()

        # If there are still fonts to cluster after the offset...
        if not font_sizes_for_clustering.empty:
            unique_font_sizes = font_sizes_for_clustering.drop_duplicates()
            
            # Assume a max of 4 heading levels for un-numbered text (H1-H4).
            max_levels = 4
            num_clusters = min(max_levels, len(unique_font_sizes))

            if num_clusters > 0:
                # Use KMeans to find the main font size tiers in the document body.
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(unique_font_sizes.values)
                
                # Rank clusters by font size. Largest font = Level 1.
                cluster_centers = kmeans.cluster_centers_.flatten()
                level_ranks = np.argsort(cluster_centers)[::-1]
                cluster_to_level_map = {rank: i + 1 for i, rank in enumerate(level_ranks)}

                # Create a map from each font size to its determined hierarchy level.
                font_to_level_map = {}
                all_font_sizes = titles_df[unnumbered_mask][[font_size_col]].dropna().drop_duplicates()
                labels = kmeans.predict(all_font_sizes.values)
                for i, font_size in enumerate(all_font_sizes[font_size_col]):
                    cluster_label = labels[i]
                    font_to_level_map[font_size] = cluster_to_level_map.get(cluster_label)

                # Assign the calculated level to each un-numbered title.
                titles_df.loc[unnumbered_mask, 'determined_level'] = titles_df.loc[unnumbered_mask, font_size_col].map(font_to_level_map)

    # Fallback for any titles that still couldn't be assigned a level.
    titles_df['determined_level'] = titles_df['determined_level'].fillna(99).astype(int)

    # --- 3. Enforce a Logical Hierarchy Order (Stack-based Correction) ---
    level_stack = [0]  # Start with the 'Title' level (0)

    for level in titles_df['determined_level']:
        # If level is 99 (undetermined), make it a child of the previous level.
        if level == 99:
            level = level_stack[-1] + 1

        # Find the correct parent in the stack. e.g., after an H3, a new H2 should be attached to the parent H1.
        while level <= level_stack[-1]:
            level_stack.pop()
        
        # Prevent invalid jumps (e.g., H1 -> H3). Level can be at most parent_level + 1.
        corrected_level = min(level, level_stack[-1] + 1)
        
        level_stack.append(corrected_level)
        hierarchy_results.append(f"H{corrected_level}")

    print("✅ Done.")
    return hierarchy_results


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
        final_columns = ['text', 'hierarchy_level', font_size_col] + [col for col in style_feature_cols if col in titles_df.columns] + ['style_cluster_id']
        # Remove duplicates for cleaner printing
        final_columns = list(dict.fromkeys(final_columns))
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
    INPUT_FOLDER = './predictions/'  # Folder containing prediction CSV files
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