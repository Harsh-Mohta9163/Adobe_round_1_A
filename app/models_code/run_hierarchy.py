# run_hierarchy.py (Updated for new feature set)
#
# This script assigns a hierarchical level (H1, H2, etc.) to text blocks
# based on numbering and the provided stylistic features.

import pandas as pd
import numpy as np
import regex as re
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
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=feature_columns)

    if df.empty:
        print("Warning: No valid data for clustering after handling non-numeric values.")
        df['style_cluster_id'] = -1
        return df

    style_features = df[feature_columns].values

    # Scale features
    scaler = StandardScaler()
    style_features_scaled = scaler.fit_transform(style_features)

    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(style_features_scaled)
    df['style_cluster_id'] = clustering.labels_
    print("✅ Done.")
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
    # This needs to come before alphabetic to catch valid Roman numerals
    roman_match = re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\.', text, re.IGNORECASE)
    if roman_match:
        return {'type': 'roman', 'depth': 1}
    
    # Check alphabetic patterns (e.g., "A.", "B.", "C.")
    # This comes after Roman numerals to avoid conflicts
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
        'level': 0,  # Changed from 1 to 0 for "Title" level
        'number_depth': first_title['numbering_info']['depth'] if first_title['numbering_info'] else -1,
        'style_cluster_id': first_title['style_cluster_id'],
        'font_size': first_title[font_size_col]
    }
    hierarchy_stack.append(first_title_info)
    hierarchy_levels.append('Title')  # Changed from 'H1' to 'Title'

    for i in range(1, len(df)):
        current_title = df.iloc[i]
        previous_title_info = hierarchy_stack[-1]

        current_number_depth = current_title['numbering_info']['depth'] if current_title['numbering_info'] else -1
        current_style_id = current_title['style_cluster_id']
        current_font_size = current_title[font_size_col]

        # If both have numbering info, use numbering depth
        if current_number_depth > -1 and previous_title_info['number_depth'] > -1:
            if current_number_depth > previous_title_info['number_depth']:
                # Deeper numbering = lower level (higher number)
                new_level = previous_title_info['level'] + 1
            elif current_number_depth == previous_title_info['number_depth']:
                # Same numbering depth = same level
                # Pop the stack to the same level
                while hierarchy_stack and hierarchy_stack[-1]['number_depth'] >= current_number_depth:
                    hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
            else:
                # Shallower numbering = higher level (lower number)
                while hierarchy_stack and current_number_depth <= hierarchy_stack[-1]['number_depth']:
                    hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
        else:
            # No numbering info - use style and font size
            if current_style_id == previous_title_info['style_cluster_id']:
                # Same style = same level
                hierarchy_stack.pop()
                new_level = hierarchy_stack[-1]['level'] + 1 if hierarchy_stack else 1
            elif current_font_size < previous_title_info['font_size']:
                # Smaller font = lower level (higher number)
                new_level = previous_title_info['level'] + 1
            else:
                # Larger font = higher level (lower number)
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

def main():
    """Main function to orchestrate the hierarchy building process."""
    # --- CONFIGURATION ---
    # NOTE: Change these column names if they are different in your CSV file.
    input_filename = '../data/test_results/predictions_merged_textblocks_file02.pdf.csv'
    output_filename = 'final_hierarchy_output.csv'
    
    font_size_col = 'avg_font_size'
    # Features for determining visual style of a title - REMOVED is_all_caps
    style_feature_cols = [
        font_size_col,
        'ratio_capitalized'
    ]

    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file '{input_filename}' not found.")
        return
    except KeyError as e:
        print(f"FATAL ERROR: A required column is missing from the CSV file: {e}")
        print("Please check the column names in the CONFIGURATION section of the script.")
        return

    # Check if model_labels column exists, fallback to title_label if not
    if 'model_labels' in df.columns:
        label_column = 'model_labels'
        print(f"Using model predictions from 'model_labels' column")
    elif 'title_label' in df.columns:
        label_column = 'title_label'
        print(f"Warning: 'model_labels' not found, using 'title_label' column")
    else:
        print("FATAL ERROR: Neither 'model_labels' nor 'title_label' column found.")
        print("Available columns:", list(df.columns))
        return

    # Filter for titles using the determined label column
    titles_df = df[df[label_column] == 1].copy().reset_index(drop=True)

    if titles_df.empty:
        print(f"No titles found ({label_column} == 1) in the input file. Exiting.")
        return

    print(f"Found {len(titles_df)} titles to process")

    # Run the full pipeline
    titles_df = get_style_clusters(titles_df, style_feature_cols)

    print("\nStep 2: Parsing titles for numbering patterns...")
    titles_df['numbering_info'] = titles_df['text'].apply(parse_numbering)
    print("✅ Done.")

    titles_df['hierarchy_level'] = build_hierarchy(titles_df, font_size_col)
    
    print("\n--- Final Document Hierarchy ---")
    final_columns = ['text', 'hierarchy_level'] + style_feature_cols + ['style_cluster_id']
    # Ensure we only try to print columns that exist
    final_columns_exist = [col for col in final_columns if col in titles_df.columns]
    print(titles_df[final_columns_exist].to_string())

    titles_df.to_csv(output_filename, index=False)
    print(f"\n✅ Success! Full results have been saved to '{output_filename}'")


if __name__ == "__main__":
    main()