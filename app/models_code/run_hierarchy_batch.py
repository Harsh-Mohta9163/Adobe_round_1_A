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
    print("  Step 1: Clustering titles by visual style...")
    print(f"  Using features: {feature_columns}")
    
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
        print("  Warning: No valid data for clustering after handling non-numeric values.")
        df['style_cluster_id'] = -1
        return df

    style_features = df[available_features].values

    # Scale features
    scaler = StandardScaler()
    style_features_scaled = scaler.fit_transform(style_features)

    # Use DBSCAN for clustering
    clustering = DBSCAN(eps=1.0, min_samples=1).fit(style_features_scaled)
    df['style_cluster_id'] = clustering.labels_
    print(f"  ✅ Done. Created {len(set(clustering.labels_))} style clusters.")
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

def build_hierarchy(df: pd.DataFrame, font_size_col: str, page_num_col: str) -> list:
    """
    Assigns hierarchy levels. The "Title" is the largest font heading on the 
    first page where headings appear. The remaining body headings are determined 
    by numbering and KMeans font clustering.
    """
    print("\n  Step 3: Building hierarchy with new title logic...")
    if df.empty:
        return []

    # Debug: Print available columns and first few rows
    print(f"  Available columns: {df.columns.tolist()}")
    print(f"  Font size column '{font_size_col}' exists: {font_size_col in df.columns}")
    print(f"  Page number column '{page_num_col}' exists: {page_num_col in df.columns}")
    
    if font_size_col in df.columns:
        print(f"  Font size data preview:")
        print(df[[font_size_col]].head())
        print(f"  Font size data types: {df[font_size_col].dtype}")
    
    if page_num_col in df.columns:
        print(f"  Page number data preview:")
        print(df[[page_num_col]].head())
        print(f"  Page number data types: {df[page_num_col].dtype}")

    # --- 1. Identify the main 'Title' row ---
    title_row_index = -1
    # The new logic can only be used if page number and font size are available.
    use_new_title_logic = page_num_col in df.columns and font_size_col in df.columns

    if use_new_title_logic:
        print("  Using new title logic: Largest font on the first page with headings.")
        
        # Ensure font size column is numeric
        df[font_size_col] = pd.to_numeric(df[font_size_col], errors='coerce')
        df[page_num_col] = pd.to_numeric(df[page_num_col], errors='coerce')
        
        # Find the page number of the very first heading in the document.
        first_heading_page = df.iloc[0][page_num_col]
        print(f"  First heading page: {first_heading_page}")
        
        # Get all headings on that first page.
        first_page_titles_df = df[df[page_num_col] == first_heading_page]
        print(f"  Number of headings on first page: {len(first_page_titles_df)}")
        
        if not first_page_titles_df.empty:
            # Debug: Show all headings on first page with their font sizes
            print("  Headings on first page:")
            debug_cols = ['text', font_size_col]
            if page_num_col in first_page_titles_df.columns:
                debug_cols.append(page_num_col)
            print(first_page_titles_df[debug_cols].to_string())
            
            # Find the index of the heading with the largest font on that page.
            # Handle NaN values by dropping them first
            valid_font_sizes = first_page_titles_df.dropna(subset=[font_size_col])
            if not valid_font_sizes.empty:
                title_row_index = valid_font_sizes[font_size_col].idxmax()
                max_font_size = valid_font_sizes[font_size_col].max()
                title_text = df.loc[title_row_index, 'text']
                print(f"  Selected title: '{title_text}' (font size: {max_font_size}, index: {title_row_index})")
            else:
                print("  Warning: No valid font sizes found on first page")
    
    if title_row_index == -1:
        print("  Warning: Could not determine title with new logic. Falling back to taking the first heading found.")
        # Fallback to original logic: the first title in the CSV is the main "Title".
        title_row_index = df.index[0]
        fallback_text = df.loc[title_row_index, 'text']
        print(f"  Fallback title: '{fallback_text}' (index: {title_row_index})")

    # --- 2. Separate Title from the rest of the document (body) ---
    body_df = df.drop(title_row_index).copy()
    
    # --- 3. Process the Document Body ---
    # This section processes all non-Title headings
    if not body_df.empty:
        # A. Determine Level from Numbering (Primary Rule)
        body_df['determined_level'] = body_df['numbering_info'].apply(
            lambda info: info.get('depth') if isinstance(info, dict) else np.nan
        )

        # B. For Un-numbered Titles, Cluster by Font Size using KMeans
        unnumbered_mask = body_df['determined_level'].isna()
        if unnumbered_mask.any():
            OFFSET_FOR_COVER_PAGE = 4
            font_sizes_for_clustering = body_df[unnumbered_mask].iloc[OFFSET_FOR_COVER_PAGE:][[font_size_col]].dropna()

            if not font_sizes_for_clustering.empty:
                unique_font_sizes = font_sizes_for_clustering.drop_duplicates()
                max_levels = 4
                num_clusters = min(max_levels, len(unique_font_sizes))

                if num_clusters > 0:
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(unique_font_sizes.values)
                    cluster_centers = kmeans.cluster_centers_.flatten()
                    level_ranks = np.argsort(cluster_centers)[::-1]
                    cluster_to_level_map = {rank: i + 1 for i, rank in enumerate(level_ranks)}

                    font_to_level_map = {}
                    all_font_sizes = body_df[unnumbered_mask][[font_size_col]].dropna().drop_duplicates()
                    labels = kmeans.predict(all_font_sizes.values)
                    for i, font_size in enumerate(all_font_sizes[font_size_col]):
                        cluster_label = labels[i]
                        font_to_level_map[font_size] = cluster_to_level_map.get(cluster_label)

                    body_df.loc[unnumbered_mask, 'determined_level'] = body_df.loc[unnumbered_mask, font_size_col].map(font_to_level_map)

        body_df['determined_level'] = body_df['determined_level'].fillna(99).astype(int)

        # C. Enforce a Logical Hierarchy Order (Stack-based Correction)
        level_stack = [0]  # Start with the 'Title' level (0) as the parent
        body_hierarchy_levels = []
        for level in body_df['determined_level']:
            if level == 99:
                level = level_stack[-1] + 1
            while level <= level_stack[-1]:
                level_stack.pop()
            corrected_level = min(level, level_stack[-1] + 1)
            level_stack.append(corrected_level)
            body_hierarchy_levels.append(f"H{corrected_level}")
        
        body_df['hierarchy_level'] = body_hierarchy_levels

    # --- 4. Combine Title and Body results ---
    # Add the 'hierarchy_level' column to the original df and update it with the body results
    df['hierarchy_level'] = None
    if not body_df.empty:
        df.update(body_df)
    
    # Set the 'Title' at its identified location
    df.loc[title_row_index, 'hierarchy_level'] = "Title"
    
    print("  ✅ Done.")
    return df['hierarchy_level'].tolist()

def process_single_hierarchy_file(input_file: str, output_file: str, style_feature_cols: list, font_size_col: str, page_num_col: str) -> bool:
    """Process a single CSV file for hierarchy analysis"""
    try:
        print(f"\n📄 Processing: {os.path.basename(input_file)}")
        df = pd.read_csv(input_file)
        
        if page_num_col not in df.columns:
            print(f"  Warning: Page number column '{page_num_col}' not found. Title detection will be less accurate.")

        # Check if model_labels column exists, fallback to title_label if not
        if 'model_labels' in df.columns:
            label_column = 'model_labels'
            print(f"  Using model predictions from 'model_labels' column")
        elif 'title_label' in df.columns:
            label_column = 'title_label'
            print(f"  Warning: 'model_labels' not found, using 'title_label' column")
        else:
            print("  ERROR: Neither 'model_labels' nor 'title_label' column found.")
            return False

        # Filter for titles using the determined label column
        titles_df = df[df[label_column] == 1].copy().reset_index(drop=True)

        if titles_df.empty:
            print(f"  No titles found ({label_column} == 1) in the input file.")
            return False

        print(f"  Found {len(titles_df)} titles to process")

        # Run the full pipeline
        titles_df = get_style_clusters(titles_df, style_feature_cols)

        print("\n  Step 2: Parsing titles for numbering patterns...")
        titles_df['numbering_info'] = titles_df['text'].apply(parse_numbering)
        print("  ✅ Done.")

        titles_df['hierarchy_level'] = build_hierarchy(titles_df, font_size_col, page_num_col)
        
        print("\n  --- Final Document Hierarchy ---")
        
        # --- CORRECTED SECTION ---
        # Define the ideal set of columns we want to see in the final output.
        ideal_columns = (
            ['text', 'hierarchy_level', font_size_col, page_num_col] 
            + style_feature_cols 
            + ['style_cluster_id']
        )
        
        # Filter the ideal list to include only columns that actually exist in the DataFrame.
        # This prevents the KeyError if 'page_num' or other columns are missing.
        final_columns = [col for col in ideal_columns if col in titles_df.columns]
        
        # Remove any duplicates from the list while preserving the order.
        final_columns = list(dict.fromkeys(final_columns))
        
        print(titles_df[final_columns].to_string())
        # --- END OF CORRECTION ---

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        titles_df.to_csv(output_file, index=False)
        print(f"  ✅ Results saved to: {output_file}")
        
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
    
    # Define column names for required data.
    # The new title logic requires a column with the page number for each text block.
    font_size_col = 'avg_font_size'
    page_num_col = 'page_number'  # NEW: Specify the page number column
    
    # Features for determining visual style of titles
    style_feature_cols = [
        font_size_col,
        'ratio_capitalized',
        'word_count',
        'is_all_caps',
        'char_density',
        'ratio_of_verbs',
        'is_hashed',
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
    
    # Find all CSV files
    csv_pattern = os.path.join(input_folder, '*.csv')
    input_files = glob.glob(csv_pattern)
    
    if not input_files:
        print(f"❌ No CSV files found in '{input_folder}'")
        print("Make sure you have prediction CSV files from the model testing phase.")
        return False
    
    print(f"🔍 Found {len(input_files)} CSV files to process:")
    for file in input_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    successful_files = 0
    failed_files = []
    
    for input_csv in input_files:
        filename = os.path.basename(input_csv)
        
        # Create output filename
        # Change from 'textblock_predictions_*.csv' to 'hierarchy_*.csv'
        if filename.startswith('textblock_predictions_'):
            output_filename = filename.replace('textblock_predictions_', 'hierarchy_')
        else:
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"hierarchy_{name_without_ext}.csv"
        
        output_csv = os.path.join(output_folder, output_filename)
        
        # Pass the page number column name to the processing function
        success = process_single_hierarchy_file(input_csv, output_csv, style_feature_cols, font_size_col, page_num_col)
        
        if success:
            successful_files += 1
        else:
            failed_files.append(filename)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully processed: {successful_files}/{len(input_files)} files")
    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    if successful_files > 0:
        print(f"\n💡 Results:")
        print(f"   - Hierarchy CSV files saved to '{output_folder}'")
        print(f"   - Each file contains title hierarchies (Title, H1, H2, H3, etc.)")
        print(f"   - Style clustering used {len(style_feature_cols)} features")

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