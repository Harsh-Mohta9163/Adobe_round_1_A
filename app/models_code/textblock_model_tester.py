import pandas as pd
import os
import glob
import numpy as np
import joblib
import nltk
import re
from collections import Counter
from pathlib import Path

def setup_nltk():
    """Downloads necessary NLTK models if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        print("✅ NLTK models are ready.")
    except nltk.downloader.DownloadError:
        print("Downloading necessary NLTK models...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK models downloaded.")

def get_pos_features(text):
    """Extracts Parts-of-Speech (POS) features."""
    tokens = nltk.word_tokenize(str(text))
    tag_counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
    features = {
        'noun_count': sum(tag_counts.get(t, 0) for t in ['NN', 'NNS', 'NNP', 'NNPS']),
        'verb_count': sum(tag_counts.get(t, 0) for t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']),
        'adj_count': sum(tag_counts.get(t, 0) for t in ['JJ', 'JJR', 'JJS']),
        'cardinal_num_count': tag_counts.get('CD', 0)
    }
    return features

def advanced_feature_engineering(df):
    """
    Generates a more robust set of features for classification, including
    pattern detection and interaction features.
    MUST MATCH THE TRAINING SCRIPT'S FEATURE ENGINEERING.
    """
    # --- Basic Text and Font Features ---
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text'].apply(len)
    df['is_all_caps'] = df['text'].apply(lambda x: str(x).isupper() and len(x) > 1).astype(int)
    df['is_title_case'] = df['text'].apply(lambda x: str(x).istitle() and len(x) > 1).astype(int)
    df['ends_with_colon'] = df['text'].apply(lambda x: str(x).strip().endswith(':')).astype(int)

    # --- NEW: Pattern-Matching Feature ---
    # Looks for text starting with "1.", "A.", "i.", etc.
    df['starts_with_list_pattern'] = df['text'].apply(lambda x: 1 if re.match(r'^\s*(\d+\.|[a-zA-Z]\.)', str(x)) else 0)

    # --- Structural and Positional Features ---
    df['page_median_font'] = df.groupby('page_number')['avg_font_size'].transform('median')
    df['relative_font_size'] = df['avg_font_size'] / df['page_median_font']

    if 'normalized_vertical_gap' not in df.columns:
        df['normalized_vertical_gap'] = 0
    df['space_above'] = df['normalized_vertical_gap'].shift(-1).fillna(0)

    # --- NEW: Interaction Feature ---
    # Boost the importance of font size for all-caps text
    df['caps_x_font'] = df['is_all_caps'] * df['relative_font_size']

    # --- Part-of-Speech (POS) Features ---
    pos_df = pd.DataFrame(df['text'].apply(get_pos_features).tolist())
    df = pd.concat([df.reset_index(drop=True), pos_df], axis=1)

    df['noun_ratio'] = df['noun_count'] / (df['word_count'] + 1e-6)
    df['verb_ratio'] = df['verb_count'] / (df['word_count'] + 1e-6)

    return df

def predict_on_new_data(model_dir: str, input_csv_path: str, output_csv_path: str):
    """
    Loads a trained model, predicts labels for a new CSV file,
    and saves the output with predictions.
    """
    print("--- 🚀 Starting Model Tester ---")
    
    # --- 1. Load Model and Supporting Files ---
    try:
        print(f"Loading model from '{model_dir}'...")
        model = joblib.load(os.path.join(model_dir, 'title_classifier.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'title_scaler.joblib'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
        print("✅ Model, scaler, and feature configuration loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model files not found in '{model_dir}'.")
        print("Please run the new training script first to create an updated model.")
        return

    # --- 2. Load and Prepare New Data ---
    try:
        print(f"Loading new data from '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path, encoding='utf-8')
        df.dropna(subset=['text'], inplace=True)
        original_df = df.copy() # Keep a copy for final output
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_csv_path}'.")
        return

    # --- 3. Engineer Features (must match training process exactly) ---
    print("\n🛠️  Engineering features for the new data...")
    setup_nltk()
    
    df_features = advanced_feature_engineering(df)
    
    # Ensure all feature columns exist, fill with 0 if not
    for col in feature_names:
        if col not in df_features.columns:
            print(f"⚠️ Warning: Feature '{col}' not in CSV. Using zeros as a placeholder.")
            df_features[col] = 0

    X_new = df_features[feature_names].fillna(0).values
    print("✅ Feature engineering complete.")

    # --- 4. Scale Data and Make Predictions ---
    print("Scaling data and making predictions...")
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # --- 5. Prepare and Save Output ---
    print("Preparing output file...")
    output_df = original_df.copy()
    output_df['model_labels'] = predictions
    output_df['confidence_score'] = np.max(probabilities, axis=1)

    # --- ✅ START: POST-PROCESSING RULES TO FIX INCORRECT PREDICTIONS ---
    
    # Rule 1: Correct predictions where the model identified a heading (1)
    # but the text starts with a lowercase letter.
    condition_lowercase = (output_df['model_labels'] == 1) & \
                         (output_df['text'].apply(lambda x: str(x).strip() and str(x).strip()[0].islower()))

    # Rule 2: Correct predictions where the model identified a heading (1)
    # but the text ends with a period.
    condition_ends_period = (output_df['model_labels'] == 1) & \
                           (output_df['text'].apply(lambda x: str(x).strip().endswith('.')))

    # Combine both conditions
    condition_to_correct = condition_lowercase | condition_ends_period

    # If any such rows exist, apply the correction
    if condition_to_correct.sum() > 0:
        lowercase_count = condition_lowercase.sum()
        period_count = condition_ends_period.sum()
        total_count = condition_to_correct.sum()
        
        print(f"⚙️  Applying correction rules:")
        if lowercase_count > 0:
            print(f"   - Overriding {lowercase_count} heading prediction(s) that start with lowercase")
        if period_count > 0:
            print(f"   - Overriding {period_count} heading prediction(s) that end with period")
        print(f"   - Total corrections: {total_count}")
        
        # Invert the confidence score for the corrected rows
        output_df.loc[condition_to_correct, 'confidence_score'] = 1.0 - output_df.loc[condition_to_correct, 'confidence_score']
        
        # Change the label from 1 (Title/Heading) to 0 (Text/Paragraph)
        output_df.loc[condition_to_correct, 'model_labels'] = 0
    # --- ✅ END: POST-PROCESSING RULES ---

    # Map the final labels to human-readable categories
    output_df['predicted_category'] = output_df['model_labels'].map({0: 'Text/Paragraph', 1: 'Title/Heading'})
    
    try:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\n💾 Predictions successfully saved to: {output_csv_path}")
    except Exception as e:
        print(f"\n❌ Error saving output CSV: {e}")

    # --- 6. Display Results in Console ---
    print("\n\n--- ✅ Prediction Results (Console Preview) ---")
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', 120)
    
    preview_df = output_df.head(15)
    for index, row in preview_df.iterrows():
        print("-" * 100)
        true_label = "Title" if 'title_label' in row and row['title_label'] == 1 else "Text"
        print(f"Text: {row['text']}")
        print(f"  ➡️ Predicted: \033[1m{row['predicted_category']}\033[0m (Confidence: {row['confidence_score']:.2%}) --- [Ground Truth: {true_label}]")
        
    print("-" * 100)
    if len(df) > 15:
        print(f"... and {len(df) - 15} more rows in the output file.")
    print("\n--- Tester Finished ---")

if __name__ == '__main__':
    MODEL_DIRECTORY = '../models/textblock_models'
    INPUT_CSV_TO_TEST = '../../data/test_labelled_merged_textblocks_gt/merged_textblocks_file01.pdf.csv' 
    OUTPUT_CSV_WITH_PREDICTIONS = './predictions/predicted_output_advanced.csv'
    
    if not os.path.exists(MODEL_DIRECTORY):
        print(f"Error: The model directory '{MODEL_DIRECTORY}' does not exist. Please re-train the model with the new script.")
    elif not os.path.exists(INPUT_CSV_TO_TEST):
        print(f"Error: The input file '{INPUT_CSV_TO_TEST}' does not exist. Please provide a valid CSV file.")
    else:
        predict_on_new_data(MODEL_DIRECTORY, INPUT_CSV_TO_TEST, OUTPUT_CSV_WITH_PREDICTIONS)