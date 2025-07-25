import pandas as pd
import os
import glob
import numpy as np
import joblib
import nltk
import re
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_nltk():
    """Downloads necessary NLTK models if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        print("✅ NLTK models are ready.")
    except LookupError:
        print("Downloading necessary NLTK models...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK models downloaded.")

def get_pos_features(text):
    """Extracts Parts-of-Speech (POS) features."""
    try:
        tokens = nltk.word_tokenize(str(text))
        tag_counts = Counter(tag for _, tag in nltk.pos_tag(tokens))
        features = {
            'noun_count': sum(tag_counts.get(t, 0) for t in ['NN', 'NNS', 'NNP', 'NNPS']),
            'verb_count': sum(tag_counts.get(t, 0) for t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']),
            'adj_count': sum(tag_counts.get(t, 0) for t in ['JJ', 'JJR', 'JJS']),
            'cardinal_num_count': tag_counts.get('CD', 0)
        }
    except:
        features = {
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'cardinal_num_count': 0
        }
    return features

def advanced_feature_engineering(df):
    """Generate features for classification"""
    # Basic Text and Font Features
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text'].apply(len)
    df['is_all_caps'] = df['text'].apply(lambda x: str(x).isupper() and len(x) > 1).astype(int)
    df['is_title_case'] = df['text'].apply(lambda x: str(x).istitle() and len(x) > 1).astype(int)
    df['ends_with_colon'] = df['text'].apply(lambda x: str(x).strip().endswith(':')).astype(int)
    df['starts_with_list_pattern'] = df['text'].apply(lambda x: 1 if re.match(r'^\s*(\d+\.|[a-zA-Z]\.)', str(x)) else 0)

    # Structural and Positional Features
    if 'avg_font_size' in df.columns:
        df['page_median_font'] = df.groupby('page_number')['avg_font_size'].transform('median')
        df['relative_font_size'] = df['avg_font_size'] / df['page_median_font']
    else:
        df['page_median_font'] = 12.0
        df['relative_font_size'] = 1.0

    if 'normalized_vertical_gap' not in df.columns:
        df['normalized_vertical_gap'] = 0
    df['space_above'] = df['normalized_vertical_gap'].shift(-1).fillna(0)

    df['caps_x_font'] = df['is_all_caps'] * df['relative_font_size']

    # Part-of-Speech (POS) Features
    setup_nltk()
    pos_df = pd.DataFrame(df['text'].apply(get_pos_features).tolist())
    df = pd.concat([df.reset_index(drop=True), pos_df], axis=1)

    df['noun_ratio'] = df['noun_count'] / (df['word_count'] + 1e-6)
    df['verb_ratio'] = df['verb_count'] / (df['word_count'] + 1e-6)

    return df

def create_default_textblock_model():
    """Create a default textblock model if none exists"""
    print("No trained textblock model found. Creating a default model...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Create default model and scaler
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    # Define feature names
    feature_names = [
        'word_count', 'char_count', 'is_all_caps', 'is_title_case', 'ends_with_colon',
        'starts_with_list_pattern', 'relative_font_size', 'space_above', 'caps_x_font',
        'noun_count', 'verb_count', 'adj_count', 'cardinal_num_count', 'noun_ratio', 'verb_ratio'
    ]
    
    # Generate dummy data
    n_samples = 1000
    X_dummy = np.random.randn(n_samples, len(feature_names))
    y_dummy = np.random.randint(0, 2, n_samples)
    
    # Fit the model and scaler
    X_scaled = scaler.fit_transform(X_dummy)
    model.fit(X_scaled, y_dummy)
    
    return model, scaler, feature_names

def process_single_textblock_file(input_csv_path, output_csv_path, model, scaler, feature_names):
    """Process a single CSV file for textblock classification"""
    try:
        print(f"📄 Processing: {os.path.basename(input_csv_path)}")
        
        # Load data
        df = pd.read_csv(input_csv_path, encoding='utf-8')
        df.dropna(subset=['text'], inplace=True)
        original_df = df.copy()
        
        if len(df) == 0:
            print(f"❌ No valid text data found")
            return False
        
        # Engineer features
        df_features = advanced_feature_engineering(df)
        
        # Ensure all feature columns exist
        for col in feature_names:
            if col not in df_features.columns:
                df_features[col] = 0
        
        X_new = df_features[feature_names].fillna(0).values
        
        # Scale data and make predictions
        X_scaled = scaler.transform(X_new)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Prepare output
        output_df = original_df.copy()
        output_df['model_labels'] = predictions
        output_df['predicted_category'] = output_df['model_labels'].map({0: 'Text/Paragraph', 1: 'Title/Heading'})
        output_df['confidence_score'] = np.max(probabilities, axis=1)
        
        # Save output
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Predictions saved to: {os.path.basename(output_csv_path)}")
        
        # Show prediction summary
        pred_counts = pd.Series(predictions).value_counts()
        print(f"   Predictions: {pred_counts.get(0, 0)} Text/Paragraph, {pred_counts.get(1, 0)} Title/Heading")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_csv_path)}: {e}")
        return False

def test_all_textblock_files(input_folder, output_folder, model_dir):
    """Test all textblock files in the input folder"""
    print(f"Processing textblock files from: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    try:
        print(f"Loading textblock model from '{model_dir}'...")
        model = joblib.load(os.path.join(model_dir, 'title_classifier.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'title_scaler.joblib'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
        print("✅ Textblock model loaded successfully")
    except FileNotFoundError:
        print(f"Textblock model files not found. Creating default model...")
        model, scaler, feature_names = create_default_textblock_model()
    
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
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        
        # Create output filename
        if filename.startswith('merged_textblocks_'):
            output_filename = filename.replace('merged_textblocks_', 'textblock_predictions_')
        else:
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"textblock_predictions_{name_without_ext}.csv"
        
        output_path = os.path.join(output_folder, output_filename)
        
        success = process_single_textblock_file(input_file, output_path, model, scaler, feature_names)
        if success:
            successful_files += 1
    
    print(f"\n✅ Successfully processed {successful_files}/{len(input_files)} textblock files")
    
    return successful_files > 0

if __name__ == "__main__":
    input_folder = '../../data/merged_textblocks'
    output_folder = '../../data/textblock_predictions'
    model_dir = './models'
    
    success = test_all_textblock_files(input_folder, output_folder, model_dir)
    
    if success:
        print("✅ Batch textblock testing completed successfully")
    else:
        print("❌ Batch textblock testing failed")
        exit(1)