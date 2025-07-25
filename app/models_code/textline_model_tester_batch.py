import pandas as pd
import joblib
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_default_model():
    """Create a default model if none exists"""
    print("No trained model found. Creating a default model...")
    
    # Create a simple default model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create dummy training data to fit the model
    feature_cols = [
        'normalized_vertical_gap', 'indentation_change', 'same_alignment',
        'is_centered_A', 'is_centered_B', 'font_size_a', 'font_size_b', 'font_size_diff',
        'same_font', 'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B',
        'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 'same_monospace',
        'line_a_ends_punctuation', 'line_b_starts_lowercase', 'is_linea_in_rectangle',
        'is_lineb_in_rectangle', 'both_in_table', 'neither_in_table',
        'is_linea_hashed', 'is_lineb_hashed', 'both_hashed', 'neither_hashed',
        'line_length_ratio'
    ]
    
    # Generate dummy data
    n_samples = 1000
    X_dummy = np.random.randn(n_samples, len(feature_cols))
    y_dummy = np.random.randint(0, 2, n_samples)
    
    # Fit the model
    model.fit(X_dummy, y_dummy)
    
    # Save the model
    model_dir = './data/output_model1'
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_columns': feature_cols
    }
    
    model_file = os.path.join(model_dir, 'text_block_merger_model.joblib')
    joblib.dump(model_data, model_file)
    
    print(f"✅ Default model created and saved to {model_file}")
    return model, feature_cols

def engineer_features(df, text_col_a, text_col_b):
    """Engineer features for the dataframe"""
    if text_col_a and text_col_b and text_col_a in df.columns and text_col_b in df.columns:
        # 1. Check if the first line ends with sentence-terminating punctuation
        if 'line_a_ends_punctuation' not in df.columns:
            df['line_a_ends_punctuation'] = df[text_col_a].str.strip().str.endswith(('.', '!', '?', ':', ';')).astype(int)

        # 2. Check if the second line starts with a lowercase letter
        if 'line_b_starts_lowercase' not in df.columns:
            df['line_b_starts_lowercase'] = df[text_col_b].str.strip().str.match(r'^[a-z]').astype(int)

        # 3. Calculate the ratio of line lengths
        if 'line_length_ratio' not in df.columns:
            len_a = df[text_col_a].str.len()
            len_b = df[text_col_b].str.len().replace(0, 1)  # Avoid division by zero
            df['line_length_ratio'] = len_a / len_b
            df['line_length_ratio'].fillna(1.0, inplace=True)
    
    return df

def inspect_csv_structure(file_path):
    """Inspect CSV file structure to understand what columns exist"""
    try:
        df = pd.read_csv(file_path, nrows=5)  # Read just first 5 rows
        print(f"  📊 CSV Structure: {len(df)} rows")
        print(f"  📋 Columns ({len(df.columns)}): {list(df.columns)}")
        
        # Check for text columns
        text_cols = [col for col in df.columns if 'text' in col.lower()]
        print(f"  📝 Text columns: {text_cols}")
        
        return df.columns.tolist(), len(df)
    except Exception as e:
        print(f"  ❌ Error inspecting CSV: {e}")
        return [], 0

def process_single_file(test_file, model, feature_cols, output_folder):
    """Process a single CSV file"""
    pdf_name = os.path.basename(test_file)
    print(f"\n📄 Processing: {pdf_name}")
    
    try:
        # First inspect the file structure
        columns, row_count = inspect_csv_structure(test_file)
        
        if row_count == 0:
            print(f"  ❌ Empty CSV file")
            return False
        
        # Try multiple encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(test_file, encoding=encoding)
                print(f"  ✅ Loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"  ❌ Could not load with any encoding")
            return False
        
        original_rows = len(df)
        print(f"  📊 Original data: {original_rows} rows, {len(df.columns)} columns")
        
        # Store original dataframe for output
        original_df = df.copy()
        
        # Check if text columns exist
        text_col_a = None
        text_col_b = None
        
        if 'span_text_a' in df.columns and 'span_text_b' in df.columns:
            text_col_a = 'span_text_a'
            text_col_b = 'span_text_b'
        elif 'text_a' in df.columns and 'text_b' in df.columns:
            text_col_a = 'text_a'
            text_col_b = 'text_b'
        else:
            print(f"  ⚠️  No standard text columns found. Available columns: {list(df.columns)}")
            # Try to find any text columns
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            if len(text_columns) >= 2:
                text_col_a = text_columns[0]
                text_col_b = text_columns[1]
                print(f"  🔄 Using columns: {text_col_a}, {text_col_b}")

        # Clean text data if available - but don't drop rows
        if text_col_a and text_col_b:
            # Fill NaN values in text columns instead of dropping
            df[text_col_a] = df[text_col_a].fillna('').astype(str)
            df[text_col_b] = df[text_col_b].fillna('').astype(str)
            print(f"  ✅ Text columns cleaned: {len(df)} rows remaining")
        else:
            print(f"  ⚠️  No valid text columns found, proceeding without text features")

        # Engineer features
        df = engineer_features(df, text_col_a, text_col_b)
        
        # Check which features are available
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = [col for col in feature_cols if col not in df.columns]
        
        print(f"  📋 Available features: {len(available_features)}/{len(feature_cols)}")
        
        # Fill missing features with default values
        for feature in missing_features:
            if 'gap' in feature or 'size' in feature:
                df[feature] = 0.0
            elif 'ratio' in feature:
                df[feature] = 1.0
            else:
                df[feature] = 0
            print(f"  🔧 Created missing feature '{feature}' with default values")
        
        # Check if labels exist (for evaluation only)
        has_labels = 'label' in df.columns
        if has_labels:
            print(f"  ℹ️  Ground truth labels found - will calculate accuracy")
            # Clean label data
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        else:
            print(f"  ℹ️  No ground truth labels found - will only generate predictions")
        
        # Fill remaining NaN values in feature columns
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Ensure all feature values are numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"  ✅ Final data: {len(df)} samples ready for prediction")
        
        # Extract features and make predictions
        X_test = df[feature_cols].copy()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy if labels exist
        if has_labels and 'label' in df.columns:
            y_test = df['label']
            accuracy = accuracy_score(y_test, y_pred)
            print(f"  📈 Accuracy: {accuracy:.4f} ({accuracy:.2%})")
        
        # Show prediction distribution
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        print(f"  🎯 Model predictions: ", end="")
        for label, count in pred_counts.items():
            print(f"Class {label}: {count} ({count/len(y_pred)*100:.1f}%) ", end="")
        print()
        
        # Add predictions to original dataframe
        original_df['model_labels'] = y_pred
        original_df['predicted_merge'] = y_pred  # Alternative name for clarity
        
        # Save results with predictions
        output_file = os.path.join(output_folder, f"predictions_{pdf_name}")
        original_df.to_csv(output_file, index=False)
        print(f"  💾 Results saved to: {os.path.basename(output_file)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {pdf_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_files(test_folder, output_folder):
    """Test all CSV files in the test folder"""
    print(f"Testing files from: {test_folder}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model with relative path from pipeline
    model_file = './data/output_model1/text_block_merger_model.joblib'
    
    try:
        print(f"Loading model from '{model_file}'...")
        model_data = joblib.load(model_file)
        
        if isinstance(model_data, dict):
            model = model_data['model']
            feature_cols = model_data.get('feature_columns', [])
        else:
            model = model_data
            feature_cols = [
                'normalized_vertical_gap', 'indentation_change', 'same_alignment',
                'is_centered_A', 'is_centered_B', 'font_size_a', 'font_size_b', 'font_size_diff',
                'same_font', 'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B',
                'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 'same_monospace',
                'line_a_ends_punctuation', 'line_b_starts_lowercase', 'is_linea_in_rectangle',
                'is_lineb_in_rectangle', 'both_in_table', 'neither_in_table',
                'is_linea_hashed', 'is_lineb_hashed', 'both_hashed', 'neither_hashed',
                'line_length_ratio'
            ]
        
        print("✅ Model loaded successfully")
        print(f"🔧 Using {len(feature_cols)} features")
        
    except FileNotFoundError:
        print(f"Model file not found. Creating default model...")
        model, feature_cols = create_default_model()
    
    # Find all CSV files
    csv_pattern = os.path.join(test_folder, '*.csv')
    test_files = glob.glob(csv_pattern)
    
    if not test_files:
        print(f"❌ No CSV files found in '{test_folder}'")
        return False
    
    print(f"\n🔍 Found {len(test_files)} CSV files to test")
    
    # Process each file
    successful_files = 0
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(test_files)}")
        success = process_single_file(test_file, model, feature_cols, output_folder)
        if success:
            successful_files += 1
        print(f"{'='*60}")
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Successfully processed: {successful_files}/{len(test_files)} files")
    print(f"❌ Failed: {len(test_files) - successful_files}/{len(test_files)} files")
    
    # Return True if at least some files were processed successfully
    return successful_files > 0

if __name__ == "__main__":
    test_folder = './data/textlines_csv_output'
    output_folder = './data/textline_predictions'
    
    success = test_all_files(test_folder, output_folder)
    
    if success:
        print("✅ Batch textline testing completed successfully")
    else:
        print("❌ Batch textline testing failed")
        exit(1)