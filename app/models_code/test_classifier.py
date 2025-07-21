import pandas as pd
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import joblib
import ast

def safe_eval_bbox(bbox_str):
    """Safely evaluates a string representation of a bbox list."""
    try:
        return ast.literal_eval(str(bbox_str))
    except (ValueError, SyntaxError):
        return [0, 0, 100, 20]

def extract_bbox_features(bbox_list):
    """Extract additional features from bounding box."""
    bbox = safe_eval_bbox(bbox_list)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect_ratio = width / height if height > 0 else 1.0
    return [width, height, aspect_ratio]

def extract_text_features(text, word_count, avg_font_size):
    """Extract text-based features without embeddings."""
    text = str(text)
    
    # Basic text statistics
    text_len = len(text)
    char_count = len(text.replace(" ", ""))
    avg_word_len = text_len / max(word_count, 1)
    
    # Font size features
    is_large_font = 1 if avg_font_size > 12 else 0
    is_small_font = 1 if avg_font_size < 10 else 0
    is_very_large_font = 1 if avg_font_size > 16 else 0
    
    # Text pattern features
    has_numbers = 1 if any(c.isdigit() for c in text) else 0
    has_special_chars = 1 if any(c in '()[]{}:;-_' for c in text) else 0
    has_punctuation = 1 if any(c in '.!?,' for c in text) else 0
    
    # Length-based features
    is_very_short = 1 if word_count <= 3 else 0
    is_short_text = 1 if word_count <= 5 else 0
    is_medium_text = 1 if 6 <= word_count <= 15 else 0
    is_long_text = 1 if word_count >= 20 else 0
    
    # Case features
    is_title_case = 1 if text.istitle() else 0
    has_uppercase_words = 1 if any(word.isupper() for word in text.split()) else 0
    
    # Starting/ending patterns
    starts_with_number = 1 if text.strip() and text.strip()[0].isdigit() else 0
    ends_with_colon = 1 if text.strip().endswith(':') else 0
    ends_with_period = 1 if text.strip().endswith('.') else 0
    
    return [
        text_len, char_count, avg_word_len, is_large_font, is_small_font, is_very_large_font,
        has_numbers, has_special_chars, has_punctuation, is_very_short, is_short_text, 
        is_medium_text, is_long_text, is_title_case, has_uppercase_words, 
        starts_with_number, ends_with_colon, ends_with_period
    ]

def load_csv_with_multiple_encodings(csv_file):
    """
    Try to load CSV with multiple encodings to handle different character sets.
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'ascii']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"✅ Loaded {os.path.basename(csv_file)} with {encoding} encoding - {len(df)} rows")
            return df, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings_to_try[-1]:  # Last encoding attempt
                print(f"❌ Error loading {os.path.basename(csv_file)}: {e}")
            continue
    
    print(f"❌ Could not load {os.path.basename(csv_file)} with any encoding")
    return None, None

def extract_features_from_dataframe(df):
    """
    Extract features from a dataframe using the same logic as training.
    """
    feature_list = []
    
    # 1. Basic CSV features - REMOVED is_all_caps
    try:
        basic_feature_columns = ['avg_font_size', 'word_count', 'char_density', 
                               'ratio_of_verbs', 'ratio_capitalized']
        available_basic_cols = [col for col in basic_feature_columns if col in df.columns]
        if available_basic_cols:
            basic_features = df[available_basic_cols].fillna(0).values
            feature_list.append(basic_features)
            print(f"  ✅ Basic features: {len(available_basic_cols)} columns")
        else:
            print("  ⚠️  No basic feature columns found")
    except Exception as e:
        print(f"  ⚠️  Error processing basic features: {e}")
    
    # 2. Bbox features
    try:
        if 'bbox' in df.columns:
            bbox_features = np.array([extract_bbox_features(bbox) for bbox in df['bbox']])
            feature_list.append(bbox_features)
            print(f"  ✅ Bbox features: 3 columns")
        else:
            print("  ⚠️  No bbox column found")
    except Exception as e:
        print(f"  ⚠️  Error processing bbox features: {e}")
    
    # 3. Text features
    try:
        text_features = []
        for _, row in df.iterrows():
            word_count = row.get('word_count', 1)
            avg_font_size = row.get('avg_font_size', 12.0)
            text_feats = extract_text_features(row['text'], word_count, avg_font_size)
            text_features.append(text_feats)
        
        text_features = np.array(text_features)
        feature_list.append(text_features)
        print(f"  ✅ Text features: 18 columns")
    except Exception as e:
        print(f"  ⚠️  Error processing text features: {e}")
    
    if not feature_list:
        print("  ❌ No features could be extracted")
        return None
    
    # Combine all features
    X = np.hstack(feature_list)
    print(f"  📊 Total features extracted: {X.shape[1]}")
    return X

def test_title_classifier(model_dir: str, test_folder: str, output_folder: str = None):
    """
    Test the trained title classifier on multiple CSV files.
    """
    print("🔍 Loading trained model...")
    
    # Load trained model, scaler, and config
    model_path = os.path.join(model_dir, 'title_classifier_lightweight.joblib')
    scaler_path = os.path.join(model_dir, 'title_scaler_lightweight.joblib')
    config_path = os.path.join(model_dir, 'feature_config_lightweight.joblib')
    
    if not all(os.path.exists(path) for path in [model_path, scaler_path, config_path]):
        print(f"❌ Model files not found in '{model_dir}'")
        print("Expected files:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - {config_path}")
        return
    
    try:
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        config = joblib.load(config_path)
        print(f"✅ Model loaded successfully")
        print(f"📋 Model type: {config.get('model_type', 'Unknown')}")
        print(f"📊 Expected features: {config.get('feature_count', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find test CSV files
    csv_pattern = os.path.join(test_folder, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{test_folder}'")
        return
    
    print(f"\n🔍 Found {len(csv_files)} CSV files to test:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Test each file
    all_results = {}
    total_correct = 0
    total_samples = 0
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n{'='*60}")
        print(f"📄 Testing: {filename}")
        print(f"{'='*60}")
        
        # Load test data
        df, encoding_used = load_csv_with_multiple_encodings(csv_file)
        
        if df is None:
            print(f"❌ Failed to load {filename}")
            continue
        
        # Clean text data
        if 'text' in df.columns:
            df['text'] = df['text'].astype(str)
            df['text'] = df['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else str(x))
        
        # Check if we have ground truth labels
        has_labels = 'title_label' in df.columns and not df['title_label'].isna().all()
        
        if has_labels:
            # Clean labels
            df['title_label'] = pd.to_numeric(df['title_label'], errors='coerce')
            df = df.dropna(subset=['title_label', 'text'])
            df['title_label'] = df['title_label'].astype(int)
            df = df[df['title_label'].isin([0, 1])]
            
            if len(df) == 0:
                print(f"❌ No valid labeled data in {filename}")
                continue
        else:
            # No labels - just predict
            df = df.dropna(subset=['text'])
            print("ℹ️  No ground truth labels found - will only show predictions")
        
        print(f"📊 Processing {len(df)} rows")
        
        # Extract features
        print("🔧 Extracting features...")
        X = extract_features_from_dataframe(df)
        
        if X is None:
            print(f"❌ Failed to extract features from {filename}")
            continue
        
        # Check feature count
        if X.shape[1] != config.get('feature_count', X.shape[1]):
            print(f"⚠️  Feature count mismatch: expected {config.get('feature_count')}, got {X.shape[1]}")
        
        # Scale features and predict
        try:
            X_scaled = scaler.transform(X)
            predictions = clf.predict(X_scaled)
            prediction_probs = clf.predict_proba(X_scaled)
            
            print(f"✅ Predictions generated")
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            continue
        
        # Add predictions to dataframe
        df['predicted_label'] = predictions
        df['model_labels'] = predictions  # Add model_labels column with same values as predicted_label
        df['title_probability'] = prediction_probs[:, 1]  # Probability of being a title
        
        # Show results
        prediction_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"\n📊 Prediction Summary:")
        for label, count in prediction_counts.items():
            label_name = "Text/Paragraph" if label == 0 else "Title/Heading"
            print(f"  {label_name}: {count} samples ({count/len(predictions)*100:.1f}%)")
        
        if has_labels:
            # Evaluate against ground truth
            y_true = df['title_label'].values
            accuracy = accuracy_score(y_true, predictions)
            
            print(f"\n🎯 Performance Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_true, predictions, target_names=['Text/Paragraph', 'Title/Heading']))
            
            print(f"\nConfusion Matrix:")
            cm = confusion_matrix(y_true, predictions)
            print(cm)
            
            # Store results
            all_results[filename] = {
                'accuracy': accuracy,
                'total_samples': len(df),
                'correct_predictions': int(accuracy * len(df))
            }
            
            total_correct += int(accuracy * len(df))
            total_samples += len(df)
        
        # Show some examples
        print(f"\n📝 Sample Predictions:")
        if has_labels:
            # Show some correct and incorrect predictions
            df['correct'] = df['title_label'] == df['predicted_label']
            
            print("\n✅ Correctly predicted titles:")
            correct_titles = df[(df['predicted_label'] == 1) & (df['correct'] == True)].head(3)
            for idx, row in correct_titles.iterrows():
                print(f"  \"{row['text'][:80]}...\" (prob: {row['title_probability']:.3f})")
            
            print("\n❌ Incorrectly predicted:")
            incorrect = df[df['correct'] == False].head(3)
            for idx, row in incorrect.iterrows():
                true_label = "Title" if row['title_label'] == 1 else "Text"
                pred_label = "Title" if row['predicted_label'] == 1 else "Text"
                print(f"  \"{row['text'][:60]}...\" (True: {true_label}, Pred: {pred_label}, prob: {row['title_probability']:.3f})")
        else:
            # Just show high-confidence predictions
            high_conf_titles = df[df['title_probability'] > 0.8].head(5)
            print(f"\n🎯 High-confidence title predictions:")
            for idx, row in high_conf_titles.iterrows():
                print(f"  \"{row['text'][:80]}...\" (prob: {row['title_probability']:.3f})")
        
        # Save results if output folder specified
        if output_folder:
            output_file = os.path.join(output_folder, f"predictions_{filename}")
            df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")
    
    # Overall summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"📊 OVERALL TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Files tested: {len(all_results)}")
        print(f"Total samples: {total_samples}")
        print(f"Overall accuracy: {total_correct/total_samples:.4f}")
        
        print(f"\nPer-file results:")
        for filename, results in all_results.items():
            print(f"  {filename}: {results['accuracy']:.4f} ({results['correct_predictions']}/{results['total_samples']})")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    MODEL_DIR = '../models'  # Directory with trained model
    TEST_FOLDER = '../data/test_textblocks'  # Folder with test CSV files
    OUTPUT_FOLDER = '../data/test_results'  # Optional: save predictions
    # -------------------
    
    print("🧪 Starting Title Classifier Testing")
    print("📋 Mode: Testing only (no training)")
    print("🔧 Multi-encoding support")
    print(f"📁 Model directory: {MODEL_DIR}")
    print(f"📁 Test folder: {TEST_FOLDER}")
    print(f"💾 Output folder: {OUTPUT_FOLDER}")
    print("\n" + "="*60)
    
    test_title_classifier(MODEL_DIR, TEST_FOLDER, OUTPUT_FOLDER)
    
    print("\n" + "="*60)
    print("🧪 Testing completed!")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the performance metrics above")
    print(f"   2. Check the prediction files in '{OUTPUT_FOLDER}'")
    print(f"   3. Analyze incorrectly predicted samples")
    print(f"   4. Consider retraining if accuracy is low")