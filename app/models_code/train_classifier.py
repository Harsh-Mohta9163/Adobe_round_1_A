import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
            # If it's not an encoding error, report it and continue
            if encoding == encodings_to_try[-1]:  # Last encoding attempt
                print(f"❌ Error loading {os.path.basename(csv_file)}: {e}")
            continue
    
    print(f"❌ Could not load {os.path.basename(csv_file)} with any encoding")
    return None, None

def train_title_classifier(input_folder: str, model_dir: str):
    """
    Trains a lightweight SGD classifier using only structural features.
    No embeddings needed - works completely offline.
    """
    print("🚀 Loading data for lightweight SGD classifier...")
    
    # Find all CSV files in the input folder
    csv_pattern = os.path.join(input_folder, '*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found in '{input_folder}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to load:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load and combine all CSV files with multiple encoding support
    all_dataframes = []
    successful_files = 0
    failed_files = []
    
    for csv_file in csv_files:
        df, encoding_used = load_csv_with_multiple_encodings(csv_file)
        
        if df is not None:
            # Only include files that have the title_label column filled
            if 'title_label' in df.columns and not df['title_label'].isna().all():
                # Clean text data to handle any remaining encoding issues
                if 'text' in df.columns:
                    df['text'] = df['text'].astype(str)
                    # Remove any problematic characters
                    df['text'] = df['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else str(x))
                
                all_dataframes.append(df)
                successful_files += 1
                print(f"✅ Added {os.path.basename(csv_file)} to training data")
            else:
                print(f"⚠️  Skipped {os.path.basename(csv_file)} - no labels found")
        else:
            failed_files.append(os.path.basename(csv_file))
    
    print(f"\n📊 File loading summary:")
    print(f"✅ Successfully loaded: {successful_files} files")
    if failed_files:
        print(f"❌ Failed to load: {len(failed_files)} files")
        for file in failed_files:
            print(f"   - {file}")
    
    if not all_dataframes:
        print("Error: No valid labeled CSV files found.")
        print("Please make sure your CSV files have labeled 'title_label' column with values 0 or 1.")
        return
    
    # Combine all dataframes
    df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined dataset: {len(df)} total rows")
    print(f"Available columns: {list(df.columns)}")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Clean data
    print("Cleaning data...")
    original_len = len(df)
    df.dropna(subset=['title_label', 'text'], inplace=True)
    df['text'] = df['text'].astype(str)
    
    # Handle title_label conversion more robustly
    try:
        df['title_label'] = pd.to_numeric(df['title_label'], errors='coerce')
        df = df.dropna(subset=['title_label'])  # Remove rows where conversion failed
        df['title_label'] = df['title_label'].astype(int)
    except Exception as e:
        print(f"Warning: Issue converting title_label column: {e}")
        return
    
    # Filter out invalid labels
    df = df[df['title_label'].isin([0, 1])]
    
    print(f"After cleaning: {len(df)} rows (dropped {original_len - len(df)} rows)")
    
    if len(df) == 0:
        print("Error: No valid labeled data found after cleaning.")
        return
    
    # Check label distribution
    label_counts = df['title_label'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        label_name = "Text/Paragraph" if label == 0 else "Title/Heading"
        print(f"  Label {label} ({label_name}): {count} samples ({count/len(df)*100:.1f}%)")
    
    if len(label_counts) < 2:
        print("Error: Need at least 2 different labels (title and text) to train classifier.")
        return

    # Feature Engineering (Lightweight - No Embeddings)
    print("Engineering structural features...")
    
    feature_list = []
    
    # 1. Basic CSV features (with error handling) - REMOVED is_all_caps
    try:
        basic_feature_columns = ['avg_font_size', 'word_count', 'char_density', 
                               'ratio_of_verbs', 'ratio_capitalized']
        # Check which columns exist
        available_basic_cols = [col for col in basic_feature_columns if col in df.columns]
        if not available_basic_cols:
            print("Warning: No basic feature columns found")
        else:
            basic_features = df[available_basic_cols].fillna(0).values  # Fill NaN with 0
            feature_list.append(basic_features)
            print(f"✅ Basic features: {len(available_basic_cols)} columns")
    except Exception as e:
        print(f"Warning: Error processing basic features: {e}")
    
    # 2. Bbox features (with error handling)
    try:
        if 'bbox' in df.columns:
            bbox_features = np.array([extract_bbox_features(bbox) for bbox in df['bbox']])
            feature_list.append(bbox_features)
            print(f"✅ Bbox features: 3 columns")
        else:
            print("Warning: No bbox column found")
    except Exception as e:
        print(f"Warning: Error processing bbox features: {e}")
    
    # 3. Enhanced text features (with error handling)
    try:
        text_features = []
        for _, row in df.iterrows():
            word_count = row.get('word_count', 1)
            avg_font_size = row.get('avg_font_size', 12.0)
            text_feats = extract_text_features(row['text'], word_count, avg_font_size)
            text_features.append(text_feats)
        
        text_features = np.array(text_features)
        feature_list.append(text_features)
        print(f"✅ Text features: 18 columns")
    except Exception as e:
        print(f"Warning: Error processing text features: {e}")
    
    if not feature_list:
        print("Error: No features could be extracted")
        return
    
    # Combine all features
    X = np.hstack(feature_list)
    y = df['title_label'].values
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Features breakdown:")
    print(f"  - Basic CSV features: {basic_features.shape[1] if 'basic_features' in locals() else 0}")
    print(f"  - Bbox features: {bbox_features.shape[1] if 'bbox_features' in locals() else 0}") 
    print(f"  - Enhanced text features: {text_features.shape[1] if 'text_features' in locals() else 0}")
    print(f"  - Total features: {X.shape[1]}")

    # Train-Test Split
    print("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SGD Classifier
    clf = SGDClassifier(
        loss='log_loss',  # Logistic regression
        random_state=42,
        class_weight='balanced',
        max_iter=2000,
        alpha=0.0001  # Regularization
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluation
    train_accuracy = clf.score(X_train_scaled, y_train)
    test_accuracy = clf.score(X_test_scaled, y_test)
    
    print(f"\n🎯 Model Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed evaluation
    y_pred = clf.predict(X_test_scaled)
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Text/Paragraph', 'Title/Heading']))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance - REMOVED is_all_caps from feature names
    if hasattr(clf, 'coef_'):
        feature_names = []
        if 'basic_features' in locals():
            feature_names.extend(['avg_font_size', 'word_count', 'char_density', 'ratio_of_verbs', 'ratio_capitalized'])
        if 'bbox_features' in locals():
            feature_names.extend(['width', 'height', 'aspect_ratio'])
        if 'text_features' in locals():
            feature_names.extend([
                'text_len', 'char_count', 'avg_word_len', 'is_large_font', 'is_small_font', 'is_very_large_font',
                'has_numbers', 'has_special_chars', 'has_punctuation', 'is_very_short', 'is_short_text',
                'is_medium_text', 'is_long_text', 'is_title_case', 'has_uppercase_words',
                'starts_with_number', 'ends_with_colon', 'ends_with_period'
            ])
        
        feature_importance = np.abs(clf.coef_[0])
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        print(f"\n📊 Top 10 Most Important Features:")
        for i, idx in enumerate(top_features_idx):
            if idx < len(feature_names):
                print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

    # Save models
    model_path = os.path.join(model_dir, 'title_classifier_lightweight.joblib')
    scaler_path = os.path.join(model_dir, 'title_scaler_lightweight.joblib')
    config_path = os.path.join(model_dir, 'feature_config_lightweight.joblib')
    
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save feature configuration
    feature_config = {
        'feature_names': feature_names if 'feature_names' in locals() else [],
        'feature_count': X.shape[1],
        'model_type': 'SGD_lightweight',
        'uses_embeddings': False,
        'offline_ready': True,
        'successful_files': successful_files,
        'failed_files': failed_files
    }
    joblib.dump(feature_config, config_path)

    print(f"\n✅ Model saved successfully!")
    print(f"📁 Model: {model_path}")
    print(f"📁 Scaler: {scaler_path}")
    print(f"📁 Config: {config_path}")
    
    # File sizes
    model_size = os.path.getsize(model_path) / 1024  # KB
    scaler_size = os.path.getsize(scaler_path) / 1024  # KB
    total_size = model_size + scaler_size
    
    print(f"\n📊 Model Statistics:")
    print(f"💾 Model size: {model_size:.1f} KB")
    print(f"💾 Scaler size: {scaler_size:.1f} KB")
    print(f"💾 Total size: {total_size:.1f} KB")
    print(f"🌐 Offline ready: ✅ Yes")
    print(f"🔧 Dependencies: scikit-learn, pandas, numpy only")

if __name__ == '__main__':
    INPUT_FOLDER = '../data/merged_textblocks_gt'
    MODEL_SAVE_DIR = '../models'
    
    print("🚀 Starting Lightweight Title Classifier Training")
    print("📋 Features: Structural only (no embeddings)")
    print("💾 Model: SGD Classifier")
    print("🌐 Offline: Completely offline")
    print("🔧 Encoding: Multi-encoding support (utf-8, latin-1, cp1252, etc.)")
    print(f"📁 Input folder: {INPUT_FOLDER}")
    print(f"💾 Model save directory: {MODEL_SAVE_DIR}")
    print("\n" + "="*60)
    
    train_title_classifier(INPUT_FOLDER, MODEL_SAVE_DIR)
    
    print("\n" + "="*60)
    print("🎯 Training completed!")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the model performance above")
    print(f"   2. Test the model on new documents")
    print(f"   3. Use the model for title/heading detection")
    print(f"   4. Model works completely offline!")