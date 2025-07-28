import pandas as pd
import os
import glob
import numpy as np
import joblib
import nltk
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

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


def load_data(input_folder):
    """Loads and combines all labeled CSV files from a directory."""
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not csv_files:
        return None
    
    all_dataframes = []
    for csv_file in csv_files:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        loaded = False
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                all_dataframes.append(df)
                loaded = True
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Could not load {os.path.basename(csv_file)}: {e}")
                break
        
        if not loaded:
            print(f"❌ Failed to load {os.path.basename(csv_file)} with any encoding")
           

    if not all_dataframes:
        return None

    df = pd.concat(all_dataframes, ignore_index=True)
    
    # Clean and prepare data
    df.dropna(subset=['text', 'title_label'], inplace=True)
    df['title_label'] = pd.to_numeric(df['title_label'], errors='coerce').fillna(0).astype(int)
    df = df[df['title_label'].isin([0, 1])]
    
    return df

def train_title_classifier(input_folder: str, model_dir: str):
    """
    Trains and evaluates multiple classifiers, prioritizing RECALL for headings.
    """
    setup_nltk()

    print(f"🚀 Loading labeled data from '{input_folder}'...")
    df = load_data(input_folder)
    if df is None or df.empty:
        print("❌ Error: No valid labeled CSV files found.")
        return
        
    print(f"Found {len(df)} labeled text blocks.")
    if len(df['title_label'].unique()) < 2:
        print("❌ Error: Dataset contains only one class. Need both titles (1) and text (0) to train.")
        return

    print("\n🛠️  Engineering advanced features...")
    df_features = advanced_feature_engineering(df)
    
    # --- UPDATED: Added new feature names ---
    feature_names = [
        'avg_font_size', 'word_count', 'char_count', 'relative_font_size',
        'is_all_caps', 'is_bold', 'is_title_case', 'ends_with_colon',
        'space_above', 'noun_count', 'verb_count', 'adj_count',
        'cardinal_num_count', 'noun_ratio', 'verb_ratio',
        'starts_with_list_pattern', # New feature
        'caps_x_font'               # New feature
    ]
    
    # Ensure all feature columns exist, fill with 0 if not
    for col in feature_names:
        if col not in df_features.columns:
            print(f"⚠️ Warning: Feature '{col}' not found. Filling with zeros.")
            df_features[col] = 0
            
    X = df_features[feature_names].fillna(0).values
    y = df_features['title_label'].values
    print(f"\n✅ Feature engineering complete. Total features: {X.shape[1]}")

    # --- Data Splitting and Scaling ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nOriginal training distribution: {Counter(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training distribution with SMOTE: {Counter(y_train_resampled)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, min_samples_leaf=3),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=4),
        "SGD Classifier": SGDClassifier(loss='log_loss', random_state=42, class_weight='balanced')
    }
    
    results = {}
    print("\n" + "="*50)
    print("--- Model Training and Evaluation (Optimizing for Recall) ---")
    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        model.fit(X_train_scaled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        
        recall = recall_score(y_test, y_pred, pos_label=1)
        results[name] = (recall, model)
        
        print(f"🎯 Heading Recall: {recall:.4f}")
        print("\nFull Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Text (0)', 'Title (1)'], zero_division=0))

    # --- Save the Best Model based on RECALL ---
    best_name = max(results, key=lambda name: results[name][0])
    best_recall, best_model = results[best_name]
    print("="*50)
    print(f"\n🏆 Best Model for Finding Headings: '{best_name}' with Recall: {best_recall:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'title_classifier.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'title_scaler.joblib'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))

    print(f"\n✅ Best model optimized for recall saved to '{model_dir}'")


if __name__ == '__main__':
    LABELED_DATA_FOLDER = '../../data/new_textblocks'
    MODEL_SAVE_DIR = '../models/textblock_models'
    
    train_title_classifier(LABELED_DATA_FOLDER, MODEL_SAVE_DIR)