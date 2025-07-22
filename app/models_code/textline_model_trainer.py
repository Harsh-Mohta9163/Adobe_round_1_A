import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def load_and_prepare_data(folder_path='.'):
    """
    Loads all CSV files from a folder, combines them, and cleans the data.
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {folder_path}")
    
    print(f"Found {len(csv_files)} CSV files to load: {csv_files}")

    # Try multiple encodings to handle different file formats
    df_list = []
    for f in csv_files:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(f, encoding=encoding)
                print(f"Successfully loaded {f} with {encoding} encoding")
                df_list.append(df)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Warning: Could not load {f} with any encoding")
    
    if not df_list:
        raise ValueError("No CSV files could be loaded with available encodings")
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # --- Data Cleaning ---
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)
    
    print(f"Rows after dropping missing labels: {len(df)}")
    
    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    print(f"Label distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")
    
    return df

try:
    # Load data from textlines_csv_output folder
    data = load_and_prepare_data('../../data/labelled_textlines')
    
    # --- Define Features (X) and Target (y) ---
    # Including all features except span_text_a and span_text_b
    feature_cols = [
        'normalized_vertical_gap', 'indentation_change', 'same_alignment',
        'is_centered_A', 'is_centered_B', 'font_size_a', 'font_size_b', 'font_size_diff',
        'same_font', 'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B',
        'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 'same_monospace',
        'line_a_ends_punctuation', 'line_b_starts_lowercase', 'is_linea_in_rectangle',
        'is_lineb_in_rectangle', 'both_in_table', 'neither_in_table',
        'is_linea_hashed', 'is_lineb_hashed', 'both_hashed', 'neither_hashed'
    ]
    
    X = data[feature_cols]
    y = data['label']

    # Check if we have enough samples for stratified split
    label_counts = y.value_counts()
    min_samples = label_counts.min()
    
    print(f"Minimum samples in any class: {min_samples}")
    
    if min_samples < 2:
        print("⚠️  Warning: One class has fewer than 2 samples. Cannot use stratified split.")
        print("Using random split instead...")
        # Use random split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
    else:
        print("Using stratified split to maintain class proportions...")
        # Use stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    # Show label distribution in train/test sets
    print(f"Training set label distribution:")
    train_counts = y_train.value_counts().sort_index()
    for label, count in train_counts.items():
        print(f"  Label {label}: {count} samples")
    
    print(f"Testing set label distribution:")
    test_counts = y_test.value_counts().sort_index()
    for label, count in test_counts.items():
        print(f"  Label {label}: {count} samples")

    # --- Initialize and Train the RandomForestClassifier ---
    print("\nTraining the RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete. ✅")

    # --- Make Predictions and Evaluate the Model ---
    print("\n--- Model Performance on Test Data ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    # Only show detailed classification report if we have both classes in test set
    unique_test_labels = sorted(y_test.unique())
    if len(unique_test_labels) > 1:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['block_start (0)', 'block_in (1)']))
        print("\nConfusion Matrix:")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                         index=['Actual Start', 'Actual In'],
                         columns=['Predicted Start', 'Predicted In']))
    else:
        print(f"⚠️  Test set only contains class {unique_test_labels[0]}, skipping detailed evaluation.")

    # --- Feature Importance ---
    print("\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False))

    # --- Save the Trained Model ---
    model_filename = '../../data/output_model1/text_block_merger_model.joblib'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    
    joblib.dump({
        'model': model,
        'feature_columns': feature_cols
    }, model_filename)
    print(f"\n✅ Model successfully saved to '{model_filename}'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")