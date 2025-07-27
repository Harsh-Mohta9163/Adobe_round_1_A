import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np # Import numpy for handling potential division by zero

def load_and_prepare_data(folder_path='.'):
    """
    Loads all CSV files from a folder, combines them, cleans the data,
    and engineers new features.
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {folder_path}")

    print(f"Found {len(csv_files)} CSV files to load: {csv_files}")

    df_list = []
    for f in csv_files:
        # Using 'latin-1' as a robust fallback encoding
        try:
            df = pd.read_csv(f, encoding='utf-8')
            print(f"Successfully loaded {f} with utf-8 encoding")
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, encoding='latin-1')
                print(f"Successfully loaded {f} with latin-1 encoding")
                df_list.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f}. Error: {e}")

    if not df_list:
        raise ValueError("No CSV files could be loaded.")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    # --- Data Cleaning ---
    df.dropna(subset=['label'], inplace=True)
    
    # Check if text columns exist before cleaning them
    if 'span_text_a' in df.columns and 'span_text_b' in df.columns:
        df.dropna(subset=['span_text_a', 'span_text_b'], inplace=True)
        df['span_text_a'] = df['span_text_a'].astype(str)
        df['span_text_b'] = df['span_text_b'].astype(str)
        text_col_a = 'span_text_a'
        text_col_b = 'span_text_b'
    elif 'text_a' in df.columns and 'text_b' in df.columns:
        df.dropna(subset=['text_a', 'text_b'], inplace=True)
        df['text_a'] = df['text_a'].astype(str)
        df['text_b'] = df['text_b'].astype(str)
        text_col_a = 'text_a'
        text_col_b = 'text_b'
    else:
        print("Warning: No text columns found for feature engineering")
        text_col_a = None
        text_col_b = None
    
    df['label'] = df['label'].astype(int)

    print(f"Rows after dropping missing labels: {len(df)}")

    # --- Feature Engineering ---
    if text_col_a and text_col_b:
        print("\nEngineering new features...")
        # 1. Check if the first line ends with sentence-terminating punctuation
        if 'line_a_ends_punctuation' not in df.columns:
            df['line_a_ends_punctuation'] = df[text_col_a].str.strip().str.endswith(('.', '!', '?', ':', ';')).astype(int)

        # 2. Check if the second line starts with a capital letter
        if 'line_b_starts_lowercase' not in df.columns:
            df['line_b_starts_lowercase'] = df[text_col_b].str.strip().str.match(r'^[a-z]').astype(int)

        # 3. Calculate the ratio of line lengths
        if 'line_length_ratio' not in df.columns:
            # Replace zero length with a small number to avoid division by zero errors
            len_b = df[text_col_b].str.len().replace(0, np.nan)
            df['line_length_ratio'] = df[text_col_a].str.len() / len_b
            df['line_length_ratio'].fillna(0, inplace=True) # Fill cases where line_b had 0 length

        print("Feature engineering complete. ✅")
    
    # --- Enforce table rule ---
    if 'both_in_table' in df.columns:
        print("\nEnforcing rule: If both lines are in a table, they belong to the same block.")
        original_ones = df['label'].sum()
        # Set label to 1 (same block) where both_in_table is 1
        df.loc[df['both_in_table'] == 1, 'label'] = 1
        new_ones = df['label'].sum()
        print(f"Updated labels based on the table rule. Changed {new_ones - original_ones} labels to 1.")


    # Check label distribution
    label_counts = df['label'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")

    return df

try:
    # Load data from textlines_csv_output folder
    data = load_and_prepare_data('../../data/labelled_textlines')

    # --- Define Features (X) and Target (y) ---
    # Include all original and newly engineered features
    feature_cols = [
        'normalized_vertical_gap', 'indentation_change', 'same_alignment',
        'is_centered_A', 'is_centered_B', 'font_size_a', 'font_size_b', 'font_size_diff',
        'same_font', 'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B',
        'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 'same_monospace',
        'line_a_ends_punctuation', 'line_b_starts_lowercase', 'is_linea_in_rectangle',
        'is_lineb_in_rectangle', 'both_in_table', 'neither_in_table',
        'is_linea_hashed', 'is_lineb_hashed', 'both_hashed', 'neither_hashed'
    ]
    
    # Add new features if they exist in the data
    if 'line_length_ratio' in data.columns:
        feature_cols.append('line_length_ratio')

    # Filter feature columns to only include those that exist in the data
    available_features = [col for col in feature_cols if col in data.columns]
    if len(available_features) != len(feature_cols):
        missing_features = set(feature_cols) - set(available_features)
        print(f"Warning: Missing features: {missing_features}")
    
    feature_cols = available_features
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

    # --- Hyperparameter Tuning with GridSearchCV ---
    print("\n--- Starting Hyperparameter Tuning for Recall ---")
    # Define the parameter grid to search.
    # We focus on weights that heavily penalize misclassifying the minority class '0'.
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', {0: 2, 1: 1}, {0: 3, 1: 1}, {0: 4, 1: 1}]
    }

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV to find the best parameters with a focus on 'recall' for class 0
    # Note: 'recall' defaults to recall of the positive class (1).
    # To focus on class 0, we'd ideally need a custom scorer.
    # However, 'f1_weighted' or 'roc_auc' are good proxies for imbalanced datasets.
    # For simplicity and directness, 'recall' on the positive class is often a good starting point.
    # A more robust approach is using `make_scorer`. Let's use 'f1_weighted' for a balanced metric.
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, n_jobs=-1, scoring='f1_weighted', verbose=2)

    # Train the model using the grid search
    grid_search.fit(X_train, y_train)

    print("\nHyperparameter tuning complete. ✅")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Use the best model found by the grid search for predictions
    best_model = grid_search.best_estimator_

    # --- Make Predictions and Evaluate the Best Model ---
    print("\n--- Model Performance on Test Data using Best Estimator ---")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # Only show detailed classification report if we have both classes in test set
    unique_test_labels = sorted(y_test.unique())
    if len(unique_test_labels) > 1:
        print("\nClassification Report:")
        print("Focus on the 'recall' for 'block_start (0)' to see how well the model identifies new blocks.")
        print(classification_report(y_test, y_pred, target_names=['block_start (0)', 'block_in (1)']))
        print("\nConfusion Matrix:")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                         index=['Actual Start (0)', 'Actual In (1)'],
                         columns=['Predicted Start (0)', 'Predicted In (1)']))
    else:
        print(f"⚠️  Test set only contains class {unique_test_labels[0]}, skipping detailed evaluation.")

    # --- Feature Importance ---
    print("\nFeature Importances from Best Model:")
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False))

    # --- Save the Trained Model ---
    model_filename = '../models/textline_models/text_block_merger_model.joblib'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    
    joblib.dump({
        'model': best_model,
        'feature_columns': feature_cols
    }, model_filename)
    print(f"\n✅ Best model successfully saved to '{model_filename}'")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")