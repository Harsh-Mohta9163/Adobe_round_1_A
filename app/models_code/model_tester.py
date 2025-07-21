import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# --- Configuration ---
TEST_FOLDER = '../data/test_csv/'
MODEL_FILE = 'text_block_merger_model.joblib'

try:
    print(f"Loading model from '{MODEL_FILE}'...")
    model_data = joblib.load(MODEL_FILE)
    
    # Handle both old and new model formats
    if isinstance(model_data, dict):
        model = model_data['model']
    else:
        model = model_data
    
    print("Model loaded successfully. ✅")
    
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_FILE}'.")
    exit()

# --- Define Features ---
feature_cols = [
    'normalized_vertical_gap', 'indentation_change', 'font_size_diff',
    'same_font', 'line_a_ends_punctuation', 'line_b_starts_lowercase',
    'same_alignment', 'is_centered_A', 'is_centered_B',
    'is_linea_in_rectangle', 'is_lineb_in_rectangle', 'both_in_table',
    'neither_in_table'
]

try:
    # --- Find all CSV files in test folder ---
    csv_pattern = os.path.join(TEST_FOLDER, '*.csv')
    test_files = glob.glob(csv_pattern)
    
    if not test_files:
        print(f"Error: No CSV files found in '{TEST_FOLDER}'")
        exit()
    
    print(f"\nFound {len(test_files)} CSV files to test:")
    for file in test_files:
        print(f"  - {os.path.basename(file)}")
    
    # --- Test each file individually and collect results ---
    individual_results = []
    all_data = []
    all_y_true = []
    all_y_pred = []
    
    print(f"\n{'='*80}")
    print("INDIVIDUAL PDF PERFORMANCE")
    print(f"{'='*80}")
    
    for i, test_file in enumerate(test_files, 1):
        pdf_name = os.path.basename(test_file)
        print(f"\n[{i}/{len(test_files)}] Testing: {pdf_name}")
        print("-" * 60)
        
        try:
            # Try multiple encodings
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(test_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"❌ Could not load with any encoding")
                continue
                
            # Check if all required features are present
            missing_features = [col for col in feature_cols if col not in df.columns]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                continue
            
            # Check if labels exist
            if 'label' not in df.columns:
                print(f"❌ No labels found")
                continue
            
            # Clean data
            original_rows = len(df)
            df.dropna(subset=['label'], inplace=True)
            df['label'] = df['label'].astype(int)
            
            if len(df) == 0:
                print(f"❌ No valid data after cleaning")
                continue
            
            print(f"✅ Loaded {len(df)} samples (dropped {original_rows - len(df)} invalid rows)")
            
            # Check label distribution
            label_counts = df['label'].value_counts().sort_index()
            print(f"Label distribution: ", end="")
            for label, count in label_counts.items():
                print(f"Class {label}: {count} ({count/len(df)*100:.1f}%) ", end="")
            print()
            
            # Extract features and make predictions
            X_test = df[feature_cols]
            y_test = df['label']
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle cases where not all classes are present
            try:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            except:
                precision = recall = f1 = 0.0
            
            # Calculate per-class metrics if both classes present
            unique_labels = sorted(y_test.unique())
            class_0_precision = class_0_recall = class_1_precision = class_1_recall = "N/A"
            
            if len(unique_labels) > 1:
                try:
                    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
                    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
                    if len(precision_per_class) >= 2:
                        class_0_precision = f"{precision_per_class[0]:.3f}"
                        class_0_recall = f"{recall_per_class[0]:.3f}"
                        class_1_precision = f"{precision_per_class[1]:.3f}"
                        class_1_recall = f"{recall_per_class[1]:.3f}"
                except:
                    pass
            
            print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%}) | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            
            # Store results
            individual_results.append({
                'PDF': pdf_name,
                'Samples': len(df),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Class_0_Precision': class_0_precision,
                'Class_0_Recall': class_0_recall,
                'Class_1_Precision': class_1_precision,
                'Class_1_Recall': class_1_recall
            })
            
            # Store for overall calculation
            all_data.append(df)
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not individual_results:
        print("Error: No valid test files could be processed.")
        exit()
    
    # --- Display Individual Results Table ---
    print(f"\n{'='*80}")
    print("INDIVIDUAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(individual_results)
    
    # Display formatted table
    print(f"{'PDF Name':<35} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['PDF']:<35} {row['Samples']:<8} {row['Accuracy']:.4f}    {row['Precision']:.3f}     {row['Recall']:.3f}   {row['F1_Score']:.3f}")
    
    # --- Summary Statistics ---
    print(f"\n{'='*50}")
    print("INDIVIDUAL RESULTS STATISTICS")
    print(f"{'='*50}")
    print(f"Total PDFs tested: {len(results_df)}")
    print(f"Average accuracy: {results_df['Accuracy'].mean():.4f} ({results_df['Accuracy'].mean():.2%})")
    print(f"Best accuracy: {results_df['Accuracy'].max():.4f} ({results_df.loc[results_df['Accuracy'].idxmax(), 'PDF']})")
    print(f"Worst accuracy: {results_df['Accuracy'].min():.4f} ({results_df.loc[results_df['Accuracy'].idxmin(), 'PDF']})")
    print(f"Standard deviation: {results_df['Accuracy'].std():.4f}")
    print(f"Median accuracy: {results_df['Accuracy'].median():.4f}")
    
    # --- Overall Combined Performance ---
    print(f"\n{'='*50}")
    print("OVERALL COMBINED PERFORMANCE")
    print(f"{'='*50}")
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    
    print(f"Total samples tested: {len(all_y_true)}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.2%})")
    print(f"Overall Precision (weighted): {overall_precision:.4f}")
    print(f"Overall Recall (weighted): {overall_recall:.4f}")
    print(f"Overall F1-Score (weighted): {overall_f1:.4f}")
    
    # Overall label distribution
    overall_label_counts = pd.Series(all_y_true).value_counts().sort_index()
    print(f"\nOverall label distribution:")
    for label, count in overall_label_counts.items():
        print(f"  Label {label}: {count} samples ({count/len(all_y_true)*100:.1f}%)")
    
    # Overall prediction distribution
    overall_pred_counts = pd.Series(all_y_pred).value_counts().sort_index()
    print(f"\nOverall prediction distribution:")
    for label, count in overall_pred_counts.items():
        print(f"  Predicted {label}: {count} samples ({count/len(all_y_pred)*100:.1f}%)")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=['block_start (0)', 'block_in (1)']))
    
    print(f"\nConfusion Matrix:")
    cm_df = pd.DataFrame(confusion_matrix(all_y_true, all_y_pred),
                        index=['Actual Start', 'Actual In'],
                        columns=['Predicted Start', 'Predicted In'])
    print(cm_df)
    
    # Additional useful metrics
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"  Specificity (True Negative Rate): {specificity:.4f}")
    print(f"  False Positive Rate: {fp / (fp + tn):.4f}" if (fp + tn) > 0 else "  False Positive Rate: N/A")
    print(f"  False Negative Rate: {fn / (fn + tp):.4f}" if (fn + tp) > 0 else "  False Negative Rate: N/A")

except Exception as e:
    print(f"An error occurred: {e}")