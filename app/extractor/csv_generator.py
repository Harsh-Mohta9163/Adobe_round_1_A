# import json
# import os
# import csv
# from collections import defaultdict
# import numpy as np

# def get_page_statistics(textlines_on_page: list) -> dict:
#     """Calculate page-level statistics like median gap"""
#     stats = {}
#     gaps = []
    
#     for i in range(len(textlines_on_page) - 1):
#         current_line = textlines_on_page[i]
#         next_line = textlines_on_page[i + 1]
        
#         # Calculate vertical gap using bbox coordinates
#         if 'features' in current_line and 'features' in next_line:
#             current_bbox = current_line['features']['bbox']
#             next_bbox = next_line['features']['bbox']
            
#             if current_bbox and next_bbox and len(current_bbox) >= 4 and len(next_bbox) >= 4:
#                 gap = next_bbox[1] - current_bbox[3]  # next_top - current_bottom
#                 if gap > 0:
#                     gaps.append(gap)
    
#     stats['median_gap'] = np.median(gaps) if gaps else 12.0
#     return stats

# def calculate_features_for_merging(line_a: dict, line_b: dict, page_stats: dict) -> dict:
#     """Calculate features for merging two consecutive lines"""
#     features = {}
#     median_gap = page_stats.get('median_gap', 12.0)
    
#     # Get bbox and features safely
#     bbox_a = line_a.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_a.get('features') else [0, 0, 0, 0]
#     bbox_b = line_b.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_b.get('features') else [0, 0, 0, 0]
    
#     features_a = line_a.get('features', {})
#     features_b = line_b.get('features', {})
    
#     # Normalized vertical gap
#     if len(bbox_a) >= 4 and len(bbox_b) >= 4:
#         vertical_gap = bbox_b[1] - bbox_a[3]  # next_top - current_bottom
#         features['normalized_vertical_gap'] = round(vertical_gap / median_gap, 2) if median_gap > 0 else 0
#     else:
#         features['normalized_vertical_gap'] = 0
    
#     # Indentation change
#     if len(bbox_a) >= 2 and len(bbox_b) >= 2:
#         features['indentation_change'] = round(bbox_b[0] - bbox_a[0], 2)
#     else:
#         features['indentation_change'] = 0
    
#     # Font sizes
#     font_size_a = features_a.get('font_size', 12.0)
#     font_size_b = features_b.get('font_size', 12.0)
#     features['font_size_a'] = round(font_size_a, 2)
#     features['font_size_b'] = round(font_size_b, 2)
#     features['font_size_diff'] = round(font_size_b - font_size_a, 2)
    
#     # Same font
#     font_name_a = features_a.get('font_name', '')
#     font_name_b = features_b.get('font_name', '')
#     features['same_font'] = 1 if font_name_a == font_name_b else 0
    
#     # Font formatting features
#     features['is_bold_A'] = 1 if features_a.get('is_bold', False) else 0
#     features['is_bold_B'] = 1 if features_b.get('is_bold', False) else 0
#     features['is_italic_A'] = 1 if features_a.get('is_italic', False) else 0
#     features['is_italic_B'] = 1 if features_b.get('is_italic', False) else 0
#     features['is_monospace_A'] = 1 if features_a.get('is_monospace', False) else 0
#     features['is_monospace_B'] = 1 if features_b.get('is_monospace', False) else 0
    
#     # Same formatting features
#     features['same_bold'] = 1 if features_a.get('is_bold', False) == features_b.get('is_bold', False) else 0
#     features['same_italic'] = 1 if features_a.get('is_italic', False) == features_b.get('is_italic', False) else 0
#     features['same_monospace'] = 1 if features_a.get('is_monospace', False) == features_b.get('is_monospace', False) else 0
    
#     # Line ending with punctuation
#     text_a = line_a.get('md_text_cleaned', '').strip()
#     features['line_a_ends_punctuation'] = 1 if text_a and text_a[-1] in '.!?:' else 0
    
#     # Line B starts with lowercase
#     text_b = line_b.get('md_text_cleaned', '').strip()
#     features['line_b_starts_lowercase'] = 1 if text_b and text_b[0].islower() else 0
    
#     # Same alignment (using bbox left coordinate)
#     if len(bbox_a) >= 1 and len(bbox_b) >= 1:
#         features['same_alignment'] = 1 if abs(bbox_a[0] - bbox_b[0]) < 5 else 0
#     else:
#         features['same_alignment'] = 0
    
#     # Is centered (rough estimation based on bbox position)
#     page_width = 595  # Approximate A4 width in points
#     if len(bbox_a) >= 3:
#         center_a = (bbox_a[0] + bbox_a[2]) / 2
#         features['is_centered_A'] = 1 if abs(center_a - page_width/2) < 50 else 0
#     else:
#         features['is_centered_A'] = 0
        
#     if len(bbox_b) >= 3:
#         center_b = (bbox_b[0] + bbox_b[2]) / 2
#         features['is_centered_B'] = 1 if abs(center_b - page_width/2) < 50 else 0
#     else:
#         features['is_centered_B'] = 0
    
#     # Table-related features
#     is_linea_in_table = line_a.get('is_in_table', False)
#     is_lineb_in_table = line_b.get('is_in_table', False)
    
#     features['is_linea_in_rectangle'] = 1 if is_linea_in_table else 0
#     features['is_lineb_in_rectangle'] = 1 if is_lineb_in_table else 0
#     features['both_in_table'] = 1 if (is_linea_in_table and is_lineb_in_table) else 0
#     features['neither_in_table'] = 1 if (not is_linea_in_table and not is_lineb_in_table) else 0
    
#     return features

# def generate_csv_from_aggregated(pdf_name: str):
#     """
#     Generate CSV with textline features from aggregated JSON file
#     Args:
#         pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
#     """
#     # Define file paths
#     input_json_path = f"../../data/aggregator_output/aggregated_{pdf_name}.json"
#     output_csv_path = f"../../data/textlines_csv_output/textlines_ground_truth_{pdf_name}.csv"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
#     if not os.path.exists(input_json_path):
#         print(f"Error: Input file not found: '{input_json_path}'")
#         return
    
#     # Load aggregated data
#     with open(input_json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # Get only matched lines (span_match = True)
#     matched_lines = [
#         line for line in data['aggregated_data'] 
#         if line.get('span_match', False)
#     ]
    
#     if len(matched_lines) < 2:
#         print(f"Warning: Not enough matched lines in {pdf_name} to generate pairs")
#         return
    
#     # Group lines by page
#     lines_by_page = defaultdict(list)
#     for line in matched_lines:
#         page_num = line.get('page_number', 1)
#         lines_by_page[page_num].append(line)
    
#     # Calculate page statistics
#     page_statistics = {}
#     for page_num, lines in lines_by_page.items():
#         page_statistics[page_num] = get_page_statistics(lines)
    
#     # Generate CSV
#     with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
#         fieldnames = [
#             'text_a', 'text_b', 'normalized_vertical_gap', 'indentation_change', 
#             'font_size_a', 'font_size_b', 'font_size_diff', 'same_font', 'line_a_ends_punctuation', 
#             'line_b_starts_lowercase', 'same_alignment', 'is_centered_A', 
#             'is_centered_B', 'is_linea_in_rectangle', 'is_lineb_in_rectangle', 
#             'both_in_table', 'neither_in_table', 
#             'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B', 
#             'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 
#             'same_monospace', 'label'
#         ]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
        
#         row_count = 0
#         for i in range(len(matched_lines) - 1):
#             line_a = matched_lines[i]
#             line_b = matched_lines[i + 1]
            
#             # Skip if lines are on different pages
#             if line_a.get('page_number') != line_b.get('page_number'):
#                 continue
            
#             page_num = line_a.get('page_number', 1)
#             page_stats = page_statistics.get(page_num, {'median_gap': 12.0})
            
#             # Calculate features
#             features = calculate_features_for_merging(line_a, line_b, page_stats)
            
#             # Add text fields - remove the extra quote you added
#             features['text_a'] = line_a.get('md_text_cleaned', '').strip()
#             features['text_b'] = line_b.get('md_text_cleaned', '').strip()
#             features['label'] = ''  # Empty label for manual annotation
            
#             # Skip if either text is empty
#             if not features['text_a'] or not features['text_b']:
#                 continue
            
#             # Write row to CSV
#             writer.writerow(features)
#             row_count += 1
    
#     print(f"Successfully created '{output_csv_path}' with {row_count} feature rows for {pdf_name}")









































# import json
# import os
# import csv
# from collections import defaultdict
# import numpy as np

# def get_page_statistics(textlines_on_page: list) -> dict:
#     """Calculate page-level statistics like median gap"""
#     stats = {}
#     gaps = []
    
#     for i in range(len(textlines_on_page) - 1):
#         current_line = textlines_on_page[i]
#         next_line = textlines_on_page[i + 1]
        
#         # Calculate vertical gap using bbox coordinates
#         if 'features' in current_line and 'features' in next_line:
#             current_bbox = current_line['features']['bbox']
#             next_bbox = next_line['features']['bbox']
            
#             if current_bbox and next_bbox and len(current_bbox) >= 4 and len(next_bbox) >= 4:
#                 gap = next_bbox[1] - current_bbox[3]  # next_top - current_bottom
#                 if gap > 0:
#                     gaps.append(gap)
    
#     stats['median_gap'] = np.median(gaps) if gaps else 12.0
#     return stats

# def calculate_features_for_merging(line_a: dict, line_b: dict, page_stats: dict) -> dict:
#     """Calculate features for merging two consecutive lines"""
#     features = {}
#     median_gap = page_stats.get('median_gap', 12.0)
    
#     # Get bbox and features safely
#     bbox_a = line_a.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_a.get('features') else [0, 0, 0, 0]
#     bbox_b = line_b.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_b.get('features') else [0, 0, 0, 0]
    
#     features_a = line_a.get('features', {})
#     features_b = line_b.get('features', {})
    
#     # Normalized vertical gap
#     if len(bbox_a) >= 4 and len(bbox_b) >= 4:
#         vertical_gap = bbox_b[1] - bbox_a[3]  # next_top - current_bottom
#         features['normalized_vertical_gap'] = round(vertical_gap / median_gap, 2) if median_gap > 0 else 0
#     else:
#         features['normalized_vertical_gap'] = 0
    
#     # Indentation change
#     if len(bbox_a) >= 2 and len(bbox_b) >= 2:
#         features['indentation_change'] = round(bbox_b[0] - bbox_a[0], 2)
#     else:
#         features['indentation_change'] = 0
    
#     # Font sizes
#     font_size_a = features_a.get('font_size', 12.0)
#     font_size_b = features_b.get('font_size', 12.0)
#     features['font_size_a'] = round(font_size_a, 2)
#     features['font_size_b'] = round(font_size_b, 2)
#     features['font_size_diff'] = round(font_size_b - font_size_a, 2)
    
#     # Same font
#     font_name_a = features_a.get('font_name', '')
#     font_name_b = features_b.get('font_name', '')
#     features['same_font'] = 1 if font_name_a == font_name_b else 0
    
#     # Font formatting features
#     features['is_bold_A'] = 1 if features_a.get('is_bold', False) else 0
#     features['is_bold_B'] = 1 if features_b.get('is_bold', False) else 0
#     features['is_italic_A'] = 1 if features_a.get('is_italic', False) else 0
#     features['is_italic_B'] = 1 if features_b.get('is_italic', False) else 0
#     features['is_monospace_A'] = 1 if features_a.get('is_monospace', False) else 0
#     features['is_monospace_B'] = 1 if features_b.get('is_monospace', False) else 0
    
#     # Same formatting features
#     features['same_bold'] = 1 if features_a.get('is_bold', False) == features_b.get('is_bold', False) else 0
#     features['same_italic'] = 1 if features_a.get('is_italic', False) == features_b.get('is_italic', False) else 0
#     features['same_monospace'] = 1 if features_a.get('is_monospace', False) == features_b.get('is_monospace', False) else 0
    
#     # Line ending with punctuation
#     text_a = line_a.get('md_text_cleaned', '').strip()
#     features['line_a_ends_punctuation'] = 1 if text_a and text_a[-1] in '.!?:' else 0
    
#     # Line B starts with lowercase
#     text_b = line_b.get('md_text_cleaned', '').strip()
#     features['line_b_starts_lowercase'] = 1 if text_b and text_b[0].islower() else 0
    
#     # Same alignment (using bbox left coordinate)
#     if len(bbox_a) >= 1 and len(bbox_b) >= 1:
#         features['same_alignment'] = 1 if abs(bbox_a[0] - bbox_b[0]) < 5 else 0
#     else:
#         features['same_alignment'] = 0
    
#     # Is centered (rough estimation based on bbox position)
#     page_width = 595  # Approximate A4 width in points
#     if len(bbox_a) >= 3:
#         center_a = (bbox_a[0] + bbox_a[2]) / 2
#         features['is_centered_A'] = 1 if abs(center_a - page_width/2) < 50 else 0
#     else:
#         features['is_centered_A'] = 0
        
#     if len(bbox_b) >= 3:
#         center_b = (bbox_b[0] + bbox_b[2]) / 2
#         features['is_centered_B'] = 1 if abs(center_b - page_width/2) < 50 else 0
#     else:
#         features['is_centered_B'] = 0
    
#     # Table-related features
#     is_linea_in_table = line_a.get('is_in_table', False)
#     is_lineb_in_table = line_b.get('is_in_table', False)
    
#     features['is_linea_in_rectangle'] = 1 if is_linea_in_table else 0
#     features['is_lineb_in_rectangle'] = 1 if is_lineb_in_table else 0
#     features['both_in_table'] = 1 if (is_linea_in_table and is_lineb_in_table) else 0
#     features['neither_in_table'] = 1 if (not is_linea_in_table and not is_lineb_in_table) else 0
    
#     return features

# def generate_csv_from_aggregated(pdf_name: str):
#     """
#     Generate CSV with textline features from aggregated JSON file
#     Args:
#         pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
#     """
#     # Define file paths
#     input_json_path = f"../../data/aggregator_output/aggregated_{pdf_name}.json"
#     output_csv_path = f"../../data/textlines_csv_output/textlines_ground_truth_{pdf_name}.csv"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
#     if not os.path.exists(input_json_path):
#         print(f"Error: Input file not found: '{input_json_path}'")
#         return
    
#     # Load aggregated data
#     with open(input_json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # Get only matched lines (span_match = True)
#     matched_lines = [
#         line for line in data['aggregated_data'] 
#         if line.get('span_match', False)
#     ]
    
#     if len(matched_lines) < 2:
#         print(f"Warning: Not enough matched lines in {pdf_name} to generate pairs")
#         return
    
#     # Group lines by page
#     lines_by_page = defaultdict(list)
#     for line in matched_lines:
#         page_num = line.get('page_number', 1)
#         lines_by_page[page_num].append(line)
    
#     # Calculate page statistics
#     page_statistics = {}
#     for page_num, lines in lines_by_page.items():
#         page_statistics[page_num] = get_page_statistics(lines)
    
#     # Generate CSV
#     with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
#         fieldnames = [
#             'text_a', 'text_b', 'normalized_vertical_gap', 'indentation_change', 
#             'font_size_a', 'font_size_b', 'font_size_diff', 'same_font', 'line_a_ends_punctuation', 
#             'line_b_starts_lowercase', 'same_alignment', 'is_centered_A', 
#             'is_centered_B', 'is_linea_in_rectangle', 'is_lineb_in_rectangle', 
#             'both_in_table', 'neither_in_table', 
#             'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B', 
#             'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 
#             'same_monospace', 'label'
#         ]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
        
#         row_count = 0
#         for i in range(len(matched_lines) - 1):
#             line_a = matched_lines[i]
#             line_b = matched_lines[i + 1]
            
#             # Skip if lines are on different pages
#             if line_a.get('page_number') != line_b.get('page_number'):
#                 continue
            
#             page_num = line_a.get('page_number', 1)
#             page_stats = page_statistics.get(page_num, {'median_gap': 12.0})
            
#             # Calculate features
#             features = calculate_features_for_merging(line_a, line_b, page_stats)
            
#             # Add text fields - remove the extra quote you added
#             features['text_a'] = line_a.get('md_text_cleaned', '').strip()
#             features['text_b'] = line_b.get('md_text_cleaned', '').strip()
#             features['label'] = ''  # Empty label for manual annotation
            
#             # Skip if either text is empty
#             if not features['text_a'] or not features['text_b']:
#                 continue
            
#             # Write row to CSV
#             writer.writerow(features)
#             row_count += 1
    
#     print(f"Successfully created '{output_csv_path}' with {row_count} feature rows for {pdf_name}")









































# import json
# import os
# import csv
# from collections import defaultdict
# import numpy as np
# import re

# def clean_span_text(span_text: str) -> str:
#     """Clean span text by removing bullet points and formatting characters"""
#     if not span_text:
#         return ""
    
#     # Remove leading/trailing whitespace
#     cleaned = span_text.strip()
    
#     # Remove bullet points and list markers at the beginning
#     cleaned = re.sub(r'^[-+=@•·▪▫◦‣⁃]+\s*', '', cleaned)

#     return cleaned.strip()

# def get_page_statistics(textlines_on_page: list) -> dict:
#     """Calculate page-level statistics like median gap"""
#     stats = {}
#     gaps = []
    
#     for i in range(len(textlines_on_page) - 1):
#         current_line = textlines_on_page[i]
#         next_line = textlines_on_page[i + 1]
        
#         # Calculate vertical gap using bbox coordinates
#         if 'features' in current_line and 'features' in next_line:
#             current_bbox = current_line['features']['bbox']
#             next_bbox = next_line['features']['bbox']
            
#             if current_bbox and next_bbox and len(current_bbox) >= 4 and len(next_bbox) >= 4:
#                 gap = next_bbox[1] - current_bbox[3]  # next_top - current_bottom
#                 if gap > 0:
#                     gaps.append(gap)
    
#     stats['median_gap'] = np.median(gaps) if gaps else 12.0
#     return stats

# def calculate_features_for_merging(line_a: dict, line_b: dict, page_stats: dict) -> dict:
#     """Calculate features for merging two consecutive lines"""
#     features = {}
#     median_gap = page_stats.get('median_gap', 12.0)
    
#     # Get bbox and features safely
#     bbox_a = line_a.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_a.get('features') else [0, 0, 0, 0]
#     bbox_b = line_b.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_b.get('features') else [0, 0, 0, 0]
    
#     features_a = line_a.get('features', {})
#     features_b = line_b.get('features', {})
    
#     # Normalized vertical gap
#     if len(bbox_a) >= 4 and len(bbox_b) >= 4:
#         vertical_gap = bbox_b[1] - bbox_a[3]  # next_top - current_bottom
#         features['normalized_vertical_gap'] = round(vertical_gap / median_gap, 2) if median_gap > 0 else 0
#     else:
#         features['normalized_vertical_gap'] = 0
    
#     # Indentation change
#     if len(bbox_a) >= 2 and len(bbox_b) >= 2:
#         features['indentation_change'] = round(bbox_b[0] - bbox_a[0], 2)
#     else:
#         features['indentation_change'] = 0
    
#     # Font sizes
#     font_size_a = features_a.get('font_size', 12.0)
#     font_size_b = features_b.get('font_size', 12.0)
#     features['font_size_a'] = round(font_size_a, 2)
#     features['font_size_b'] = round(font_size_b, 2)
#     features['font_size_diff'] = round(font_size_b - font_size_a, 2)
    
#     # Same font
#     font_name_a = features_a.get('font_name', '')
#     font_name_b = features_b.get('font_name', '')
#     features['same_font'] = 1 if font_name_a == font_name_b else 0
    
#     # Font formatting features
#     features['is_bold_A'] = 1 if features_a.get('is_bold', False) else 0
#     features['is_bold_B'] = 1 if features_b.get('is_bold', False) else 0
#     features['is_italic_A'] = 1 if features_a.get('is_italic', False) else 0
#     features['is_italic_B'] = 1 if features_b.get('is_italic', False) else 0
#     features['is_monospace_A'] = 1 if features_a.get('is_monospace', False) else 0
#     features['is_monospace_B'] = 1 if features_b.get('is_monospace', False) else 0
    
#     # Same formatting features
#     features['same_bold'] = 1 if features_a.get('is_bold', False) == features_b.get('is_bold', False) else 0
#     features['same_italic'] = 1 if features_a.get('is_italic', False) == features_b.get('is_italic', False) else 0
#     features['same_monospace'] = 1 if features_a.get('is_monospace', False) == features_b.get('is_monospace', False) else 0
    
#     # Line ending with punctuation
#     text_a = line_a.get('md_text_cleaned', '').strip()
#     features['line_a_ends_punctuation'] = 1 if text_a and text_a[-1] in '.!?:' else 0
    
#     # Line B starts with lowercase
#     text_b = line_b.get('md_text_cleaned', '').strip()
#     features['line_b_starts_lowercase'] = 1 if text_b and text_b[0].islower() else 0
    
#     # Same alignment (using bbox left coordinate)
#     if len(bbox_a) >= 1 and len(bbox_b) >= 1:
#         features['same_alignment'] = 1 if abs(bbox_a[0] - bbox_b[0]) < 5 else 0
#     else:
#         features['same_alignment'] = 0
    
#     # Is centered (rough estimation based on bbox position)
#     page_width = 595  # Approximate A4 width in points
#     if len(bbox_a) >= 3:
#         center_a = (bbox_a[0] + bbox_a[2]) / 2
#         features['is_centered_A'] = 1 if abs(center_a - page_width/2) < 50 else 0
#     else:
#         features['is_centered_A'] = 0
        
#     if len(bbox_b) >= 3:
#         center_b = (bbox_b[0] + bbox_b[2]) / 2
#         features['is_centered_B'] = 1 if abs(center_b - page_width/2) < 50 else 0
#     else:
#         features['is_centered_B'] = 0
    
#     # Table-related features
#     is_linea_in_table = line_a.get('is_in_table', False)
#     is_lineb_in_table = line_b.get('is_in_table', False)
    
#     features['is_linea_in_rectangle'] = 1 if is_linea_in_table else 0
#     features['is_lineb_in_rectangle'] = 1 if is_lineb_in_table else 0
#     features['both_in_table'] = 1 if (is_linea_in_table and is_lineb_in_table) else 0
#     features['neither_in_table'] = 1 if (not is_linea_in_table and not is_lineb_in_table) else 0
    
#     # Header/hash-related features (NEW)
#     is_linea_hashed = line_a.get('is_hashed', False)
#     is_lineb_hashed = line_b.get('is_hashed', False)
    
#     features['is_linea_hashed'] = 1 if is_linea_hashed else 0
#     features['is_lineb_hashed'] = 1 if is_lineb_hashed else 0
#     features['both_hashed'] = 1 if (is_linea_hashed and is_lineb_hashed) else 0
#     features['neither_hashed'] = 1 if (not is_linea_hashed and not is_lineb_hashed) else 0
    
#     return features

# def generate_csv_from_aggregated(pdf_name: str):
#     """
#     Generate CSV with textline features from aggregated JSON file
#     Args:
#         pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
#     """
#     # Define file paths
#     input_json_path = f"../../data/aggregator_output/aggregated_{pdf_name}.json"
#     output_csv_path = f"../../data/textlines_csv_output/textlines_ground_truth_{pdf_name}.csv"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
#     if not os.path.exists(input_json_path):
#         print(f"Error: Input file not found: '{input_json_path}'")
#         return
    
#     # Load aggregated data
#     with open(input_json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # Get only matched lines (span_match = True)
#     matched_lines = [
#         line for line in data['aggregated_data'] 
#         if line.get('span_match', False)
#     ]
    
#     if len(matched_lines) < 2:
#         print(f"Warning: Not enough matched lines in {pdf_name} to generate pairs")
#         return
    
#     # Group lines by page
#     lines_by_page = defaultdict(list)
#     for line in matched_lines:
#         page_num = line.get('page_number', 1)
#         lines_by_page[page_num].append(line)
    
#     # Calculate page statistics
#     page_statistics = {}
#     for page_num, lines in lines_by_page.items():
#         page_statistics[page_num] = get_page_statistics(lines)
    
#     # Generate CSV
#     with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
#         fieldnames = [
#             # Text fields - markdown and corresponding span text
#             'text_a', 'span_text_a', 'text_b', 'span_text_b',
            
#             # Layout features
#             'normalized_vertical_gap', 'indentation_change', 'same_alignment', 
#             'is_centered_A', 'is_centered_B',
            
#             # Font features
#             'font_size_a', 'font_size_b', 'font_size_diff', 'same_font',
#             'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B', 
#             'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 
#             'same_monospace',
            
#             # Text pattern features
#             'line_a_ends_punctuation', 'line_b_starts_lowercase',
            
#             # Table features
#             'is_linea_in_rectangle', 'is_lineb_in_rectangle', 
#             'both_in_table', 'neither_in_table',
            
#             # Header/hash features (NEW)
#             'is_linea_hashed', 'is_lineb_hashed', 
#             'both_hashed', 'neither_hashed',
            
#             # Label for manual annotation
#             'label'
#         ]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
        
#         row_count = 0
#         for i in range(len(matched_lines) - 1):
#             line_a = matched_lines[i]
#             line_b = matched_lines[i + 1]
            
#             # Skip if lines are on different pages
#             if line_a.get('page_number') != line_b.get('page_number'):
#                 continue
            
#             page_num = line_a.get('page_number', 1)
#             page_stats = page_statistics.get(page_num, {'median_gap': 12.0})
            
#             # Calculate features
#             features = calculate_features_for_merging(line_a, line_b, page_stats)
            
#             # Add text fields - both markdown and span text
#             features['text_a'] = line_a.get('md_text_cleaned', '').strip()
#             features['text_b'] = line_b.get('md_text_cleaned', '').strip()
            
#             # Add span text fields - cleaned to remove bullet points and formatting
#             features['span_text_a'] = clean_span_text(line_a.get('span_text', ''))
#             features['span_text_b'] = clean_span_text(line_b.get('span_text', ''))
            
#             features['label'] = ''  # Empty label for manual annotation
            
#             # Skip if either text is empty
#             if not features['text_a'] or not features['text_b']:
#                 continue
            
#             # Write row to CSV
#             writer.writerow(features)
#             row_count += 1
    
#     print(f"Successfully created '{output_csv_path}' with {row_count} feature rows for {pdf_name}")
    
#     # Print statistics about hash features
#     hash_stats = {
#         'total_pairs': row_count,
#         'hashed_a': sum(1 for line in matched_lines if line.get('is_hashed', False)),
#         'both_hashed_pairs': 0,
#         'neither_hashed_pairs': 0
#     }
    
#     # Count hash pair statistics from the generated CSV
#     with open(output_csv_path, 'r', encoding='utf-8-sig') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             if row.get('both_hashed') == '1':
#                 hash_stats['both_hashed_pairs'] += 1
#             elif row.get('neither_hashed') == '1':
#                 hash_stats['neither_hashed_pairs'] += 1
    
#     print(f"Hash feature statistics:")
#     print(f"  Total hashed lines: {hash_stats['hashed_a']}")
#     print(f"  Pairs where both are hashed: {hash_stats['both_hashed_pairs']}")
#     print(f"  Pairs where neither is hashed: {hash_stats['neither_hashed_pairs']}")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python csv_generator.py input.pdf")
#         sys.exit(1)
    
#     pdf_name = sys.argv[1]
#     generate_csv_from_aggregated(pdf_name)

























import json
import os
import csv
from collections import defaultdict
import numpy as np
import re

def clean_span_text(span_text: str) -> str:
    """Clean span text by removing bullet points and formatting characters"""
    if not span_text:
        return ""
    
    # Remove leading/trailing whitespace
    cleaned = span_text.strip()
    
    # Remove bullet points and list markers at the beginning
    cleaned = re.sub(r'^[-+=@•·▪▫◦‣⁃]+\s*', '', cleaned)

    return cleaned.strip()

def get_page_statistics(textlines_on_page: list) -> dict:
    """Calculate page-level statistics like median gap"""
    stats = {}
    gaps = []
    
    for i in range(len(textlines_on_page) - 1):
        current_line = textlines_on_page[i]
        next_line = textlines_on_page[i + 1]
        
        # Calculate vertical gap using bbox coordinates
        if 'features' in current_line and 'features' in next_line:
            current_bbox = current_line['features']['bbox']
            next_bbox = next_line['features']['bbox']
            
            if current_bbox and next_bbox and len(current_bbox) >= 4 and len(next_bbox) >= 4:
                gap = next_bbox[1] - current_bbox[3]  # next_top - current_bottom
                if gap > 0:
                    gaps.append(gap)
    
    stats['median_gap'] = np.median(gaps) if gaps else 12.0
    return stats

def calculate_features_for_merging(line_a: dict, line_b: dict, page_stats: dict) -> dict:
    """Calculate features for merging two consecutive lines"""
    features = {}
    median_gap = page_stats.get('median_gap', 12.0)
    
    # Get bbox and features safely
    bbox_a = line_a.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_a.get('features') else [0, 0, 0, 0]
    bbox_b = line_b.get('features', {}).get('bbox', [0, 0, 0, 0]) if line_b.get('features') else [0, 0, 0, 0]
    
    features_a = line_a.get('features', {})
    features_b = line_b.get('features', {})
    
    # Normalized vertical gap
    if len(bbox_a) >= 4 and len(bbox_b) >= 4:
        vertical_gap = bbox_b[1] - bbox_a[3]  # next_top - current_bottom
        features['normalized_vertical_gap'] = round(vertical_gap / median_gap, 2) if median_gap > 0 else 0
    else:
        features['normalized_vertical_gap'] = 0
    
    # Indentation change
    if len(bbox_a) >= 2 and len(bbox_b) >= 2:
        features['indentation_change'] = round(bbox_b[0] - bbox_a[0], 2)
    else:
        features['indentation_change'] = 0

    page_num_a = line_a.get('page_number', 1)
    page_num_b = line_b.get('page_number', 1)
    features['page_number_a'] = page_num_a
    features['page_number_b'] = page_num_b
    # Font sizes
    font_size_a = features_a.get('font_size', 12.0)
    font_size_b = features_b.get('font_size', 12.0)
    features['font_size_a'] = round(font_size_a, 2)
    features['font_size_b'] = round(font_size_b, 2)
    features['font_size_diff'] = round(font_size_b - font_size_a, 2)
    
    # Same font
    font_name_a = features_a.get('font_name', '')
    font_name_b = features_b.get('font_name', '')
    features['same_font'] = 1 if font_name_a == font_name_b else 0
    
    # Font formatting features
    features['is_bold_A'] = 1 if features_a.get('is_bold', False) else 0
    features['is_bold_B'] = 1 if features_b.get('is_bold', False) else 0
    features['is_italic_A'] = 1 if features_a.get('is_italic', False) else 0
    features['is_italic_B'] = 1 if features_b.get('is_italic', False) else 0
    features['is_monospace_A'] = 1 if features_a.get('is_monospace', False) else 0
    features['is_monospace_B'] = 1 if features_b.get('is_monospace', False) else 0
    
    # Same formatting features
    features['same_bold'] = 1 if features_a.get('is_bold', False) == features_b.get('is_bold', False) else 0
    features['same_italic'] = 1 if features_a.get('is_italic', False) == features_b.get('is_italic', False) else 0
    features['same_monospace'] = 1 if features_a.get('is_monospace', False) == features_b.get('is_monospace', False) else 0
    
    # Line ending with punctuation
    text_a = line_a.get('md_text_cleaned', '').strip()
    features['line_a_ends_punctuation'] = 1 if text_a and text_a[-1] in '.!?:' else 0
    
    # Line B starts with lowercase
    text_b = line_b.get('md_text_cleaned', '').strip()
    features['line_b_starts_lowercase'] = 1 if text_b and text_b[0].islower() else 0
    
    # Same alignment (using bbox left coordinate)
    if len(bbox_a) >= 1 and len(bbox_b) >= 1:
        features['same_alignment'] = 1 if abs(bbox_a[0] - bbox_b[0]) < 5 else 0
    else:
        features['same_alignment'] = 0
    
    # Is centered (rough estimation based on bbox position)
    page_width = 595  # Approximate A4 width in points
    if len(bbox_a) >= 3:
        center_a = (bbox_a[0] + bbox_a[2]) / 2
        features['is_centered_A'] = 1 if abs(center_a - page_width/2) < 50 else 0
    else:
        features['is_centered_A'] = 0
        
    if len(bbox_b) >= 3:
        center_b = (bbox_b[0] + bbox_b[2]) / 2
        features['is_centered_B'] = 1 if abs(center_b - page_width/2) < 50 else 0
    else:
        features['is_centered_B'] = 0
    
    # Table-related features
    is_linea_in_table = line_a.get('is_in_table', False)
    is_lineb_in_table = line_b.get('is_in_table', False)
    
    features['is_linea_in_rectangle'] = 1 if is_linea_in_table else 0
    features['is_lineb_in_rectangle'] = 1 if is_lineb_in_table else 0
    features['both_in_table'] = 1 if (is_linea_in_table and is_lineb_in_table) else 0
    features['neither_in_table'] = 1 if (not is_linea_in_table and not is_lineb_in_table) else 0
    
    # Header/hash-related features (NEW)
    is_linea_hashed = line_a.get('is_hashed', False)
    is_lineb_hashed = line_b.get('is_hashed', False)
    
    features['is_linea_hashed'] = 1 if is_linea_hashed else 0
    features['is_lineb_hashed'] = 1 if is_lineb_hashed else 0
    features['both_hashed'] = 1 if (is_linea_hashed and is_lineb_hashed) else 0
    features['neither_hashed'] = 1 if (not is_linea_hashed and not is_lineb_hashed) else 0
    
    return features

def generate_csv_from_aggregated(pdf_name: str):
    """
    Generate CSV with textline features from aggregated JSON file
    Args:
        pdf_name (str): Name of the PDF file (e.g., 'file01.pdf')
    """
    # Define file paths
    input_json_path = f"../../data/aggregator_output/aggregated_{pdf_name}.json"
    output_csv_path = f"../../data/textlines_csv_output/textlines_ground_truth_{pdf_name}.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    if not os.path.exists(input_json_path):
        print(f"Error: Input file not found: '{input_json_path}'")
        return
    
    # Load aggregated data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get only matched lines (span_match = True)
    matched_lines = [
        line for line in data['aggregated_data'] 
        if line.get('span_match', False)
    ]
    
    if len(matched_lines) < 2:
        print(f"Warning: Not enough matched lines in {pdf_name} to generate pairs")
        return
    
    # Group lines by page
    lines_by_page = defaultdict(list)
    for line in matched_lines:
        page_num = line.get('page_number', 1)
        lines_by_page[page_num].append(line)
    
    # Calculate page statistics
    page_statistics = {}
    for page_num, lines in lines_by_page.items():
        page_statistics[page_num] = get_page_statistics(lines)
    
    # Generate CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            # Text fields - markdown and corresponding span text
            'text_a', 'span_text_a', 'text_b', 'span_text_b',
            
            # Layout features
            'normalized_vertical_gap', 'indentation_change', 'same_alignment', 
            'is_centered_A', 'is_centered_B',
            
            # Font features
            'font_size_a', 'font_size_b', 'font_size_diff', 'same_font',
            'is_bold_A', 'is_bold_B', 'is_italic_A', 'is_italic_B', 
            'is_monospace_A', 'is_monospace_B', 'same_bold', 'same_italic', 
            'same_monospace',
            
            # Text pattern features
            'line_a_ends_punctuation', 'line_b_starts_lowercase',
            
            # Table features
            'is_linea_in_rectangle', 'is_lineb_in_rectangle', 
            'both_in_table', 'neither_in_table',
            
            # Header/hash features (NEW)
            'is_linea_hashed', 'is_lineb_hashed', 
            'both_hashed', 'neither_hashed',
            'page_number_a','page_number_b',
            # Label for manual annotation
            'label'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        row_count = 0
        for i in range(len(matched_lines) - 1):
            line_a = matched_lines[i]
            line_b = matched_lines[i + 1]
            
            # Skip if lines are on different pages
            if line_a.get('page_number') != line_b.get('page_number'):
                continue
            
            page_num = line_a.get('page_number', 1)
            page_stats = page_statistics.get(page_num, {'median_gap': 12.0})
            
            # Calculate features
            features = calculate_features_for_merging(line_a, line_b, page_stats)
            
            # Add text fields - both markdown and span text
            features['text_a'] = line_a.get('md_text_cleaned', '').strip()
            features['text_b'] = line_b.get('md_text_cleaned', '').strip()
            
            # Add span text fields - cleaned to remove bullet points and formatting
            features['span_text_a'] = clean_span_text(line_a.get('span_text', ''))
            features['span_text_b'] = clean_span_text(line_b.get('span_text', ''))
            
            features['label'] = ''  # Empty label for manual annotation
            
            # Skip if either text is empty
            if not features['text_a'] or not features['text_b']:
                continue
            
            # Write row to CSV
            writer.writerow(features)
            row_count += 1
    
    print(f"Successfully created '{output_csv_path}' with {row_count} feature rows for {pdf_name}")
    
    # Print statistics about hash features
    hash_stats = {
        'total_pairs': row_count,
        'hashed_a': sum(1 for line in matched_lines if line.get('is_hashed', False)),
        'both_hashed_pairs': 0,
        'neither_hashed_pairs': 0
    }
    
    # Count hash pair statistics from the generated CSV
    with open(output_csv_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('both_hashed') == '1':
                hash_stats['both_hashed_pairs'] += 1
            elif row.get('neither_hashed') == '1':
                hash_stats['neither_hashed_pairs'] += 1
    
    print(f"Hash feature statistics:")
    print(f"  Total hashed lines: {hash_stats['hashed_a']}")
    print(f"  Pairs where both are hashed: {hash_stats['both_hashed_pairs']}")
    print(f"  Pairs where neither is hashed: {hash_stats['neither_hashed_pairs']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python csv_generator.py input.pdf")
        sys.exit(1)
    
    pdf_name = sys.argv[1]
    generate_csv_from_aggregated(pdf_name)