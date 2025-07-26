#!/usr/bin/env python3
import os
import sys
import shutil
import glob
import json
from pathlib import Path

# Add the app directories to Python path
sys.path.append('/app')
sys.path.append('/app/app/extractor')
sys.path.append('/app/app/models_code')
sys.path.append('/app/app/merging')

def setup_environment():
    """Setup the environment for the pipeline"""
    print("🐳 Setting up Docker environment...")
    
    # Copy PDFs from /app/input to /app/data/input (internal processing)
    input_pdfs = glob.glob('/app/input/*.pdf')
    if not input_pdfs:
        print("❌ No PDF files found in /app/input")
        return False
    
    print(f"📁 Found {len(input_pdfs)} PDF files to process:")
    for pdf in input_pdfs:
        pdf_name = os.path.basename(pdf)
        shutil.copy2(pdf, f'/app/data/input/{pdf_name}')
        print(f"  - {pdf_name}")
    
    return True

def convert_csv_to_json(csv_file, output_dir):
    """Convert CSV hierarchy results to JSON format for output"""
    try:
        import pandas as pd
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Extract base filename (remove hierarchy_ prefix and .csv extension)
        base_name = os.path.basename(csv_file)
        if base_name.startswith('hierarchy_textblock_predictions_merged_textblocks_predictions_textlines_ground_truth_'):
            # Complex naming pattern - extract the original PDF name
            pdf_name = base_name.replace('hierarchy_textblock_predictions_merged_textblocks_predictions_textlines_ground_truth_', '').replace('.csv', '.json')
        elif base_name.startswith('hierarchy_'):
            pdf_name = base_name.replace('hierarchy_', '').replace('.csv', '.json')
        else:
            pdf_name = base_name.replace('.csv', '.json')
        
        # Create JSON structure with document hierarchy
        json_data = {
            "document_metadata": {
                "source_pdf": pdf_name.replace('.json', '.pdf'),
                "processing_timestamp": "",
                "total_text_blocks": len(df),
                "hierarchy_levels": df['hierarchy_level'].value_counts().to_dict() if 'hierarchy_level' in df.columns else {}
            },
            "document_structure": []
        }
        
        # Process each text block in the hierarchy
        for _, row in df.iterrows():
            block_data = {
                "text": row.get('text', ''),
                "hierarchy_level": row.get('hierarchy_level', 'Text'),
                "font_size": row.get('avg_font_size', 12.0),
                "page_number": row.get('page_number', 1),
                "bbox": row.get('bbox', [0, 0, 0, 0]),
                "style_features": {
                    "is_bold": row.get('is_bold_A', 0),
                    "is_italic": row.get('is_italic_A', 0),
                    "is_all_caps": row.get('is_all_caps', 0),
                    "word_count": row.get('word_count', 0)
                }
            }
            json_data["document_structure"].append(block_data)
        
        # Save as JSON
        output_file = os.path.join(output_dir, pdf_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Converted {os.path.basename(csv_file)} to {pdf_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {csv_file} to JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_pipeline():
    """Run the complete document processing pipeline"""
    try:
        print("🚀 Starting Complete Document Processing Pipeline...")
        
        # Import and initialize the pipeline
        from complete_pipeline import DocumentProcessingPipeline
        
        # Create pipeline instance with correct paths
        pipeline = DocumentProcessingPipeline()
        
        # Override paths to use Docker mount points
        pipeline.input_folder = "/app/data/input"
        pipeline.intermediate_folders = {
            'textlines_csv': '/app/data/textlines_csv_output',
            'textline_predictions': '/app/data/textline_predictions',
            'merged_textblocks': '/app/data/merged_textblocks',
            'textblock_predictions': '/app/data/textblock_predictions',
            'final_results': '/app/data/final_results'
        }
        
        # Ensure all directories exist
        for folder in pipeline.intermediate_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Run the complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if not success:
            print("❌ Pipeline execution failed")
            return False
            
        print("✅ Complete pipeline finished successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error running complete pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def copy_results_to_output():
    """Copy final results to /app/output and convert to JSON"""
    try:
        # Create output directory
        os.makedirs('/app/output', exist_ok=True)
        
        # Find all CSV files in final results
        csv_files = glob.glob('/app/data/final_results/*.csv')
        
        if not csv_files:
            print("⚠️ No result files found in /app/data/final_results")
            # Check other possible locations
            alternative_paths = [
                '/app/data/textblock_predictions/*.csv',
                '/app/data/merged_textblocks/*.csv'
            ]
            for alt_path in alternative_paths:
                alt_files = glob.glob(alt_path)
                if alt_files:
                    print(f"📁 Found {len(alt_files)} files in {os.path.dirname(alt_path)}")
                    csv_files.extend(alt_files)
                    break
        
        if not csv_files:
            print("❌ No result files found in any location")
            return False
        
        print(f"📄 Converting {len(csv_files)} result files to JSON...")
        
        successful_conversions = 0
        for csv_file in csv_files:
            print(f"🔄 Processing: {os.path.basename(csv_file)}")
            if convert_csv_to_json(csv_file, '/app/output'):
                successful_conversions += 1
        
        print(f"✅ Successfully converted {successful_conversions}/{len(csv_files)} files")
        
        # List the output files
        output_files = glob.glob('/app/output/*.json')
        print(f"📁 Generated output files:")
        for file in output_files:
            print(f"  - {os.path.basename(file)}")
        
        return successful_conversions > 0
        
    except Exception as e:
        print(f"❌ Error copying results: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function for Docker container"""
    print("=" * 80)
    print("🐳 ADOBE DOCUMENT PROCESSING PIPELINE - DOCKER EXECUTION")
    print("=" * 80)
    
    try:
        # Step 1: Setup environment and copy input files
        print("\n" + "="*50)
        print("STEP 1: ENVIRONMENT SETUP")
        print("="*50)
        if not setup_environment():
            print("❌ Environment setup failed")
            sys.exit(1)
        
        # Step 2: Run the complete pipeline
        print("\n" + "="*50)
        print("STEP 2: RUN COMPLETE PIPELINE")
        print("="*50)
        if not run_complete_pipeline():
            print("❌ Pipeline execution failed")
            sys.exit(1)
        
        # Step 3: Copy results to output and convert to JSON
        print("\n" + "="*50)
        print("STEP 3: OUTPUT GENERATION")
        print("="*50)
        if not copy_results_to_output():
            print("❌ Failed to generate output files")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("🎉 DOCKER EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📁 Check /app/output for the generated JSON files")
        print("📊 Each input PDF has a corresponding JSON output file")
        
    except Exception as e:
        print(f"\n❌ DOCKER EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()