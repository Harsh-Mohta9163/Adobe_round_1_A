import os
import sys
import time
import subprocess
import glob
from pathlib import Path

# Add the app directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'extractor'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'models_code'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'merging'))

# Import necessary modules
from app.extractor.extractor import extract_all_pdfs
from app.merging.merge_textlines import merge_textlines

class DocumentProcessingPipeline:
    def __init__(self):
        """Initialize the complete document processing pipeline"""
        self.input_folder = "./data/input"
        self.intermediate_folders = {
            'textlines_csv': './data/textlines_csv_output',
            'textline_predictions': './data/textline_predictions',
            'merged_textblocks': './data/merged_textblocks',
            'textblock_predictions': './data/textblock_predictions',
            'final_results': './data/final_results'
        }
        
        # Create all necessary directories
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary output directories"""
        directories = [
            self.input_folder,
            "./data/md_files",
            "./data/spans_output", 
            "./data/aggregator_output",
            "./data/textlines_csv_output",
            "./data/textline_predictions",
            "./data/merged_textblocks",
            "./data/textblock_predictions",
            "./data/final_results",
            "./data/output_model1"  # Add model directory
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created/verified directory: {directory}")
    
    def step1_extract_pdfs(self):
        """Step 1: Extract text from PDFs using the extractor pipeline"""
        print(f"\n{'='*80}")
        print("STEP 1: PDF EXTRACTION")
        print(f"{'='*80}")
        
        # Change to the extractor directory to run extraction
        original_cwd = os.getcwd()
        try:
            os.chdir('./app/extractor')
            
            # Get all PDF files from input folder
            pdf_files = [f for f in os.listdir('../../data/input') if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in ../../data/input")
            
            print(f"Found {len(pdf_files)} PDF files to process:")
            for pdf in pdf_files:
                print(f"  - {pdf}")
            
            # Run the extraction pipeline
            successful, failed, results, timing_data = extract_all_pdfs(pdf_files)
            
            if failed:
                print(f"⚠️  Warning: {len(failed)} PDFs failed extraction")
                for pdf in failed:
                    print(f"  - {pdf}")
            
            print(f"✅ Step 1 completed: {len(successful)} PDFs successfully extracted")
            return successful
            
        finally:
            os.chdir(original_cwd)
    
    def step2_textline_model_testing(self):
        """Step 2: Run textline model testing on extracted CSV files"""
        print(f"\n{'='*80}")
        print("STEP 2: TEXTLINE MODEL TESTING")
        print(f"{'='*80}")
        
        # Use the updated textline model tester function
        from app.models_code.textline_model_tester_batch import test_all_files
        
        # Check if we have CSV files to test
        csv_files = glob.glob(os.path.join(self.intermediate_folders['textlines_csv'], '*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.intermediate_folders['textlines_csv']}")
        
        print(f"Found {len(csv_files)} CSV files to test")
        
        # Run batch testing
        success = test_all_files(
            test_folder=self.intermediate_folders['textlines_csv'],
            output_folder=self.intermediate_folders['textline_predictions']
        )
        
        if success:
            print("✅ Step 2 completed: Textline model testing finished")
        else:
            raise RuntimeError("Textline model testing failed")
    
    def step3_merge_textlines(self):
        """Step 3: Merge textlines based on model predictions"""
        print(f"\n{'='*80}")
        print("STEP 3: MERGE TEXTLINES")
        print(f"{'='*80}")
        
        # Get all prediction CSV files
        prediction_files = glob.glob(os.path.join(self.intermediate_folders['textline_predictions'], '*.csv'))
        
        if not prediction_files:
            raise FileNotFoundError(f"No prediction files found in {self.intermediate_folders['textline_predictions']}")
        
        successful_merges = 0
        
        for pred_file in prediction_files:
            filename = os.path.basename(pred_file)
            
            # Create output filename for merged textblocks
            if filename.startswith('predictions_'):
                output_filename = filename.replace('predictions_', 'merged_textblocks_')
            else:
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"merged_textblocks_{name_without_ext}.csv"
            
            output_path = os.path.join(self.intermediate_folders['merged_textblocks'], output_filename)
            
            print(f"📄 Merging: {filename}")
            
            try:
                success = merge_textlines(pred_file, output_path)
                if success:
                    successful_merges += 1
                    print(f"✅ Successfully merged: {output_filename}")
                else:
                    print(f"❌ Failed to merge: {filename}")
            except Exception as e:
                print(f"❌ Error merging {filename}: {e}")
        
        print(f"✅ Step 3 completed: {successful_merges} files successfully merged")
    
    def step4_textblock_model_testing(self):
        """Step 4: Run textblock model testing on merged textblocks"""
        print(f"\n{'='*80}")
        print("STEP 4: TEXTBLOCK MODEL TESTING")
        print(f"{'='*80}")
        
        # Use the updated textblock model tester function
        from app.models_code.textblock_model_tester_batch import test_all_textblock_files
        
        # Check if we have merged files to test
        merged_files = glob.glob(os.path.join(self.intermediate_folders['merged_textblocks'], '*.csv'))
        if not merged_files:
            raise FileNotFoundError(f"No merged textblock files found in {self.intermediate_folders['merged_textblocks']}")
        
        print(f"Found {len(merged_files)} merged textblock files to test")
        
        # Run batch testing
        success = test_all_textblock_files(
            input_folder=self.intermediate_folders['merged_textblocks'],
            output_folder=self.intermediate_folders['textblock_predictions'],
            model_dir='./app/models/textblock_models'
        )
        
        if success:
            print("✅ Step 4 completed: Textblock model testing finished")
        else:
            raise RuntimeError("Textblock model testing failed")
    
    def step5_run_hierarchy(self):
        """Step 5: Run hierarchy analysis on textblock predictions"""
        print(f"\n{'='*80}")
        print("STEP 5: HIERARCHY ANALYSIS")
        print(f"{'='*80}")
        
        # Use the updated hierarchy function
        from app.models_code.run_hierarchy_batch import process_all_hierarchy_files
        
        # Check if we have textblock prediction files
        prediction_files = glob.glob(os.path.join(self.intermediate_folders['textblock_predictions'], '*.csv'))
        if not prediction_files:
            raise FileNotFoundError(f"No textblock prediction files found in {self.intermediate_folders['textblock_predictions']}")
        
        print(f"Found {len(prediction_files)} textblock prediction files to process")
        
        # Run batch hierarchy processing
        success = process_all_hierarchy_files(
            input_folder=self.intermediate_folders['textblock_predictions'],
            output_folder=self.intermediate_folders['final_results']
        )
        
        if success:
            print("✅ Step 5 completed: Hierarchy analysis finished")
        else:
            raise RuntimeError("Hierarchy analysis failed")
    
    def run_complete_pipeline(self):
        """Run the complete document processing pipeline"""
        start_time = time.time()
        
        print(f"{'='*80}")
        print("STARTING COMPLETE DOCUMENT PROCESSING PIPELINE")
        print(f"{'='*80}")
        print(f"Input folder: {self.input_folder}")
        print(f"Final results folder: {self.intermediate_folders['final_results']}")
        
        try:
            # Step 1: Extract PDFs
            successful_pdfs = self.step1_extract_pdfs()
            
            if not successful_pdfs:
                raise RuntimeError("No PDFs were successfully extracted in Step 1")
            
            # Step 2: Textline model testing
            self.step2_textline_model_testing()
            
            # Step 3: Merge textlines
            self.step3_merge_textlines()
            
            # Step 4: Textblock model testing
            self.step4_textblock_model_testing()
            
            # Step 5: Hierarchy analysis
            self.step5_run_hierarchy()
            
            # Final summary
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"⏱️  Total processing time: {total_time:.2f} seconds")
            print(f"📁 Final results available in: {self.intermediate_folders['final_results']}")
            
            # List final output files
            final_files = glob.glob(os.path.join(self.intermediate_folders['final_results'], '*.csv'))
            print(f"📄 Generated {len(final_files)} final result files:")
            for file in final_files:
                print(f"   - {os.path.basename(file)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the complete pipeline"""
    pipeline = DocumentProcessingPipeline()
    
    # Check if input folder has PDF files
    pdf_files = [f for f in os.listdir(pipeline.input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pipeline.input_folder}")
        print(f"Please add PDF files to the input folder and try again.")
        return
    
    print(f"🔍 Found {len(pdf_files)} PDF files in input folder")
    
    # Run the complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ All steps completed successfully!")
        print(f"📁 Check the final results in: {pipeline.intermediate_folders['final_results']}")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()