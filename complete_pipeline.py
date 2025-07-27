import os
import sys
import time
import glob
from pathlib import Path
# Add the app directories to the Python path relative to this file's location
# This makes the script runnable from any directory
# sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
# Add the app directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'extractor'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'models_code'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'merging'))
# Your imports for extractor, merging, etc.
from app.extractor.extractor import extract_all_pdfs
from app.merging.merge_textlines import merge_textlines
from app.models_code.textline_model_tester_batch import test_all_files
from app.models_code.textblock_model_tester_batch import test_all_textblock_files
from app.models_code.run_hierarchy_batch import process_all_hierarchy_files


class DocumentProcessingPipeline:

    def __init__(self, input_folder, final_output_folder):
        """Initialize the pipeline with master input/output paths."""
        self.input_folder = input_folder
        self.final_output_folder = final_output_folder
        
        # All intermediate files will live in one temporary directory inside the container.
        self.temp_dir = "/app/data"
        
        # Define paths for each intermediate step using the temp directory
        self.intermediate_paths = {
            'textlines_csv': os.path.join(self.temp_dir, 'textlines_csv_output'),
            'textline_predictions': os.path.join(self.temp_dir, 'textline_predictions'),
            'merged_textblocks': os.path.join(self.temp_dir, 'merged_textblocks'),
            'textblock_predictions': os.path.join(self.temp_dir, 'textblock_predictions'),
        }
        self.create_directories()

    def create_directories(self):
        """Create all necessary directories for the pipeline to run."""
        print("✓ Creating necessary directories...")
        os.makedirs(self.temp_dir, exist_ok=True)
        for path in self.intermediate_paths.values():
            os.makedirs(path, exist_ok=True)
        os.makedirs(self.final_output_folder, exist_ok=True)

    def step1_extract_pdfs(self):
        print("\n--- STEP 1: PDF EXTRACTION ---")
        successful, failed, _, _ = extract_all_pdfs(
            self.input_folder,
            self.intermediate_paths['textlines_csv'],self.temp_dir 
        )
        if failed:
            print(f"⚠️  Warning: {len(failed)} PDFs failed extraction: {failed}")
        if not successful:
            print("❌ Step 1 failed: No PDFs were successfully extracted.")
            return None
        print(f"✅ Step 1 completed: {len(successful)} PDFs successfully extracted.")
        return successful

    def step2_textline_model_testing(self):
        print("\n--- STEP 2: TEXTLINE MODEL TESTING ---")
        csv_files = glob.glob(os.path.join(self.intermediate_paths['textlines_csv'], '*.csv'))
        if not csv_files:
            print(f"❌ Step 2 failed: No CSV files found in {self.intermediate_paths['textlines_csv']} to test.")
            return False
        
        success = test_all_files(
            test_folder=self.intermediate_paths['textlines_csv'],
            output_folder=self.intermediate_paths['textline_predictions']
        )
        if success:
            print("✅ Step 2 completed.")
        else:
            print("❌ Step 2 failed.")
        return success

    def step3_merge_textlines(self):
        print("\n--- STEP 3: MERGE TEXTLINES ---")
        prediction_files = glob.glob(os.path.join(self.intermediate_paths['textline_predictions'], '*.csv'))
        if not prediction_files:
            print(f"❌ Step 3 failed: No prediction files found in {self.intermediate_paths['textline_predictions']} for merging.")
            return False
            
        successful_merges = 0
        for pred_file in prediction_files:
            try:
                output_path = os.path.join(self.intermediate_paths['merged_textblocks'], os.path.basename(pred_file))
                if merge_textlines(pred_file, output_path):
                    successful_merges += 1
                else:
                    print(f"⚠️  Warning: Merging failed for {os.path.basename(pred_file)}")
            except Exception as e:
                print(f"❌ Error merging {os.path.basename(pred_file)}: {e}")

        if successful_merges > 0:
            print(f"✅ Step 3 completed: {successful_merges} files merged.")
            return True
        else:
            print("❌ Step 3 failed: No files were successfully merged.")
            return False

    def step4_textblock_model_testing(self):
        print("\n--- STEP 4: TEXTBLOCK MODEL TESTING ---")
        merged_files = glob.glob(os.path.join(self.intermediate_paths['merged_textblocks'], '*.csv'))
        if not merged_files:
            print(f"❌ Step 4 failed: No merged textblock files found in {self.intermediate_paths['merged_textblocks']}.")
            return False
            
        success = test_all_textblock_files(
            input_folder=self.intermediate_paths['merged_textblocks'],
            output_folder=self.intermediate_paths['textblock_predictions'],
            model_dir='./app/models/textblock_models' # This path is relative to the project root
        )
        if success:
            print("✅ Step 4 completed.")
        else:
            print("❌ Step 4 failed.")
        return success

    def step5_run_hierarchy(self):
        print("\n--- STEP 5: HIERARCHY ANALYSIS ---")
        prediction_files = glob.glob(os.path.join(self.intermediate_paths['textblock_predictions'], '*.csv'))
        if not prediction_files:
            print(f"❌ Step 5 failed: No textblock prediction files found in {self.intermediate_paths['textblock_predictions']}.")
            return False

        success = process_all_hierarchy_files(
            input_folder=self.intermediate_paths['textblock_predictions'],
            output_folder=self.final_output_folder 
        )
        if success:
            print("✅ Step 5 completed.")
        else:
            print("❌ Step 5 failed.")
        return success

    def run_complete_pipeline(self):
        """Run the complete document processing pipeline with robust error checking."""
        start_time = time.time()
        
        try:
            if not self.step1_extract_pdfs():
                raise RuntimeError("Step 1 (PDF Extraction) failed, stopping pipeline.")
            
            if not self.step2_textline_model_testing():
                raise RuntimeError("Step 2 (Textline Testing) failed, stopping pipeline.")
            
            if not self.step3_merge_textlines():
                raise RuntimeError("Step 3 (Merge Textlines) failed, stopping pipeline.")
            
            if not self.step4_textblock_model_testing():
                raise RuntimeError("Step 4 (Textblock Testing) failed, stopping pipeline.")
            
            if not self.step5_run_hierarchy():
                raise RuntimeError("Step 5 (Hierarchy Analysis) failed, stopping pipeline.")

            total_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total processing time: {total_time:.2f} seconds")
            print(f"📁 Final results available in: {self.final_output_folder}")
            print(f"{'='*80}")
            return True
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False