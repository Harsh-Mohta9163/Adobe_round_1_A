# round1a/app/extractor/extractor.py
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Import your newly refactored functions from your worker scripts
# IMPORTANT: Make sure these helper scripts are refactored to accept full paths
from markdowntext import pdf_to_markdown
from span_extractor import extract_columns_and_split
from aggregator import aggregate_md_to_spans
from csv_generator import generate_csv_from_aggregated

def process_markdown(input_pdf_path, output_md_path):
    """Wrapper function to time and call the markdown converter."""
    start_time = time.time()
    try:
        pdf_to_markdown(input_pdf_path, output_md_path)
        elapsed = time.time() - start_time
        print(f"[MD] ✓ Completed in {elapsed:.2f}s")
        return True, "Success", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[MD] ✗ Error: {e} (after {elapsed:.2f}s)")
        return False, str(e), elapsed

def process_spans(input_pdf_path, output_spans_path):
    """Wrapper function to time and call the span extractor."""
    start_time = time.time()
    try:
        extract_columns_and_split(input_pdf_path, output_spans_path)
        elapsed = time.time() - start_time
        print(f"[SPAN] ✓ Completed in {elapsed:.2f}s")
        return True, "Success", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[SPAN] ✗ Error: {e} (after {elapsed:.2f}s)")
        return False, str(e), elapsed

def process_single_pdf(pdf_name, input_dir, temp_dir, output_dir):
    """Process a single PDF through the entire pipeline with detailed logging."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {pdf_name}")
    print(f"{'='*60}")
    
    total_start_time = time.time()
    results = {}
    timing_data = {}
    
    # Define paths for this specific PDF
    base_name = pdf_name.replace('.pdf', '')
    paths = {
        "full_pdf_path": os.path.join(input_dir, pdf_name),
        "md_json_path": os.path.join(temp_dir, 'md_files', f"{base_name}.json"),
        "spans_json_path": os.path.join(temp_dir, 'spans_output', f"spans_{pdf_name}.json"),
        "agg_json_path": os.path.join(temp_dir, 'aggregator_output', f"aggregated_{pdf_name}.json"),
        "final_csv_path": os.path.join(output_dir, f"textlines_ground_truth_{pdf_name}.csv")
    }

    # Step 1: Parallel processing of Markdown and Spans
    print(f"\n[STEP 1] Starting parallel processing (Markdown + Spans) for {pdf_name}")
    step1_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        md_future = executor.submit(process_markdown, paths["full_pdf_path"], paths["md_json_path"])
        span_future = executor.submit(process_spans, paths["full_pdf_path"], paths["spans_json_path"])
        
        md_success, md_result, md_time = md_future.result()
        span_success, span_result, span_time = span_future.result()
        
        results['markdown'] = (md_success, md_result)
        results['spans'] = (span_success, span_result)
        timing_data['markdown_time'] = md_time
        timing_data['spans_time'] = span_time
    
    step1_time = time.time() - step1_start
    timing_data['step1_total_time'] = step1_time
    print(f"[STEP 1] Completed in {step1_time:.2f}s")

    if not (md_success and span_success):
        print(f"\n[ERROR] Step 1 failed for {pdf_name}. Skipping remaining steps.")
        return False, results, timing_data

    # Step 2: Aggregation
    print(f"\n[STEP 2] Starting aggregation for {pdf_name}")
    start_agg_time = time.time()
    try:
        aggregate_md_to_spans(paths["md_json_path"], paths["spans_json_path"], paths["agg_json_path"])
        agg_success = True
    except Exception as e:
        print(f"[AGG] ✗ Error: {e}")
        agg_success = False
    timing_data['aggregation_time'] = time.time() - start_agg_time
    
    if not agg_success:
        print(f"\n[ERROR] Step 2 (aggregation) failed for {pdf_name}. Skipping CSV generation.")
        return False, results, timing_data

    # Step 3: CSV Generation
    print(f"\n[STEP 3] Starting CSV generation for {pdf_name}")
    start_csv_time = time.time()
    try:
        generate_csv_from_aggregated(paths["agg_json_path"], paths["final_csv_path"])
        csv_success = True
    except Exception as e:
        print(f"[CSV] ✗ Error: {e}")
        csv_success = False
    timing_data['csv_time'] = time.time() - start_csv_time
    
    total_time = time.time() - total_start_time
    timing_data['total_time'] = total_time

    if csv_success:
        print(f"\n✓ COMPLETED: {pdf_name} - Total time: {total_time:.2f}s")
        return True, results, timing_data
    else:
        print(f"\n✗ FAILED: {pdf_name} - Total time: {total_time:.2f}s")
        return False, results, timing_data

def extract_all_pdfs(input_dir, output_dir, temp_dir):
    """Main orchestration function, now with your detailed summary logging."""
    overall_start_time = time.time()
    
    os.makedirs(os.path.join(temp_dir, 'md_files'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'spans_output'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'aggregator_output'), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("Warning: No PDF files found in input directory.")
        return [], [], {}, {}

    successful_pdfs = []
    failed_pdfs = []
    all_results = {}
    all_timing_data = {}

    for i, pdf_name in enumerate(pdf_files, 1):
        print(f"\n\nPROCESSING PDF {i}/{len(pdf_files)}: {pdf_name}")
        success, results, timing_data = process_single_pdf(pdf_name, input_dir, temp_dir, output_dir)
        all_results[pdf_name] = (success, results)
        all_timing_data[pdf_name] = timing_data
        
        if success:
            successful_pdfs.append(pdf_name)
        else:
            failed_pdfs.append(pdf_name)
    
    # --- RESTORED FINAL SUMMARY ---
    total_time = time.time() - overall_start_time
    print(f"\n\n{'='*80}")
    print("EXTRACTION PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Successful: {len(successful_pdfs)}")
    print(f"Failed: {len(failed_pdfs)}")
    
    if successful_pdfs:
        print(f"\n✓ Successfully processed:")
        for pdf in successful_pdfs:
            print(f"  - {pdf}: {all_timing_data[pdf]['total_time']:.2f}s")
    
    if failed_pdfs:
        print(f"\n✗ Failed to process:")
        for pdf in failed_pdfs:
            print(f"  - {pdf}")
            
    return successful_pdfs, failed_pdfs, all_results, all_timing_data

# This block allows the script to be run directly for local testing
if __name__ == "__main__":
    print("--- Running extractor.py in Standalone (Local) Mode ---")
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    local_input = os.path.join(base_dir, 'data', 'input')
    local_output = os.path.join(base_dir, 'data', 'textlines_csv_output')
    local_temp = os.path.join(base_dir, 'data')
    
    extract_all_pdfs(local_input, local_output, local_temp)