import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the functions from other modules
from markdowntext import pdf_to_markdown
from span_extractor import extract_columns_and_split
from aggregator import aggregate_md_to_spans
from csv_generator import generate_csv_from_aggregated

def create_output_directories():
    """Create all necessary output directories"""
    directories = [
        "../../data/md_files",
        "../../data/spans_output", 
        "../../data/aggregator_output",
        "../../data/textlines_csv_output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")

def process_markdown(pdf_name):
    """Process PDF to Markdown conversion"""
    print(f"[MD] Starting markdown conversion for {pdf_name}")
    start_time = time.time()
    
    try:
        # Check if input file exists
        input_pdf_path = f"../../data/input/{pdf_name}"
        if not os.path.exists(input_pdf_path):
            raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")
        
        # Convert PDF to Markdown - only pass pdf_name
        pdf_to_markdown(pdf_name)
        
        elapsed_time = time.time() - start_time
        print(f"[MD] ✓ Completed markdown conversion for {pdf_name} in {elapsed_time:.2f}s")
        return True, f"Markdown conversion successful for {pdf_name}", elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"[MD] ✗ Error processing {pdf_name}: {str(e)} (after {elapsed_time:.2f}s)"
        print(error_msg)
        return False, error_msg, elapsed_time

def process_spans(pdf_name):
    """Process PDF span extraction"""
    print(f"[SPAN] Starting span extraction for {pdf_name}")
    start_time = time.time()
    
    try:
        # Check if input file exists
        input_pdf_path = f"../../data/input/{pdf_name}"
        if not os.path.exists(input_pdf_path):
            raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")
        
        # Extract spans
        extract_columns_and_split(pdf_name)
        
        elapsed_time = time.time() - start_time
        print(f"[SPAN] ✓ Completed span extraction for {pdf_name} in {elapsed_time:.2f}s")
        return True, f"Span extraction successful for {pdf_name}", elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"[SPAN] ✗ Error processing {pdf_name}: {str(e)} (after {elapsed_time:.2f}s)"
        print(error_msg)
        return False, error_msg, elapsed_time

def process_aggregation(pdf_name):
    """Process aggregation of markdown and spans"""
    print(f"[AGG] Starting aggregation for {pdf_name}")
    start_time = time.time()
    
    try:
        # Check if required input files exist
        base_filename = pdf_name.replace('.pdf', '')
        md_file = f"../../data/md_files/{base_filename}.md"
        json_file = f"../../data/spans_output/spans_{pdf_name}.json"
        
        if not os.path.exists(md_file):
            raise FileNotFoundError(f"Markdown file not found: {md_file}")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Spans JSON file not found: {json_file}")
        
        # Run aggregation
        aggregate_md_to_spans(pdf_name)
        
        elapsed_time = time.time() - start_time
        print(f"[AGG] ✓ Completed aggregation for {pdf_name} in {elapsed_time:.2f}s")
        return True, f"Aggregation successful for {pdf_name}", elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"[AGG] ✗ Error processing {pdf_name}: {str(e)} (after {elapsed_time:.2f}s)"
        print(error_msg)
        return False, error_msg, elapsed_time

def process_csv_generation(pdf_name):
    """Process CSV generation from aggregated data"""
    print(f"[CSV] Starting CSV generation for {pdf_name}")
    start_time = time.time()
    
    try:
        # Check if required input file exists
        aggregated_file = f"../../data/aggregator_output/aggregated_{pdf_name}.json"
        
        if not os.path.exists(aggregated_file):
            raise FileNotFoundError(f"Aggregated JSON file not found: {aggregated_file}")
        
        # Generate CSV
        generate_csv_from_aggregated(pdf_name)
        
        elapsed_time = time.time() - start_time
        print(f"[CSV] ✓ Completed CSV generation for {pdf_name} in {elapsed_time:.2f}s")
        return True, f"CSV generation successful for {pdf_name}", elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"[CSV] ✗ Error processing {pdf_name}: {str(e)} (after {elapsed_time:.2f}s)"
        print(error_msg)
        return False, error_msg, elapsed_time

def process_single_pdf(pdf_name):
    """Process a single PDF through the entire pipeline"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {pdf_name}")
    print(f"{'='*60}")
    
    total_start_time = time.time()
    results = {}
    timing_data = {}
    
    # Step 1: Parallel processing of Markdown and Spans
    print(f"\n[STEP 1] Starting parallel processing (Markdown + Spans) for {pdf_name}")
    step1_start = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        md_future = executor.submit(process_markdown, pdf_name)
        span_future = executor.submit(process_spans, pdf_name)
        
        # Wait for both to complete
        md_success, md_result, md_time = md_future.result()
        span_success, span_result, span_time = span_future.result()
        
        results['markdown'] = (md_success, md_result)
        results['spans'] = (span_success, span_result)
        timing_data['markdown_time'] = md_time
        timing_data['spans_time'] = span_time
    
    step1_time = time.time() - step1_start
    timing_data['step1_total_time'] = step1_time
    print(f"[STEP 1] Completed in {step1_time:.2f}s")
    print(f"  ├─ Markdown: {timing_data['markdown_time']:.2f}s")
    print(f"  └─ Spans: {timing_data['spans_time']:.2f}s")
    
    # Check if Step 1 was successful
    if not (md_success and span_success):
        print(f"\n[ERROR] Step 1 failed for {pdf_name}. Skipping remaining steps.")
        print(f"  Markdown: {'✓' if md_success else '✗'}")
        print(f"  Spans: {'✓' if span_success else '✗'}")
        
        total_time = time.time() - total_start_time
        timing_data['total_time'] = total_time
        
        return False, results, timing_data
    
    # Step 2: Aggregation
    print(f"\n[STEP 2] Starting aggregation for {pdf_name}")
    agg_success, agg_result, agg_time = process_aggregation(pdf_name)
    results['aggregation'] = (agg_success, agg_result)
    timing_data['aggregation_time'] = agg_time
    
    if not agg_success:
        print(f"\n[ERROR] Step 2 (aggregation) failed for {pdf_name}. Skipping CSV generation.")
        
        total_time = time.time() - total_start_time
        timing_data['total_time'] = total_time
        
        return False, results, timing_data
    
    # Step 3: CSV Generation
    print(f"\n[STEP 3] Starting CSV generation for {pdf_name}")
    csv_success, csv_result, csv_time = process_csv_generation(pdf_name)
    results['csv'] = (csv_success, csv_result)
    timing_data['csv_time'] = csv_time
    
    total_time = time.time() - total_start_time
    timing_data['total_time'] = total_time
    
    # Print detailed timing summary for this PDF
    print(f"\n{'='*50}")
    print(f"TIMING SUMMARY FOR {pdf_name}")
    print(f"{'='*50}")
    print(f"Step 1 - Parallel Processing: {timing_data['step1_total_time']:.2f}s")
    print(f"  ├─ Markdown conversion: {timing_data['markdown_time']:.2f}s")
    print(f"  └─ Span extraction: {timing_data['spans_time']:.2f}s")
    print(f"Step 2 - Aggregation: {timing_data['aggregation_time']:.2f}s")
    print(f"Step 3 - CSV Generation: {timing_data['csv_time']:.2f}s")
    print(f"{'─'*50}")
    print(f"TOTAL PROCESSING TIME: {timing_data['total_time']:.2f}s")
    print(f"{'='*50}")
    
    if csv_success:
        print(f"\n✓ COMPLETED: {pdf_name} - Total time: {total_time:.2f}s")
        return True, results, timing_data
    else:
        print(f"\n✗ FAILED: {pdf_name} - Total time: {total_time:.2f}s")
        return False, results, timing_data

def extract_all_pdfs(pdf_names):
    """
    Process multiple PDFs through the extraction pipeline
    Args:
        pdf_names (list): List of PDF filenames (e.g., ['file01.pdf', 'file02.pdf'])
    """
    print("Starting PDF extraction pipeline...")
    print(f"PDFs to process: {pdf_names}")
    
    # Create output directories
    create_output_directories()
    
    overall_start_time = time.time()
    successful_pdfs = []
    failed_pdfs = []
    all_results = {}
    all_timing_data = {}
    
    # Process each PDF
    for i, pdf_name in enumerate(pdf_names, 1):
        print(f"\n\nPROCESSING PDF {i}/{len(pdf_names)}: {pdf_name}")
        
        # Check if input file exists before processing
        input_path = f"../../data/input/{pdf_name}"
        if not os.path.exists(input_path):
            print(f"[ERROR] Input file not found: {input_path}")
            failed_pdfs.append(pdf_name)
            all_results[pdf_name] = (False, {"error": f"Input file not found: {input_path}"})
            all_timing_data[pdf_name] = {"error": "File not found", "total_time": 0}
            continue
        
        success, results, timing_data = process_single_pdf(pdf_name)
        all_results[pdf_name] = (success, results)
        all_timing_data[pdf_name] = timing_data
        
        if success:
            successful_pdfs.append(pdf_name)
        else:
            failed_pdfs.append(pdf_name)
    
    # Print final summary with detailed timing
    total_time = time.time() - overall_start_time
    print(f"\n\n{'='*80}")
    print("EXTRACTION PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total PDFs processed: {len(pdf_names)}")
    print(f"Successful: {len(successful_pdfs)}")
    print(f"Failed: {len(failed_pdfs)}")
    print(f"Success rate: {len(successful_pdfs)/len(pdf_names)*100:.1f}%")
    
    if successful_pdfs:
        print(f"\n✓ Successfully processed:")
        total_md_time = 0
        total_span_time = 0
        total_agg_time = 0
        total_csv_time = 0
        
        for pdf in successful_pdfs:
            timing = all_timing_data[pdf]
            total_processing_time = timing['total_time']
            
            print(f"  - {pdf}: {total_processing_time:.2f}s")
            print(f"    ├─ Markdown: {timing.get('markdown_time', 0):.2f}s")
            print(f"    ├─ Spans: {timing.get('spans_time', 0):.2f}s")
            print(f"    ├─ Aggregation: {timing.get('aggregation_time', 0):.2f}s")
            print(f"    └─ CSV: {timing.get('csv_time', 0):.2f}s")
            
            total_md_time += timing.get('markdown_time', 0)
            total_span_time += timing.get('spans_time', 0)
            total_agg_time += timing.get('aggregation_time', 0)
            total_csv_time += timing.get('csv_time', 0)
        
        print(f"\n📊 AGGREGATE TIMING ANALYSIS:")
        print(f"  ├─ Total Markdown time: {total_md_time:.2f}s (avg: {total_md_time/len(successful_pdfs):.2f}s per PDF)")
        print(f"  ├─ Total Span time: {total_span_time:.2f}s (avg: {total_span_time/len(successful_pdfs):.2f}s per PDF)")
        print(f"  ├─ Total Aggregation time: {total_agg_time:.2f}s (avg: {total_agg_time/len(successful_pdfs):.2f}s per PDF)")
        print(f"  └─ Total CSV time: {total_csv_time:.2f}s (avg: {total_csv_time/len(successful_pdfs):.2f}s per PDF)")
        
        # Calculate efficiency metrics
        sequential_time = total_md_time + total_span_time + total_agg_time + total_csv_time
        parallel_savings = sequential_time - total_time
        print(f"\n⚡ PARALLELIZATION EFFICIENCY:")
        print(f"  ├─ Sequential processing time would be: {sequential_time:.2f}s")
        print(f"  ├─ Actual parallel processing time: {total_time:.2f}s")
        print(f"  └─ Time saved through parallelization: {parallel_savings:.2f}s ({parallel_savings/sequential_time*100:.1f}%)")
    
    if failed_pdfs:
        print(f"\n✗ Failed to process:")
        for pdf in failed_pdfs:
            timing = all_timing_data.get(pdf, {})
            if 'error' in timing:
                print(f"  - {pdf}: {timing['error']}")
            else:
                print(f"  - {pdf}: Failed after {timing.get('total_time', 0):.2f}s")
    
    print(f"\nOutput files saved in:")
    print(f"  - Markdown files: ../../data/md_files/")
    print(f"  - Span JSON files: ../../data/spans_output/")
    print(f"  - Aggregated JSON files: ../../data/aggregator_output/")
    print(f"  - CSV files: ../../data/textlines_csv_output/")
    
    return successful_pdfs, failed_pdfs, all_results, all_timing_data

if __name__ == "__main__":
    # Define list of PDF files to process
    pdf_names = [
        "file01.pdf",
        "file02.pdf",
        "file03.pdf",
        "file04.pdf",
        "file05.pdf",
        "file06.pdf",
        "file07.pdf",
    ]
    
    # You can also specify PDFs via command line arguments
    if len(sys.argv) > 1:
        pdf_names = sys.argv[1:]
        print(f"Using PDFs from command line: {pdf_names}")
    
    # Run the extraction pipeline
    successful, failed, results, timing_data = extract_all_pdfs(pdf_names)
    
    # Exit with appropriate code
    if failed:
        sys.exit(1)  # Exit with error if any PDF failed
    else:
        sys.exit(0)  # Exit successfully if all PDFs processed