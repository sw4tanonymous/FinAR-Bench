import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from process_table_results import process_results
    logging.info("Successfully imported process_results function")
except ImportError as e:
    logging.error(f"Failed to import process_results: {str(e)}")

def extract_model_name(input_file):
    """Get model name from input file path"""
    path_parts = input_file.split('/')
    if len(path_parts) >= 2:
        # Return the second last part, usually the model name
        return path_parts[-2]
    else:
        return os.path.splitext(os.path.basename(input_file))[0]

def process_file(input_file, number_theta, list_type):
    """Process a single file with the given parameters"""
    model_name = extract_model_name(input_file)
    model_name = model_name.replace('/', '_').replace('\\', '_').replace(' ', '_')

    output_dir = os.path.join('results_summary', list_type, model_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    output_file = os.path.join(output_dir, f'{model_name}_theta_{number_theta}.json')
    logging.info(f"Output file: {output_file}")
    
    prefix_list = ["结果：\n", "markdown\n",'结果如下\n','财务指标\n','财务指标计算\n','数据：\n'] 
    suffix_list = ["\n检查", "\n注",'\n### 计算','\n计算','\n* 注：','\n解释：','\n说明','\n请注意','\n**注意','\n这']
    
    success = process_results(
        input_file=input_file,
        output_file=output_file,
        prefix_list=prefix_list,
        suffix_list=suffix_list,
        number_theta=number_theta,
        company_list=None
    )
    
    if success:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logging.info(f"Process successful: {output_file} (Size: {file_size} bytes)")
            return True
        else:
            logging.error(f"Function call succeeded but file was not created: {output_file}")
    else:
        logging.error(f"Function call failed for {input_file}")
    
    return False

def main():
    # Create results directory
    os.makedirs('results_summary', exist_ok=True)
    
    # Dictionary of file lists by type
    file_lists = {
        "text": ['experiments_text/without_more_prompt/mixtral-8x7b-instruct-v0.1/results_mixtral-8x7b-instruct-v0.1_05161621.json'],
        "text_more_prompt": [],
        "mineru": [],
        "pdfminer": [],
        "pdfplumber": [],
        "pdftotext": [],
        "pymupdf": [],
        "pypdf": []
    }
    
    # Select which file list to use (modify here to choose)
    list_type = "text"
    file_list = file_lists[list_type]
    
    logging.info(f"Using file list: {list_type}")
    logging.info(f"Files to process: {len(file_list)}")
    
    # Values of number_theta to test
    number_theta_values = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 2]
    # Uncomment for more values: [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 2]
    
    # Process files
    processed_files = 0
    failed_files = 0
    
    for input_file in file_list:
        if not os.path.exists(input_file):
            logging.error(f"File not found: {input_file}")
            failed_files += 1
            continue
        
        file_processed = False
        for number_theta in number_theta_values:
            if process_file(input_file, number_theta, list_type):
                file_processed = True
                
        if file_processed:
            processed_files += 1
        else:
            failed_files += 1
    
    logging.info(f"Processing summary:")
    logging.info(f"  - Total files: {len(file_list)}")
    logging.info(f"  - Successfully processed: {processed_files}")
    logging.info(f"  - Failed: {failed_files}")
    logging.info(f"Results saved to: results_summary/{list_type}/")

if __name__ == '__main__':
    main() 