import sys
import os
import json
import argparse
import logging
import traceback
import time
from datetime import datetime
from pathlib import Path
from config import (
    INPUT_FILE_PATH,
    USE_MODEL_FOR_PDF,
    BATCH_SIZE,
    LOGGING_LEVEL,
    MAX_WORKERS,
    TASK_NUM
)

from model_configs import (
    get_model_params, 
    get_interface_type, 
    get_api_key,
    get_base_url,
    INTERFACE_TYPE_VOLCENGINE,
    INTERFACE_TYPE_OPENAI
)

from reasoning_processor import process_reasoning_tasks, setup_logging, load_reasoning_tasks as load_tasks, init_api_client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_reasoning_tasks(file_path, use_textpdf=False, pdf_extractor="pymupdf"):
    # load reasoning tasks from file
    all_data = load_tasks(file_path, use_textpdf, pdf_extractor)
    
    if not all_data:
        return None

    allowed_task_nums = TASK_NUM
    if allowed_task_nums and len(allowed_task_nums) > 0:
        logging.info(f"Only process the following task numbers: {', '.join(allowed_task_nums)}")
    else:
        logging.info("No task number filter specified, will process all tasks")
        allowed_task_nums = None 
    

    reasoning_data = []
    reasoning_instances_count = 0
    filtered_count = 0
    filtered_by_num_count = 0
    
    for data_item in all_data:
        reasoning_instances = []
        for instance in data_item.get("instances", []):
            if instance.get("task_type", "").lower() == "reasoning":
                if allowed_task_nums and str(instance.get("task_num")) not in allowed_task_nums:
                    filtered_by_num_count += 1
                    continue
                reasoning_instances.append(instance)
            else:
                filtered_count += 1
                logging.debug(f"filter out non-reasoning task: {instance.get('task_id')}, type: {instance.get('task_type', '')}")
        
        if reasoning_instances:
            data_copy = data_item.copy()
            data_copy["instances"] = reasoning_instances
            reasoning_data.append(data_copy)
            reasoning_instances_count += len(reasoning_instances)

    log_message = f"successfully loaded {len(reasoning_data)} reasoning tasks, containing {reasoning_instances_count} reasoning instances"
    log_message += f"\nfiltered out {filtered_count} non-reasoning instances"
    if allowed_task_nums:
        log_message += f" filtered out {filtered_by_num_count} instances due to task number mismatch"
    logging.info(log_message)
    
    if reasoning_instances_count == 0:
        logging.warning("no reasoning instances found! please check the data")
    elif filtered_count > 0:
        logging.warning(f"filtered out {filtered_count} non-reasoning instances")
    
    return reasoning_data

def parse_args():
    parser = argparse.ArgumentParser(description="reasoning task evaluation")
    
    # basic parameters
    parser.add_argument("--model", type=str, default="meta/llama-3.1-405b-instruct", 
                      help="The model name to use, e.g. 'meta/llama-3.1-405b-instruct', 'deepseek-r1'")
    parser.add_argument("--mode", choices=["text", "pdf", "textpdf"], default="text", 
                      help="Data source mode: text uses JSON tables, pdf uses PDF files, textpdf uses extracted PDF text")
    parser.add_argument("--output", type=str, default=None, 
                      help="Result save path, default is ./experiments_reasoning/model_name")
    parser.add_argument("--more-prompt", action="store_true",
                      help="Use a more detailed reasoning task prompt, including financial metrics calculation formulas")
    parser.add_argument("--task-types", type=str, default="reasoning",
                      help="The task types to process, separated by commas, default is 'reasoning'")
    
    # optional
    parser.add_argument("--api_key", default=None, help="API key")
    parser.add_argument("--base_url", default=None, help="API base URL")
    parser.add_argument("--input", default=INPUT_FILE_PATH, help="Input data file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers")
    parser.add_argument("--log_level", default=LOGGING_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--use_model_for_pdf", action="store_true", default=USE_MODEL_FOR_PDF, help="Whether to use model to process PDF")
    parser.add_argument("--pdf_extractor", choices=["pdfplumber", "pdfminer", "pypdf", "pymupdf", "pdftotext", "mineru"], 
                       default="pdfplumber", help="When using textpdf mode, select which extractor result")
    
    return parser.parse_args()

def main():
    start_time = time.time()
    
    args = parse_args()
    
    setup_logging(level=args.log_level)
    
    use_pdf = (args.mode.lower() == "pdf")
    use_textpdf = (args.mode.lower() == "textpdf")
    
    if args.output:
        output_dir = args.output
    else:
        model_short_name = args.model.split('/')[-1] if '/' in args.model else args.model
        if args.more_prompt:
            output_dir = f"./experiments_reasoning/{args.mode}/with_more_prompt/{model_short_name}"
        else:
            output_dir = f"./experiments_reasoning/{args.mode}/without_more_prompt/{model_short_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_params = get_model_params(args.model)
    interface_type = get_interface_type(args.model)
    
    logging.info("\n=============== Reasoning Evaluation ===============")
    logging.info(f"model: {args.model}")
    
    if interface_type == INTERFACE_TYPE_OPENAI:
        logging.info(f"using OpenAI compatible API")
        logging.info(f"API URL: {get_base_url(args.model) or 'Default'}")
    elif interface_type == INTERFACE_TYPE_VOLCENGINE:
        logging.info(f"using Volcengine API")
    
    logging.info(f"model parameters: temperature={model_params.get('temperature', 0.0)}, top_p={model_params.get('top_p', 0.9)}, max_tokens={model_params.get('max_tokens', 1024)}")
    logging.info(f"data mode: {'PDF' if use_pdf else ('extracted text from PDF' if use_textpdf else 'text')}")
    
    if use_textpdf:
        logging.info(f"PDF text extractor: {args.pdf_extractor}")
    
    if args.more_prompt:
        logging.info(f"using detailed financial indicator calculation prompt")
    
    logging.info(f"task types: {args.task_types}")
    
    logging.info(f"input file: {args.input}")
    logging.info(f"output directory: {output_dir}")
    logging.info(f"batch size: {args.batch_size}")
    logging.info(f"number of parallel workers: {args.workers}")
    logging.info(f"log level: {args.log_level}")
    logging.info("=======================================\n")
    
    try:
        results_file = process_reasoning_tasks(
            model_name=args.model,
            input_file=args.input,
            output_dir=output_dir,
            batch_size=args.batch_size,
            use_pdf=use_pdf,
            use_textpdf=use_textpdf,
            pdf_extractor=args.pdf_extractor,
            use_model_for_pdf=args.use_model_for_pdf,
            more_prompt=args.more_prompt,
            task_types=args.task_types
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if results_file:
            logging.info(f"\nprocessing complete, total time: {elapsed:.2f} seconds, overall results saved to: {results_file}")
        
    except Exception as e:
        logging.error(f"\nerror processing reasoning tasks: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 