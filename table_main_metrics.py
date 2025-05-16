import sys
import os
import json
import argparse
import logging
import traceback
import time
from pathlib import Path
from config import (
    INPUT_FILE_PATH,
    USE_MODEL_FOR_PDF,
    BATCH_SIZE,
    LOGGING_LEVEL,
    MAX_WORKERS,
    NVIDIA_MODEL_NAME,
)

from model_configs import (
    get_model_params, 
    get_interface_type, 
    get_model_id, 
    get_api_key,
    get_base_url,
    INTERFACE_TYPE_VOLCENGINE,
    INTERFACE_TYPE_OPENAI
)

from table_metrics_processor import process_table_metrics, setup_logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="table main metrics evaluation")
    
    parser.add_argument("--model", type=str, default="gpt-4o", 
                      help="the model name, e.g. 'meta/llama-3.1-405b-instruct', 'deepseek-r1'")
    parser.add_argument("--mode", choices=["text", "pdf", "textpdf"], default="text", 
                      help="data source mode: text uses table data in JSON, pdf uses PDF files, textpdf uses extracted PDF text")
    parser.add_argument("--save_path", type=str, default=None, 
                      help="the path to save the results")
    parser.add_argument("--more-prompt", action="store_true",
                      help="use more detailed financial indicator calculation prompt, and only process indicator type tasks")
    parser.add_argument("--task-types", type=str, default="fact,indicator",
                      help="the task types to process, separated by commas, default is 'fact,indicator'")
    
    parser.add_argument("--api_key", default=None, help="API key")
    parser.add_argument("--base_url", default=None, help="API base URL")
    parser.add_argument("--input", default=INPUT_FILE_PATH, help="input data file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="number of parallel workers")
    parser.add_argument("--log_level", default=LOGGING_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="log level")
    parser.add_argument("--use_model_for_pdf", action="store_true", default=USE_MODEL_FOR_PDF, help="whether to use model to process PDF")
    parser.add_argument("--pdf_extractor", choices=["pdfplumber", "pdfminer", "pypdf", "pymupdf", "pdftotext", "mineru"], 
                       default="pymupdf", help="when using textpdf mode, choose which extractor")
    
    return parser.parse_args()

def main():
    start_time = time.time()
    
    args = parse_args()
    
    setup_logging(level=args.log_level)
    
    interface_type = get_interface_type(args.model)
    model_params = get_model_params(args.model)

    
    if args.api_key is None:
        args.api_key = get_api_key(args.model)
        
    if args.base_url is None:
        args.base_url = get_base_url(args.model)
    
    use_pdf = (args.mode.lower() == "pdf")
    use_textpdf = (args.mode.lower() == "textpdf")
    
    if args.save_path:
        output_dir = args.save_path
    elif args.mode == "textpdf":
        if args.more_prompt:
            model_short_name = args.model.split('/')[-1] if '/' in args.model else args.model
            output_dir = Path(f"./experiments_textpdf/with_more_prompt/{model_short_name}")
        else:
            model_short_name = args.model.split('/')[-1] if '/' in args.model else args.model
            output_dir = Path(f"./experiments_textpdf/without_more_prompt/{model_short_name}")
    else:
        if args.more_prompt:
            model_short_name = args.model.split('/')[-1] if '/' in args.model else args.model
            output_dir = os.path.join("./experiments_text", "with_more_prompt", model_short_name)
        else:
            model_short_name = args.model.split('/')[-1] if '/' in args.model else args.model
            output_dir = os.path.join("./experiments_text", "without_more_prompt", model_short_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if interface_type == INTERFACE_TYPE_VOLCENGINE:
        model_name = get_model_id(args.model)
    else:
        model_name = args.model
    
    logging.info("\n=============== Table Evaluation ===============")
    logging.info(f"model: {args.model}")
    logging.info(f"actual used model ID: {model_name}")
    
    if interface_type == INTERFACE_TYPE_OPENAI:
        logging.info(f"using OpenAI compatible API")
        logging.info(f"API URL: {args.base_url}")
    elif interface_type == INTERFACE_TYPE_VOLCENGINE:
        logging.info(f"using Volcengine API")
        logging.info(f"Model ID: {get_model_id(args.model)}")
    
    logging.info(f"model parameters: temperature={model_params.get('temperature', 0.0)}, top_p={model_params.get('top_p', 0.9)}, max_tokens={model_params.get('max_tokens', 1024)}")
    logging.info(f"data mode: {'PDF' if use_pdf else ('extracted text from PDF' if use_textpdf else 'text')}")
    if use_textpdf:
        logging.info(f"PDF text extractor: {args.pdf_extractor}")
    if args.more_prompt:
        logging.info(f"using detailed financial indicator calculation prompt, only process indicator type tasks")
        logging.info(f"note: in this mode, only the tasks with actual API call failure will be recorded in the failure task record")
    logging.info(f"input file: {args.input}")
    logging.info(f"output directory: {output_dir}")
    logging.info(f"batch size: {args.batch_size}")
    logging.info(f"number of parallel workers: {args.workers}")
    logging.info(f"log level: {args.log_level}")
    logging.info("=======================================\n")
    
    try:
        results_file = process_table_metrics(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=model_name,
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
        
    except Exception as e:
        logging.error(f"\nerror processing table metrics: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()