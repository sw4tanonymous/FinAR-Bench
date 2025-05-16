import sys
import os
import json
import logging
from pathlib import Path
import re
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import traceback
import base64
import time
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from config import TASK_NUM
from deplot import metrics

try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    logging.warning("Volcengine SDK not installed, API will not be available")
    class Ark:
        def __init__(self, *args, **kwargs):
            raise ImportError("Volcengine Ark SDK not installed")

from model_configs import (
    init_api_client as model_init_api_client,
    get_interface_type, 
    get_model_params,
    get_model_id,
    INTERFACE_TYPE_VOLCENGINE,
    INTERFACE_TYPE_OPENAI
)


def setup_logging(log_dir="logs", level=None):
    """Configure logging"""
    log_level = getattr(logging, level or config.LOGGING_LEVEL) if hasattr(config, 'LOGGING_LEVEL') else logging.INFO
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'table_metrics_{timestamp}.log'

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding='utf-8')
        ],
        force=True
    )
    logging.info("Logging system initialized successfully, level: %s", logging.getLevelName(log_level))


def init_api_client(api_key=None, base_url=None, use_volcengine=False, use_nvidia=False, model_name=None):

    try:
        if model_name:
            return model_init_api_client(model_name, custom_api_key=api_key, custom_base_url=base_url)
        
        if use_volcengine:
            logging.info(f"Initializing Volcengine Ark SDK client")
            # Set environment variables (recommended by Volcengine)
            os.environ["ARK_API_KEY"] = api_key
            # Initialize Volcengine client with longer timeout
            return Ark(
                api_key=api_key, 
                timeout=1800,  # Set 30-minute timeout
            )
        elif use_nvidia:
            logging.info(f"Initializing NVIDIA API client")
            # Use OpenAI client to access NVIDIA API
            return OpenAI(base_url=base_url, api_key=api_key)
        else:
            # OpenAI API
            logging.info(f"Initializing OpenAI API client (URL: {base_url})")
            return OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize API client: {str(e)}")
        raise


def encode_pdf_to_base64(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"PDF encoding failed: {str(e)}")
        return None


def extract_table_from_pdf_with_model(client, pdf_path, model_name):
    logging.info(f"using model to directly read PDF file: {pdf_path}")
    
    # check if the file exists
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file does not exist: {pdf_path}")
        return None
    
    try:
        # encode the PDF file
        pdf_base64 = encode_pdf_to_base64(pdf_path)
        if not pdf_base64:
            return None
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "请从这份PDF文件中提取所有财务表格数据（利润表、资产负债表、现金流量表等），保持原始格式输出。将每个表格用Markdown格式呈现，并在表格前添加表格名称作为标题（# 表格名称）。"},
                    {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4000,
            temperature=0.0
        )
        
        table_text = response.choices[0].message.content.strip()
        logging.info(f"model successfully extracted table data, content length: {len(table_text)}")
        
        return table_text
    
    except Exception as e:
        logging.error(f"failed to use model to read PDF: {str(e)}")
        logging.error(traceback.format_exc())
        return None


def check_txt_file_exists(company_code, pdf_extractor="pdfplumber"):
    # remove the possible stock suffix (.SH, .SZ, etc.)
    company_code = company_code.split('.')[0] if '.' in company_code else company_code
    txt_file = f"{company_code}.txt"
    
    txt_path = os.path.join("pdf_extractor_result", "txt_output", pdf_extractor, txt_file)
    
    return os.path.exists(txt_path)


def load_pdf_text_from_file(company_code, pdf_extractor="pdfplumber"):
    company_code = company_code.split('.')[0] if '.' in company_code else company_code
    txt_file = f"{company_code}.txt"
    
    txt_path = os.path.join("pdf_extractor_result", "txt_output", pdf_extractor, txt_file)
    logging.info(f"🔍 trying to read PDF text from file: {txt_path}")
    
    if not os.path.exists(txt_path):
        logging.error(f"text file not found: {txt_path}")
        return None
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        logging.info(f"successfully read PDF text from file: {txt_path}, content length: {len(text_content)}")
        return text_content
    except Exception as e:
        logging.error(f"failed to read PDF text file: {str(e)}")
        return None


def load_task_metrics(file_path, use_textpdf=False, pdf_extractor="pdfplumber", task_types=None):

    try:
        allowed_task_types = None
        if task_types:
            if isinstance(task_types, str):
                allowed_task_types = [t.strip() for t in task_types.split(',')]
            elif isinstance(task_types, list):
                allowed_task_types = task_types
            logging.info(f"only process the following task types: {', '.join(allowed_task_types)}")
        
        allowed_task_nums = TASK_NUM
        if allowed_task_nums and len(allowed_task_nums) > 0:
            logging.info(f"only process the following task numbers: {', '.join(allowed_task_nums)}")
        else:
            logging.info("no task number filter, will process all tasks")
            allowed_task_nums = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = []
            skipped_count = 0
            filtered_by_type_count = 0
            filtered_by_num_count = 0
            filtered_by_range_count = 0
            start_idx = config.PROCESS_DATA_RANGE["start"] - 1 
            end_idx = config.PROCESS_DATA_RANGE["end"] 
            step = config.PROCESS_DATA_RANGE["step"]
            
            for line_num, line in enumerate(f, 1):
                if (line_num - 1) < start_idx:
                    filtered_by_range_count += 1
                    continue
                    
                if end_idx is not None and line_num > end_idx:
                    filtered_by_range_count += 1
                    break
                    
                if (line_num - start_idx - 1) % step != 0:
                    filtered_by_range_count += 1
                    continue
                
                try:
                    if line.strip():
                        data = json.loads(line.strip())
                        
                        # check if there is an instances field
                        if "instances" not in data:
                            logging.warning(f"skipping line {line_num}: missing instances field")
                            skipped_count += 1
                            continue
                            
                        valid_instances = []
                        for instance in data.get("instances", []):
                            if not all(key in instance for key in ["task_id", "task", "ground_truth", "task_type", "task_num", "company", "company_code"]):
                                logging.warning(f"skipping instance {instance.get('task_id', 'unknown')}: missing necessary fields")
                                skipped_count += 1
                                continue
                            
                            if allowed_task_types and instance.get("task_type") not in allowed_task_types:
                                filtered_by_type_count += 1
                                continue
                            
                            if allowed_task_nums and str(instance.get("task_num")) not in allowed_task_nums:
                                filtered_by_num_count += 1
                                continue
                                
                            # if use_textpdf mode, check if the text file exists
                            if use_textpdf:
                                company_code = instance.get("company_code")
                                if not check_txt_file_exists(company_code, pdf_extractor):
                                    logging.warning(f"skipping instance {instance.get('task_id')}: text file not found")
                                    skipped_count += 1
                                    continue
                                    
                            valid_instances.append(instance)
                            
                        if valid_instances:
                            new_data = data.copy()
                            new_data["instances"] = valid_instances
                            data_list.append(new_data)
                            
                except json.JSONDecodeError as e:
                    logging.warning(f"skipping line {line_num}: JSON parsing error - {str(e)}")
                    skipped_count += 1
                    continue
        
        log_message = f"successfully loaded {len(data_list)} task data, skipped {skipped_count} invalid data"
        if allowed_task_types:
            log_message += f"，filtered {filtered_by_type_count} tasks due to task type mismatch"
        if allowed_task_nums:
            log_message += f"，filtered {filtered_by_num_count} tasks due to task number mismatch"
        logging.info(log_message)
        
        logging.info(f"filtered {filtered_by_range_count} lines due to data range")
        logging.info(f"data range: from {config.PROCESS_DATA_RANGE['start']} to {config.PROCESS_DATA_RANGE['end'] or 'end'}, step {config.PROCESS_DATA_RANGE['step']}")
        return data_list
    except Exception as e:
        logging.error(f"failed to load task metrics data file: {str(e)}")
        raise


def extract_tasks_and_tables(data, client, model_name, use_pdf=False, use_textpdf=False, pdf_extractor="pdfplumber", use_model_for_pdf=False):
    instances = data.get("instances", [])
    
    if use_textpdf and instances:
        # get company_code from the first instance
        company_code = instances[0].get("company_code")
        if not company_code:
            logging.warning("Cannot get company_code from instance, will use table data from JSON")
            table = data.get("table", "")
            return table, instances
            
        logging.info(f"Attempting to load table from extracted text, company code: {company_code}")
        
        # load table from text file
        table = load_pdf_text_from_file(company_code, pdf_extractor)
        
        if not table:
            # if extraction from text file fails, return None to skip this task
            logging.warning(f"failed to load table from extracted text, company code: {company_code}, will skip this task")
            return None, None  # return None, None to indicate this task should be skipped
            
        logging.info(f"successfully loaded table from extracted text, company code: {company_code}")
    elif use_pdf and "file_path" in data:
        # if specified to use PDF, extract table from PDF file
        pdf_path = data.get("file_path")
        logging.info(f"attempting to load table from PDF file: {pdf_path}")
        
        if use_model_for_pdf and client:
            # Use multimodal model to directly read PDF
            table = extract_table_from_pdf_with_model(client, pdf_path, model_name)
        else:
            print("failed to use mode pdf")
            
        if not table:
            logging.warning(f"Failed to extract table from PDF, using table data from text as fallback")
            table = data.get("table", "")
    else:
        table = data.get("table", "")
        logging.info("Using table data from text")
    
    return table, instances


def analyze_table_with_task(client, table, task, model_name, max_retries=None, use_textpdf=False, more_prompt=False, conditions=None, task_type=None):
    """
    Analyze table data with specified task
    Args:
        client: API client instance
        table: table data
        task: task description
        model_name: model name
        max_retries: maximum number of retries
        use_textpdf: whether to use PDF text mode
        more_prompt: whether to use more detailed prompt
        conditions: conditions constraints
        task_type: task type
        
    Returns:
        Tuple[str, Dict]: (API response content, token usage statistics)
    """
    from model_configs import get_model_params, get_interface_type, get_model_id, INTERFACE_TYPE_VOLCENGINE
    
    interface_type = get_interface_type(model_name)
    
    model_params = get_model_params(model_name)
    
    actual_model_name = model_name
    if interface_type == INTERFACE_TYPE_VOLCENGINE:
        actual_model_name = get_model_id(model_name)
    
    retries = max_retries or getattr(config, 'MAX_RETRIES', 3)
    
    token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }
    
    logging.info(f"table data length: {len(table) if table else 0}")
    
    for attempt in range(retries):
        try:
            # 使用配置中的prompt模板
            if use_textpdf:
                if more_prompt:
                    # more_prompt task in textpdf mode
                    prompt = f"""请{task}，

财务指标的计算方式如下：
净资产收益率=归属于母公司净利润/((期初归属于母公司所有者权益+期末归属于母公司所有者权益)/2)
总资产收益率=净利润/((期初总资产+期末总资产)/2)
销售毛利率=(营业收入-营业成本)/营业收入
销售净利率=净利润/营业收入
资产负债率=总负债/总资产
流动比率=流动资产/流动负债
速动比率=(流动资产-存货-预付帐款-一年内到期的非流动资产-其他流动资产)/流动负债
期间费用率=毛利率-净利率
权益乘数=总资产/所有者权益
产权比率=总负债/归属于母公司所有者权益合计
存货周转天数=360/存货周转率
应收账款周转天数=360/应收账款周转率
应付账款周转天数=360/应付账款周转率
营业周期=存货周转天数+应收账款周转天数
总资产周转率=营业收入/((期初总资产+期末总资产)/2)
存货周转率=营业成本/((期初存货+期末存货)/2)
应收账款周转率=营业收入/((期初应收账款+期末应收账款)/2)
应付账款周转率=(期末存货+营业成本-期初存货)/((期初应付账款+期末应付账款)/2)
流动资产周转率=营业收入/((期初流动资产+期末流动资产)/2)
固定资产周转率=营业收入/((期初固定资产+期末固定资产)/2)

以下是财务报表数据：
{table}

，输出的数字不要带有%和其他单位，请输出markdown"""
                    logging.info(f"using textpdf more prompt mode")
                else:
                    # Regular reasoning task in text mode
                    prompt = f"请{task}，\n\n以下是财务报表数据：\n\n{table}\n\n ，输出的数字不要带有%和其他单位，请输出markdown"
                    logging.info(f"using textpdf mode")
            else:
                if more_prompt:
                    prompt = f"""请{task}，

财务指标的计算方式如下：
净资产收益率=归属于母公司净利润/((期初归属于母公司所有者权益+期末归属于母公司所有者权益)/2)
总资产收益率=净利润/((期初总资产+期末总资产)/2)
销售毛利率=(营业收入-营业成本)/营业收入
销售净利率=净利润/营业收入
资产负债率=总负债/总资产
流动比率=流动资产/流动负债
速动比率=(流动资产-存货-预付帐款-一年内到期的非流动资产-其他流动资产)/流动负债
期间费用率=毛利率-净利率
权益乘数=总资产/所有者权益
产权比率=总负债/归属于母公司所有者权益合计
存货周转天数=360/存货周转率
应收账款周转天数=360/应收账款周转率
应付账款周转天数=360/应付账款周转率
营业周期=存货周转天数+应收账款周转天数
总资产周转率=营业收入/((期初总资产+期末总资产)/2)
存货周转率=营业成本/((期初存货+期末存货)/2)
应收账款周转率=营业收入/((期初应收账款+期末应收账款)/2)
应付账款周转率=(期末存货+营业成本-期初存货)/((期初应付账款+期末应付账款)/2)
流动资产周转率=营业收入/((期初流动资产+期末流动资产)/2)
固定资产周转率=营业收入/((期初固定资产+期末固定资产)/2)

以下是财务表格数据，单位为元:

{table}

，输出的数字不要带有%和其他单位，请输出markdown"""
                    logging.info(f"using more prompt mode")
                else:
                    prompt = f"请{task}，\n\n以下是财务表格数据，单位为元:\n\n{table}\n\n ，输出的数字不要带有%和其他单位，请输出markdown"
                    logging.info(f"using standard mode")
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            if interface_type == INTERFACE_TYPE_VOLCENGINE:
                response = client.chat.completions.create(
                    model=actual_model_name,
                    messages=messages,
                    temperature=model_params.get('temperature', 0.6),
                    top_p=model_params.get('top_p', 0.7),
                    max_tokens=model_params.get('max_tokens', 1500)
                )
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    logging.debug(f"deep thinking content: {response.choices[0].message.reasoning_content}")
                result = response.choices[0].message.content.strip()
            else:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=model_params.get('temperature', 0.6),
                        top_p=model_params.get('top_p', 0.7),
                        max_tokens=model_params.get('max_tokens', 1500)
                    )
                    
                    result = response.choices[0].message.content.strip()
                except Exception as api_error:
                    if hasattr(api_error, 'status_code') and api_error.status_code == 429:
                        wait_time = 5 + attempt * 10  # increase wait time gradually
                        logging.warning(f"API rate limit triggered (429)! Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
            
            if hasattr(response, 'usage') and response.usage is not None:
                token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                    'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                    'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                }
                logging.info(f"API returned token usage: {token_usage}")
            
            logging.info(f"Token usage: input={token_usage.get('prompt_tokens', 0)}, output={token_usage.get('completion_tokens', 0)}, total={token_usage.get('total_tokens', 0)}")
            
            logging.info(f"API call successful, response length: {len(result)}")
            return result, token_usage
            
        except Exception as e:
            logging.warning(f"API call failed: {str(e)}, retrying... ({attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            continue
    
    logging.error(f"API call failed after {retries} retries")
    return None, token_usage


def extract_markdown_from_response(response):
    """Extract markdown data from API response"""

    markdown_pattern = r"```(?:markdown)?\s*([\s\S]*?)```"
    match = re.search(markdown_pattern, response, re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        split_content = content.split('\n\n', 1)
        return split_content[0].strip() if len(split_content) > 1 else content
    
    return response.strip()


def normalize_markdown(markdown_text):
    """Convert Markdown text to DePlot evaluation supported format
    
    Args:
        markdown_text (str): Markdown text output by model
        
    Returns:
        str: Normalized markdown text
    """
    if not markdown_text:
        return ""
        
    # Remove extra empty lines
    lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
    
    # Process header separators
    normalized_lines = []
    for i, line in enumerate(lines):
        # Skip header separator lines
        if i >  0 and all(c in ['-', '|'] for c in line.replace(' ', '')):
            continue
        # Process regular lines
        if '|' in line:
            # Clean extra vertical bars
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if parts:
                # Process numbers in each cell (remove commas)
                processed_parts = []
                for j, part in enumerate(parts):
                    # If it's a header row (first row) or header column (first column), keep as is
                    if i == 0 or j == 0:
                        processed_parts.append(part)
                    else:
                        # Try to convert cell content to number (remove commas)
                        try:
                            # If it's a number (may contain commas), remove commas
                            if any(c.isdigit() for c in part):
                                # Remove all commas
                                processed_part = part.replace(',', '')
                                # If it's in scientific notation, keep the format
                                if 'e' in processed_part.lower():
                                    processed_parts.append(processed_part)
                                else:
                                    # Try to convert to number and back to string to ensure correct format
                                    try:
                                        num = float(processed_part)
                                        processed_parts.append(str(num))
                                    except ValueError:
                                        processed_parts.append(processed_part)
                            else:
                                processed_parts.append(part)
                        except:
                            processed_parts.append(part)
                # Remove vertical bars at beginning and end
                normalized_lines.append(' | '.join(processed_parts))
        else:
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)

def evaluate_prediction(prediction, ground_truth):
    """Evaluate differences between prediction and ground truth"""
    try:
        # Clean and normalize prediction and ground truth
        gt_markdown = normalize_markdown(ground_truth)
        pred_markdown = normalize_markdown(prediction)

        # Print normalized data
        logging.info(f"Normalized ground truth: {gt_markdown}")
        logging.info(f"Normalized prediction: {pred_markdown}")

        # Use normalized data to construct DePlot evaluation format
        targets = [[gt_markdown]]       # list[list[str]] each outer layer represents a sample, inner layer is multiple candidate answers (usually 1)
        predictions = [pred_markdown]   # list[str] corresponding one-to-one with targets, each is model output

        # Calculate various evaluation metrics
        result = {}
        result = metrics.table_datapoints_precision_recall(targets, predictions)
        result.update(metrics.table_number_accuracy_extext(targets, predictions))
        
        # Log evaluation results
        logging.info(f"Evaluation results: "
                    f"Precision={result['table_datapoints_precision_onlyvalue']:.2f}, "
                    f"Recall={result['table_datapoints_recall_onlyvalue']:.2f}, "
                    f"F1_score={result['table_datapoints_f1_onlyvalue']:.2f}, "
                    f"Numbers accuracy excluding headers={result['numbers_match_extext']:.2f}")
        
        return result
    
    except Exception as e:
        logging.error(f"Error evaluating prediction result: {str(e)}")
        logging.error(f"Prediction: {prediction}")
        logging.error(f"Ground truth: {ground_truth}")
        logging.error(traceback.format_exc())
        return {
            "table_datapoints_precision": 0,
            "table_datapoints_recall": 0,
            "table_datapoints_f1": 0,
            "numbers_match_extext": 0
        }


def process_single_task(task_data, table, client, model_name, use_textpdf=False, more_prompt=False):
    """Process a single table metrics task"""
    task_id = task_data.get("task_id", "")
    task_type = task_data.get("task_type", "")
    task_description = task_data.get("task", "")
    ground_truth = task_data.get("ground_truth", "")
    task_num = task_data.get("task_num", 0)  # Get task variable count, default is 0
    company_code = task_data.get("company_code", "")  # Get company_code, default is empty string
    
    # If it's a reasoning type task, get the conditions field
    conditions = None
    if task_type == "reasoning":
        conditions = task_data.get("conditions", "")
        if conditions:
            logging.info(f"Task {task_id} is reasoning type, contains conditions: {conditions}")
    
    # If using more_prompt mode, only process indicator type tasks
    if more_prompt and task_type != "indicator":
        logging.info(f"Skipping non-indicator type task: {task_id} ({task_type})")
        # Return None, None to indicate this is a skipped task, not a failed task
        return None, None
    
    logging.info(f"Processing task: {task_id} ({task_type}), task variable count: {task_num}")
    
    # Get API prediction result
    try:
        prediction, token_usage = analyze_table_with_task(
            client, 
            table, 
            task_description, 
            model_name, 
            use_textpdf=use_textpdf,
            more_prompt=more_prompt,
            conditions=conditions,
            task_type=task_type
        )
        
        if not prediction:
            logging.error(f"Task {task_id} failed to get prediction result")
            return None, {"task_id": task_id, "task_type": task_type, "task_num": task_num, "company_code": company_code, "reason": "No prediction result"}
            
        # Print complete model response
        logging.info(f"Complete model response: {prediction}")
        
        # Extract CSV data
        markdown_prediction = extract_markdown_from_response(prediction)
        
        # Print extracted markdown
        logging.info(f"Extracted markdown: {markdown_prediction}")
        
        # Convert to markdown format
        gt_markdown = normalize_markdown(ground_truth)
        pred_markdown = normalize_markdown(markdown_prediction)
        
        # Print normalized markdown
        logging.info(f"Normalized markdown: {pred_markdown}")
        
        # Evaluate results
        evaluation = evaluate_prediction(markdown_prediction, ground_truth)
        
        # Return results, including token usage information
        return {
            "task_id": task_id,
            "task_type": task_type,
            "task_num": task_num,  # Add task variable count information
            "task": task_description,
            "ground_truth": gt_markdown,
            "prediction": pred_markdown,
            "metrics": evaluation,
            "company_code": company_code,
            "token_usage": token_usage 
        }, None
    except Exception as e:
        # Record failed task
        error_message = f"Task processing failed: {str(e)}"
        logging.error(f"Task {task_id} {error_message}")
        logging.error(traceback.format_exc())
        return None, {"task_id": task_id, "task_type": task_type, "task_num": task_num, "company_code": company_code, "reason": error_message}


def process_tasks_parallel(task_instances, table, client, model_name, use_textpdf=False, more_prompt=False):
    """Process multiple table analysis tasks in parallel"""
    results = []
    failed_tasks = []
    all_predictions = []
    all_ground_truths = []
    
    # Initialize token usage statistics
    total_token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }
    
    # Use thread pool to process multiple tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # Submit tasks to thread pool
        future_to_task = {
            executor.submit(
                process_single_task, 
                task, 
                table, 
                client, 
                model_name, 
                use_textpdf,
                more_prompt
            ): task for task in task_instances
        }
        
        # Process completed tasks
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(task_instances), desc="Processing tasks"):
            task = future_to_task[future]
            try:
                result, failed_info = future.result()
                if result:
                    results.append(result)
                    # Ensure predictions and ground truths are added simultaneously
                    all_predictions.append(result["prediction"])
                    all_ground_truths.append(result["ground_truth"])
                    
                    # Accumulate token usage
                    if "token_usage" in result:
                        total_token_usage["prompt_tokens"] += result["token_usage"].get("prompt_tokens", 0)
                        total_token_usage["completion_tokens"] += result["token_usage"].get("completion_tokens", 0)
                        total_token_usage["total_tokens"] += result["token_usage"].get("total_tokens", 0)
                        
                        logging.debug(f"Task {result['task_id']} token usage: input={result['token_usage']['prompt_tokens']}, output={result['token_usage']['completion_tokens']}, total={result['token_usage']['total_tokens']}")
                    
                    logging.info(f"Task {result['task_id']} completed")
                # Only record as failed task when failed_info is not None and has actual content
                elif failed_info:
                    failed_tasks.append(failed_info)
                    logging.warning(f"Task {failed_info['task_id']} recorded as failed task")
            except Exception as e:
                task_id = task.get("task_id", "unknown")
                company_code = task.get("company_code", "")
                logging.error(f"Task {task_id} processing failed: {str(e)}")
                failed_tasks.append({
                    "task_id": task_id,
                    "task_type": task.get("task_type", "unknown"),
                    "reason": str(e),
                    "company_code": company_code
                })
    
    # Log total token usage for this batch
    logging.info(f"Batch total token usage: input={total_token_usage['prompt_tokens']}, output={total_token_usage['completion_tokens']}, total={total_token_usage['total_tokens']}")
    
    return results, failed_tasks, all_predictions, all_ground_truths, total_token_usage


def save_failed_tasks(failed_tasks, output_dir, model_name, batch_id=None, more_prompt=False):
    """Save records of failed tasks"""
    if not failed_tasks:
        logging.info("No failed tasks to record")
        return
    
    # Filter for indicator type failed tasks
    indicator_failed_tasks = []
    for task in failed_tasks:
        if task.get("task_type") == "indicator":
            indicator_failed_tasks.append(task)
    
    if not indicator_failed_tasks:
        logging.info("No indicator type failed tasks to record")
        return
    
    # Create failed directory
    failed_dir = Path(output_dir) / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%m%d%H%M')
    batch_suffix = f"_batch{batch_id}" if batch_id is not None else ""
    model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
    failed_file = failed_dir / f"failed_tasks_{model_short_name}{batch_suffix}_{timestamp}.json"
    
    # Save failed records
    with open(failed_file, 'w', encoding='utf-8') as f:
        json.dump(indicator_failed_tasks, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Saved {len(indicator_failed_tasks)} indicator type actual failed task records to: {failed_file} (not including skipped tasks)")
    return str(failed_file)


def batch_process_tasks(data, api_key, base_url, model_name, output_dir, batch_id=None, use_pdf=False, use_textpdf=False, pdf_extractor="pdfplumber", use_model_for_pdf=False, more_prompt=False):
    """Batch process a group of table metrics tasks"""
    client = init_api_client(api_key=api_key, base_url=base_url, use_volcengine=False, 
                            use_nvidia=False, model_name=model_name)
    
    # Extract tables and tasks
    table, instances = extract_tasks_and_tables(
        data, 
        client, 
        model_name, 
        use_pdf=use_pdf, 
        use_textpdf=use_textpdf,
        pdf_extractor=pdf_extractor,
        use_model_for_pdf=use_model_for_pdf
    )
    
    # If in textpdf mode and PDF text not found, skip this task
    if use_textpdf and table is None:
        company = data.get("instances", [])[0].get("company", "Unknown") if data.get("instances", []) else "Unknown"
        company_code = data.get("instances", [])[0].get("company_code", "Unknown") if data.get("instances", []) else "Unknown"
        logging.info(f"Skipping company {company} ({company_code}) because corresponding PDF text file could not be found")
        return {
            "results": [],
            "overall_metrics": {"all": {"count": 0}},
            "failed_tasks_count": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "skipped": True,
            "skipped_reason": f"Cannot find PDF text file for company {company_code}",
            "failed_tasks": [{
                "task_id": instance.get("task_id", "unknown"),
                "task_type": instance.get("task_type", "unknown"),
                "company_code": instance.get("company_code", company_code),
                "reason": f"Cannot find PDF text file for company {company_code}"
            } for instance in instances]
        }
    
    # Process tasks in parallel (Volcengine API may have concurrency limits, respect the setting in config)
    max_workers = config.MAX_WORKERS
    results, failed_tasks, all_predictions, all_ground_truths, total_token_usage = process_tasks_parallel(
        instances, table, client, model_name, use_textpdf, more_prompt
    )
    
    # Save failed tasks
    if failed_tasks:
        save_failed_tasks(failed_tasks, output_dir, model_name, batch_id, more_prompt)
    
    # Calculate overall metrics (using all predictions and ground truths)
    # Construct DePlot evaluation format
    all_targets = [[gt] for gt in all_ground_truths]
    all_preds = all_predictions  # list[str]
    
    # Check if there is valid target data
    if not all_targets:
        logging.warning("No valid target data, skipping metrics calculation")
        overall_metrics = {
            "all": {
                "table_datapoints_precision_onlyvalue": 0.0,
                "table_datapoints_recall_onlyvalue": 0.0,
                "table_datapoints_f1_onlyvalue": 0.0,
                "numbers_match_extext": 0.0,
                "count": 0
            }
        }
    else:
        try:
            overall_metrics = {
                "all": metrics.table_datapoints_precision_recall(all_targets, all_preds)
            }
            overall_metrics["all"].update(metrics.table_number_accuracy_extext(all_targets, all_preds))
            overall_metrics["all"]["count"] = len(results)
        except ZeroDivisionError:
            logging.warning("Zero division error when calculating evaluation metrics, using default values")
            overall_metrics = {
                "all": {
                    "table_datapoints_precision_onlyvalue": 0.0,
                    "table_datapoints_recall_onlyvalue": 0.0,
                    "table_datapoints_f1_onlyvalue": 0.0,
                    "numbers_match_extext": 0.0,
                    "count": 0
                }
            }
    
    # Calculate metrics by task type
    task_types = {}
    for result in results:
        task_type = result.get("task_type", "other")
        if task_type not in task_types:
            task_types[task_type] = {"predictions": [], "ground_truths": []}
        # Convert to markdown format
        pred_markdown = result['prediction']
        gt_markdown = result['ground_truth']
        task_types[task_type]["predictions"].append(pred_markdown)
        task_types[task_type]["ground_truths"].append(gt_markdown)
    
    for task_type, data in task_types.items():
        # Construct DePlot evaluation format
        targets = [[gt] for gt in data["ground_truths"]]  # list[list[str]]
        predictions = data["predictions"]  # list[str]
        
        overall_metrics[task_type] = metrics.table_datapoints_precision_recall(targets, predictions)
        overall_metrics[task_type].update(metrics.table_number_accuracy_extext(targets, predictions))
        overall_metrics[task_type]["count"] = len(data["predictions"])
        
        # Log evaluation results for each task type
        logging.info(f"\n Evaluation metrics for task type '{task_type}':")
        logging.info(f"   Precision: {overall_metrics[task_type]['table_datapoints_precision_onlyvalue']:.2f}")
        logging.info(f"   Recall: {overall_metrics[task_type]['table_datapoints_recall_onlyvalue']:.2f}")
        logging.info(f"   F1_score: {overall_metrics[task_type]['table_datapoints_f1_onlyvalue']:.2f}")
        logging.info(f"   Numbers accuracy excluding headers: {overall_metrics[task_type]['numbers_match_extext']:.2f}")
        logging.info(f"   Task count: {overall_metrics[task_type]['count']}")
    
    # Log total token usage
    logging.info(f"\n Token usage statistics:")
    logging.info(f"   Input tokens: {total_token_usage['prompt_tokens']}")
    logging.info(f"   Output tokens: {total_token_usage['completion_tokens']}")
    logging.info(f"   Total tokens: {total_token_usage['total_tokens']}")
    
    batch_result = {
        "results": results,
        "overall_metrics": overall_metrics,
        "failed_tasks_count": len(failed_tasks),
        "token_usage": total_token_usage
    }
    
    if len(failed_tasks) > 0:
        logging.info(f"This batch has {len(failed_tasks)} actual failed tasks (not including skipped tasks)")
    
    return batch_result

def merge_batch_results(all_batch_results):
    """Merge results from multiple batches"""
    all_results = []
    all_predictions = []
    all_ground_truths = []
    task_types = {}
    task_nums = {}  # Group by task variable count
    failed_tasks_count = 0
    
    # Initialize total token usage statistics
    total_token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }
    
    # Merge all results
    for batch_result in all_batch_results:
        all_results.extend(batch_result["results"])
        failed_tasks_count += batch_result.get("failed_tasks_count", 0)
        
        # Accumulate token usage
        if "token_usage" in batch_result:
            total_token_usage["prompt_tokens"] += batch_result["token_usage"].get("prompt_tokens", 0)
            total_token_usage["completion_tokens"] += batch_result["token_usage"].get("completion_tokens", 0)
            total_token_usage["total_tokens"] += batch_result["token_usage"].get("total_tokens", 0)
        
        # Collect predictions and ground truths by task type and task variable count
        for result in batch_result["results"]:
            task_type = result.get("task_type", "other")
            task_num = result.get("task_num", 0)  # Get task variable count, default is 0
            
            # Group by task type
            if task_type not in task_types:
                task_types[task_type] = {"predictions": [], "ground_truths": []}
            
            task_types[task_type]["predictions"].append(result["prediction"])
            task_types[task_type]["ground_truths"].append(result["ground_truth"])
            
            # Group by task variable count
            task_num_key = f"task_num_{task_num}"
            if task_num_key not in task_nums:
                task_nums[task_num_key] = {"predictions": [], "ground_truths": []}
            
            task_nums[task_num_key]["predictions"].append(result["prediction"])
            task_nums[task_num_key]["ground_truths"].append(result["ground_truth"])
            
            all_predictions.append(result["prediction"])
            all_ground_truths.append(result["ground_truth"])
    
    # Calculate overall metrics
    all_targets = [[gt] for gt in all_ground_truths]
    overall_metrics = {
        "all": metrics.table_datapoints_precision_recall(all_targets, all_predictions)
    }
    overall_metrics["all"].update(metrics.table_number_accuracy_extext(all_targets, all_predictions))
    overall_metrics["all"]["count"] = len(all_results)
    
    # Calculate metrics for each task type
    for task_type, data in task_types.items():
        targets = [[gt] for gt in data["ground_truths"]]
        predictions = data["predictions"]
        
        overall_metrics[task_type] = metrics.table_datapoints_precision_recall(targets, predictions)
        overall_metrics[task_type].update(metrics.table_number_accuracy_extext(targets, predictions))
        overall_metrics[task_type]["count"] = len(data["predictions"])
    
    # Calculate metrics for each task variable count
    for task_num_key, data in task_nums.items():
        targets = [[gt] for gt in data["ground_truths"]]
        predictions = data["predictions"]
        
        overall_metrics[task_num_key] = metrics.table_datapoints_precision_recall(targets, predictions)
        overall_metrics[task_num_key].update(metrics.table_number_accuracy_extext(targets, predictions))
        overall_metrics[task_num_key]["count"] = len(predictions)
    
    # Log overall evaluation results
    logging.info("\n\n======= Overall Evaluation Results =======")
    logging.info(f"Total task count: {len(all_results)}")
    if failed_tasks_count > 0:
        logging.info(f"Failed task count: {failed_tasks_count}")
    logging.info(f"Overall Precision: {overall_metrics['all']['table_datapoints_precision_onlyvalue']:.2f}")
    logging.info(f"Overall Recall: {overall_metrics['all']['table_datapoints_recall_onlyvalue']:.2f}")
    logging.info(f"Overall F1_score: {overall_metrics['all']['table_datapoints_f1_onlyvalue']:.2f}")
    logging.info(f"Overall numbers accuracy excluding headers: {overall_metrics['all']['numbers_match_extext']:.2f}")
    
    # Log token usage
    logging.info(f"\n Total token usage statistics:")
    logging.info(f"   Input tokens: {total_token_usage['prompt_tokens']}")
    logging.info(f"   Output tokens: {total_token_usage['completion_tokens']}")
    logging.info(f"   Total tokens: {total_token_usage['total_tokens']}")
    
    # Log overall evaluation results for each task type
    for task_type, metrics_data in overall_metrics.items():
        if task_type != "all" and not task_type.startswith("task_num_"):
            logging.info(f"\n Overall evaluation metrics for task type '{task_type}':")
            logging.info(f"   Precision: {metrics_data['table_datapoints_precision_onlyvalue']:.2f}")
            logging.info(f"   Recall: {metrics_data['table_datapoints_recall_onlyvalue']:.2f}")
            logging.info(f"   F1_score: {metrics_data['table_datapoints_f1_onlyvalue']:.2f}")
            logging.info(f"   Numbers accuracy excluding headers: {metrics_data['numbers_match_extext']:.2f}")
            logging.info(f"   Task count: {metrics_data['count']}")
    
    # Log overall evaluation results for each task variable count
    logging.info("\n\n======= Evaluation Results by Task Variable Count =======")
    for task_num_key, metrics_data in sorted([(k, v) for k, v in overall_metrics.items() if k.startswith("task_num_")]):
        task_num = task_num_key.replace("task_num_", "")
        logging.info(f"\n Overall evaluation metrics for task variable count '{task_num}':")
        logging.info(f"   Precision: {metrics_data['table_datapoints_precision_onlyvalue']:.2f}")
        logging.info(f"   Recall: {metrics_data['table_datapoints_recall_onlyvalue']:.2f}")
        logging.info(f"   F1_score: {metrics_data['table_datapoints_f1_onlyvalue']:.2f}")
        logging.info(f"   Numbers accuracy excluding headers: {metrics_data['numbers_match_extext']:.2f}")
        logging.info(f"   Task count: {metrics_data['count']}")
    
    return {
        "results": all_results,
        "overall_metrics": overall_metrics,
        "failed_tasks_count": failed_tasks_count,
        "token_usage": total_token_usage
    }


def process_table_metrics(api_key, base_url, model_name, input_file, output_dir, batch_size=3, use_pdf=False, use_textpdf=False, pdf_extractor="pdfplumber", use_model_for_pdf=False, more_prompt=False, task_types=None):
    """Main function for processing table metrics evaluation"""
    try:
        setup_logging()
        start_time = time.time()
        
        # 根据model_name确定接口类型
        interface_type = get_interface_type(model_name)
        model_params = get_model_params(model_name)
        
        # 根据接口类型设置标志（为了向后兼容）
        use_volcengine = (interface_type == INTERFACE_TYPE_VOLCENGINE)
        use_nvidia = (interface_type == INTERFACE_TYPE_OPENAI and model_params.get("provider") == "nvidia")
        
        # Load task data, pass use_textpdf and pdf_extractor parameters, as well as task_types
        data_list = load_task_metrics(input_file, use_textpdf, pdf_extractor, task_types)
        
        if not data_list:
            logging.error("No valid task data found")
            return None
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log API type being used with more detailed information
        if interface_type == INTERFACE_TYPE_VOLCENGINE:
            logging.info(f"Using Volcengine API: {model_name}")
            logging.info(f"  - Model ID: {model_params.get('model_id', 'unknown')}")
            logging.info(f"  - Provider: {model_params.get('provider', 'volcengine')}")
        elif interface_type == INTERFACE_TYPE_OPENAI:
            provider = model_params.get('provider', 'unknown')
            base_url_info = model_params.get('base_url') or base_url or 'Default'
            logging.info(f"Using OpenAI compatible API: {model_name}")
            logging.info(f"  - Provider: {provider}")
            logging.info(f"  - Base URL: {base_url_info}")
        else:
            logging.info(f"Using custom API interface: {model_name}")
            
        # Get maximum batch count, default to value in config or use environment variable
        max_workers = config.MAX_WORKERS
        
        # Process all data in batches
        all_batch_results = []
        
        for batch_idx, batch_start in enumerate(range(0, len(data_list), batch_size), 1):
            batch = data_list[batch_start:batch_start + batch_size]
        
            logging.info(f"\nProcessing batch {batch_idx}/{(len(data_list) + batch_size - 1) // batch_size}, {len(batch)} items")
        
            # Batch processing
            batch_results = []
            for i, d in enumerate(batch, 1):
                # Display progress information
                company = d.get("instances", [])[0].get("company", "Unknown") if d.get("instances", []) else "Unknown"
                company_code = d.get("instances", [])[0].get("company_code", "Unknown") if d.get("instances", []) else "Unknown"
                tasks_count = len(d.get("instances", []))
                
                logging.info(f"  Processing: [{i}/{len(batch)}] {company} ({company_code}), {tasks_count} tasks")
                
                # Process single data item
                result = batch_process_tasks(
                    d, 
                    api_key, 
                    base_url, 
                    model_name, 
                    output_dir,
                    batch_id=batch_idx,
                    use_pdf=use_pdf,
                    use_textpdf=use_textpdf,
                    pdf_extractor=pdf_extractor,
                    use_model_for_pdf=use_model_for_pdf,
                    more_prompt=more_prompt
                )
                
                batch_results.append(result)
        
            if batch_results:
                # Merge results for current batch
                batch_result = merge_batch_results(batch_results)
                all_batch_results.append(batch_result)
                
                # Save batch results using simplified timestamp format
                timestamp = datetime.now().strftime('%m%d%H%M')
                batch_file = output_dir / f"batch_{batch_idx}_{timestamp}.json"
                
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_result, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Batch {batch_idx} processing complete, results saved to: {batch_file}")
                
                # Temporary summary, generate overall results for batches processed so far
                current_results = merge_batch_results(all_batch_results)
                timestamp = datetime.now().strftime('%m%d%H%M')
                model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                current_file = output_dir / f"results_{model_short_name}_{timestamp}_partial.json"
                
                with open(current_file, 'w', encoding='utf-8') as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Currently processed {batch_idx}/{(len(data_list) + batch_size - 1) // batch_size} batches, temporary summary saved to: {current_file}")
        
        # Merge results from all batches
        if all_batch_results:
            final_result = merge_batch_results(all_batch_results)
            
            # Add failed task summary information
            if final_result.get("failed_tasks_count", 0) > 0:
                # Filter for indicator type failed tasks
                indicator_failed_tasks = []
                for task in final_result.get("failed_tasks", []):
                    if task.get("task_type") == "indicator":
                        indicator_failed_tasks.append(task)
                
                indicator_failed_count = len(indicator_failed_tasks)
                
                if indicator_failed_count > 0:
                    # Create failed task summary file
                    failed_dir = output_dir / "failed"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = datetime.now().strftime('%m%d%H%M')
                    model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                    
                    # Save detailed failed task records
                    failed_tasks_file = failed_dir / f"failed_tasks_{model_short_name}_{timestamp}.json"
                    with open(failed_tasks_file, 'w', encoding='utf-8') as f:
                        json.dump(indicator_failed_tasks, f, ensure_ascii=False, indent=2)
                    
                    # Save summary information
                    failed_summary_file = failed_dir / f"failed_summary_{model_short_name}_{timestamp}.json"
                    with open(failed_summary_file, 'w', encoding='utf-8') as f:
                        json.dump([{
                            "task_id": "Summary Failure",
                            "task_type": "Summary Failure",
                            "reason": f"Indicator type failed task count: {indicator_failed_count}",
                            "failed_tasks_file": str(failed_tasks_file)
                        }], f, ensure_ascii=False, indent=2)
                    
                    logging.info(f"Indicator type failed task detailed information saved to: {failed_tasks_file}")
                    logging.info(f"Indicator type failed task summary information saved to: {failed_summary_file}")
                else:
                    logging.info("No indicator type failed tasks to record")
            
            # Save overall results
            timestamp = datetime.now().strftime('%m%d%H%M')
            model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
            results_file = output_dir / f"results_{model_short_name}_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"All batches processed, total time: {total_time:.2f} seconds, overall results saved to: {results_file}")
            
            # Output number of tasks processed and failed
            total_tasks = final_result["overall_metrics"]["all"]["count"]
            failed_tasks_count = final_result.get("failed_tasks_count", 0)
            total_attempted = total_tasks + failed_tasks_count
            success_rate = (total_tasks / total_attempted * 100) if total_attempted > 0 else 0
            
            logging.info(f"\nTask processing statistics:")
            logging.info(f"   Total attempted tasks: {total_attempted}")
            logging.info(f"   Successfully processed tasks: {total_tasks}")
            logging.info(f"   Failed tasks: {failed_tasks_count}")
            logging.info(f"   Success rate: {success_rate:.2f}%")
            
            # Output token usage statistics
            token_usage = final_result.get("token_usage", {})
            logging.info(f"\nTotal token usage statistics:")
            logging.info(f"   Input tokens: {token_usage.get('prompt_tokens', 0)}")
            logging.info(f"   Output tokens: {token_usage.get('completion_tokens', 0)}")
            logging.info(f"   Total tokens: {token_usage.get('total_tokens', 0)}")
            
            return str(results_file)
        else:
            logging.error("No successfully processed batches")
            return None
            
    except Exception as e:
        logging.error(f"Error processing table metrics: {str(e)}")
        logging.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Get configuration from command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="process table metrics evaluation")
    parser.add_argument("--api_key", required=True, help="API key")
    parser.add_argument("--base_url", default=None, help="API base URL")
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--input_file", required=True, help="input data file path")
    parser.add_argument("--output_dir", required=True, help="output directory")
    parser.add_argument("--batch_size", type=int, default=3, help="number of data items to process in each batch")
    parser.add_argument("--use_pdf", action="store_true", help="whether to use PDF files as table data source")
    parser.add_argument("--use_textpdf", action="store_true", help="whether to use extracted PDF text as table data source")
    parser.add_argument("--pdf_extractor", choices=["pdfplumber", "pdfminer", "pypdf", "pymupdf", "pdftotext", "mineru"], 
                         default="pdfplumber", help="when using textpdf mode, choose which extractor result")
    parser.add_argument("--use_model_for_pdf", action="store_true", help="whether to use multimodal model to directly read PDF")
    parser.add_argument("--task_types", type=str, default="fact,indicator", help="task types to process, separated by commas, e.g. 'fact,indicator'")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Call main function
    results_file = process_table_metrics(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        input_file=args.input_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_pdf=args.use_pdf,
        use_textpdf=args.use_textpdf,
        pdf_extractor=args.pdf_extractor,
        use_model_for_pdf=args.use_model_for_pdf,
        task_types=args.task_types
    )
    
    if results_file:
        print(f"processing complete! overall results saved to: {results_file}") 