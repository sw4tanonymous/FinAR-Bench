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
import PyPDF2
import time
import concurrent.futures
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from deplot import metrics

from model_configs import (
    get_model_params, 
    get_interface_type, 
    get_model_id,
    INTERFACE_TYPE_VOLCENGINE,
    INTERFACE_TYPE_OPENAI
)

def setup_logging(log_dir="logs", level=None):
    log_level = getattr(logging, level or config.LOGGING_LEVEL) if hasattr(config, 'LOGGING_LEVEL') else logging.INFO
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'reasoning_{timestamp}.log'

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding='utf-8')
        ],
        force=True
    )
    logging.info("logging system initialized, level: %s", logging.getLevelName(log_level))


def init_api_client(model_name, custom_api_key=None, custom_base_url=None):
    try:
        from model_configs import init_api_client as model_init_api_client
        return model_init_api_client(model_name, custom_api_key=custom_api_key, custom_base_url=custom_base_url)
    except Exception as e:
        logging.error(f"API client initialization failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def encode_pdf_to_base64(pdf_path):
    """Encode PDF to base64 string"""
    try:
        with open(pdf_path, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"PDF encoding failed: {str(e)}")
        return None


def extract_table_from_pdf_with_model(client, pdf_path, model_name):
    logging.info(f"using model to read PDF file: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file does not exist: {pdf_path}")
        return None
    
    try:
        pdf_base64 = encode_pdf_to_base64(pdf_path)
        if not pdf_base64:
            return None
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "请从这份PDF文件中提取所有财务表格数据（利润表、资产负债表、现金流量表等），保持原始格式输出。将每个表格用Markdown格式呈现，并在表格前添加表格名称作为标题（# 表格名称）。"},
                        {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_base64}"}}
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.0
        )
        
        table_text = response.choices[0].message.content.strip()
        logging.info(f"model successfully extracted table data, content length: {len(table_text)}")
        
        return table_text
    
    except Exception as e:
        logging.error(f"using model to read PDF failed: {str(e)}")
        logging.error(traceback.format_exc())
        return None


def check_txt_file_exists(company_code, pdf_extractor="pdfplumber"):
    """Check if text file exists for the given company code"""
    # Remove the suffix (e.g. .SH or .SZ) from the company code
    company_code = company_code.split('.')[0]
    
    # Get the project root directory (the parent directory of the current file)
    root_dir = Path(__file__).parent.parent
    txt_dir = root_dir / "pdf_extractor_result/txt_output" / pdf_extractor
    
    # Check if the directory exists
    if not txt_dir.exists():
        logging.warning(f"Text file directory does not exist: {txt_dir}")
        return None
        
    # Try to find the corresponding text file
    pattern = f"{company_code}*.txt"
    matching_files = list(txt_dir.glob(pattern))
    
    if matching_files:
        return str(matching_files[0])
    
    return None


def load_pdf_text_from_file(company_code, pdf_extractor="pdfplumber"):
    """load PDF text from the pre-extracted text file"""
    txt_path = check_txt_file_exists(company_code, pdf_extractor)
    
    if not txt_path:
        logging.error(f"pre-extracted text file for {company_code} does not exist")
        return None
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"successfully loaded PDF text from file: {txt_path}")
        return text
    except Exception as e:
        logging.error(f"failed to read text file: {str(e)}")
        return None


def load_reasoning_tasks(file_path, use_textpdf=False, pdf_extractor="pdfplumber", task_types=None):
    """load reasoning task data from file"""
    try:
        logging.info(f"loading reasoning task data: {file_path}")
        
        if not os.path.exists(file_path):
            logging.error(f"reasoning task data file does not exist: {file_path}")
            return None
        
        allowed_task_types = None
        if task_types:
            if isinstance(task_types, str):
                allowed_task_types = [t.strip() for t in task_types.split(',')]
            elif isinstance(task_types, list):
                allowed_task_types = task_types
            logging.info(f"will only process tasks of the following types: {', '.join(allowed_task_types)}")
        else:
            allowed_task_types = ["reasoning"]
            logging.info("default only process reasoning type tasks")
            
        data_list = []
        skipped_count = 0
        filtered_by_range_count = 0
        filtered_by_type_count = 0
        start_idx = config.PROCESS_DATA_RANGE["start"] - 1 
        end_idx = config.PROCESS_DATA_RANGE["end"] 
        step = config.PROCESS_DATA_RANGE["step"]
        
        with open(file_path, 'r', encoding='utf-8') as f:
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
                    if not line.strip():
                        continue
                        
                    data = json.loads(line.strip())
                    
                    if "instances" not in data:
                        logging.warning(f"skipping line {line_num}: missing instances field")
                        skipped_count += 1
                        continue
                        
                    valid_instances = []
                    for instance in data.get("instances", []):
                        if not all(key in instance for key in ["task_id", "task", "ground_truth", "task_type", "company_code"]):
                            logging.warning(f"skipping instance {instance.get('task_id', 'unknown')}: missing required fields")
                            skipped_count += 1
                            continue
                            
                        instance_type = instance.get("task_type", "").lower()
                        if allowed_task_types and instance_type not in [t.lower() for t in allowed_task_types]:
                            logging.debug(f"skipping instance {instance.get('task_id', 'unknown')}: task type {instance_type} not in allowed types")
                            filtered_by_type_count += 1
                            continue
                            
                        if use_textpdf:
                            company_code = instance.get("company_code")
                            if not check_txt_file_exists(company_code, pdf_extractor):
                                logging.warning(f"skipping instance {instance.get('task_id')}: can't find corresponding text file")
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
                    
        logging.info(f"successfully loaded {len(data_list)} task data, skipped {skipped_count} invalid data")
        logging.info(f"filtered {filtered_by_range_count} lines based on data range")
        if filtered_by_type_count > 0:
            logging.info(f"filtered {filtered_by_type_count} instances based on task type")
        
        if not data_list:
            logging.warning("no valid task data found, please check the data")
        
        return data_list
    
    except Exception as e:
        logging.error(f"failed to load task data: {str(e)}")
        logging.error(traceback.format_exc())
        return None


def extract_reasoning_tasks_and_tables(data, client, model_name, use_pdf=False, use_textpdf=False, pdf_extractor="pdfplumber", use_model_for_pdf=False):
    """extract reasoning tasks and table content from data"""
    instances = data.get("instances", [])
    
    reasoning_instances = []
    for instance in instances:
        if instance.get("task_type", "").lower() == "reasoning":
            reasoning_instances.append(instance)
        else:
            logging.warning(f"在 extract_reasoning_tasks_and_tables 中跳过非 reasoning 任务: {instance.get('task_id', 'unknown')}, 类型: {instance.get('task_type', 'unknown')}")
    
    instances = reasoning_instances
    
    if not instances:
        logging.warning("no valid reasoning task instances found, please check the data")
        return None, []
    
    if use_textpdf and instances:
        company_code = instances[0].get("company_code")
        if not company_code:
            logging.warning("no company code found in instances")
            table = data.get("table", "")
            return table, instances
            
        logging.info(f"attempting to load table from extracted text, company code: {company_code}")
        
        table = load_pdf_text_from_file(company_code, pdf_extractor)
        
        if not table:
            logging.warning(f"failed to load table from extracted text, company code: {company_code}, will skip this task")
            return None, None 
            
        logging.info(f"successfully loaded table from extracted text, company code: {company_code}")
        
        enriched_instances = []
        for instance in instances:
            instance_copy = instance.copy()
            instance_copy["_pdf_text"] = table
            enriched_instances.append(instance_copy)
        instances = enriched_instances
    elif use_pdf and "file_path" in data:
        pdf_path = data.get("file_path")
        logging.info(f"attempting to load table from PDF file: {pdf_path}")
        
        if use_model_for_pdf and client:
            table = extract_table_from_pdf_with_model(client, pdf_path, model_name)
        else:
            print("failed to use model for pdf")
            
        if not table:
            logging.warning(f"failed to extract table from PDF, using table in JSON as backup")
            table = data.get("table", "")
    else:
        table = data.get("table", "")
        logging.info(f"using table in JSON, length: {len(table) if table else 0}")
    
    return table, instances


def analyze_reasoning_task(client, table, task, model_name, max_retries=3, use_textpdf=False, more_prompt=False, conditions=None):
    retries = max_retries if max_retries is not None else 3
    
    from model_configs import get_model_params, get_interface_type, get_model_id, INTERFACE_TYPE_VOLCENGINE
    
    model_params = get_model_params(model_name)
    
    interface_type = get_interface_type(model_name)
    
    actual_model_name = model_name
    if interface_type == INTERFACE_TYPE_VOLCENGINE:
        actual_model_name = get_model_id(model_name)
    
    logging.info(f"table data length: {len(table) if table else 0}")
    
    token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }
    
    for attempt in range(retries):
        try:
            indicator_formulas = """
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
"""
            
            if use_textpdf:
                if more_prompt:
                    prompt = f"请{task}\n\n判断条件：{conditions}\n\n{indicator_formulas}\n\n以下是财务报表数据：\n\n{table}"
                else:
                    prompt = f"请{task}\n\n判断条件：{conditions}\n\n以下是财务报表数据：\n\n{table}"
            else:
                if more_prompt:
                    prompt = f"请{task}\n\n判断条件：{conditions}\n\n{indicator_formulas}\n\n以下是财务表格数据，单位为元:\n\n{table}"
                else:
                    prompt = f"请{task}\n\n判断条件：{conditions}\n\n以下是财务表格数据，单位为元:\n\n{table}"
            
            messages = [{"role": "user", "content": prompt}]
            
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
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=model_params.get('temperature', 0.6),
                    top_p=model_params.get('top_p', 0.7),
                    max_tokens=model_params.get('max_tokens', 1500)
                )
            
            result = response.choices[0].message.content.strip()
            
            if hasattr(response, 'usage'):
                token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                    'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                    'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
                }
            
            logging.info(f"token usage: in={token_usage.get('prompt_tokens', 0)}, out={token_usage.get('completion_tokens', 0)}, total={token_usage.get('total_tokens', 0)}")
            logging.info(f"API call successful, response length: {len(result)}")
            return result, token_usage
            
        except Exception as e:
            logging.error(f"API call failed (attempt {attempt+1}/{retries}): {str(e)}")
            logging.error(traceback.format_exc())
            
            if attempt < retries - 1:
                sleep_time = 2 ** attempt
                logging.info(f"waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
    
    logging.error(f"all retries failed")
    return None, token_usage


def process_single_reasoning_task(task_data, client, model_name, use_textpdf=False, more_prompt=False):
    """process a single reasoning task"""
    task_id = task_data.get("task_id", "")
    task_type = task_data.get("task_type", "")
    task_description = task_data.get("task", "")
    ground_truth = task_data.get("ground_truth", "")
    company_code = task_data.get("company_code", "")

    if task_type.lower() != "reasoning":
        logging.warning(f"skip non-reasoning type task: {task_id}, type: {task_type}")
        return None, {"task_id": task_id, "task_type": task_type, "company_code": company_code, "reason": "task type is not reasoning"}

    table = task_data.get("_pdf_text", "") if use_textpdf else task_data.get("table", "")
    
    if not table:
        logging.error(f"task {task_id} has no table data")
        return None, {"task_id": task_id, "task_type": task_type, "company_code": company_code, "reason": "no table data"}
    
    conditions = task_data.get("conditions", "")
    
    logging.info(f"processing reasoning task: {task_id} ({task_type})")
    
    try:
        prediction, token_usage = analyze_reasoning_task(
            client, 
            table, 
            task_description, 
            model_name, 
            use_textpdf=use_textpdf,
            more_prompt=more_prompt,
            conditions=conditions
        )
        
        if not prediction:
            logging.error(f"task {task_id} failed to get prediction result")
            return None, {"task_id": task_id, "task_type": task_type, "company_code": company_code, "reason": "no prediction result"}
            
        return {
            "task_id": task_id,
            "task_type": task_type,
            "task": task_description,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "company_code": company_code,
            "token_usage": token_usage
        }, None
        
    except Exception as e:
        logging.error(f"task {task_id} processing failed: {str(e)}")
        logging.error(traceback.format_exc())
        return None, {"task_id": task_id, "task_type": task_type, "company_code": company_code, "reason": str(e)}


def process_reasoning_tasks_parallel(task_instances, client, model_name, max_workers=None, use_textpdf=False, more_prompt=False):
    """process multiple reasoning tasks in parallel"""
    results = []
    failed_tasks = []
    
    total_token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }
    
    workers = max_workers if max_workers is not None else min(8, len(task_instances))
    logging.info(f"use {workers} threads to process {len(task_instances)} tasks")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                process_single_reasoning_task, 
                task, 
                client, 
                model_name, 
                use_textpdf,
                more_prompt
            ): task for task in task_instances
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(task_instances), desc="processing tasks"):
            task = future_to_task[future]
            try:
                result, failed_info = future.result()
                if result:
                    results.append(result)
                    
                    if "token_usage" in result:
                        total_token_usage["prompt_tokens"] += result["token_usage"]["prompt_tokens"]
                        total_token_usage["completion_tokens"] += result["token_usage"]["completion_tokens"]
                        total_token_usage["total_tokens"] += result["token_usage"]["total_tokens"]
                        
                        logging.debug(f"task {result['task_id']} token usage: in={result['token_usage']['prompt_tokens']}, out={result['token_usage']['completion_tokens']}, total={result['token_usage']['total_tokens']}")
                    
                    logging.info(f"task {result['task_id']} processed")
                elif failed_info:
                    failed_tasks.append(failed_info)
                    logging.warning(f"task {failed_info['task_id']} recorded as failed task")
            except Exception as e:
                task_id = task.get("task_id", "unknown")
                company_code = task.get("company_code", "")
                logging.error(f"task {task_id} processing failed: {str(e)}")
                failed_tasks.append({
                    "task_id": task_id,
                    "task_type": task.get("task_type", "unknown"),
                    "company_code": company_code,
                    "reason": str(e)
                })
    
    logging.info(f"batch total token usage: in={total_token_usage['prompt_tokens']}, out={total_token_usage['completion_tokens']}, total={total_token_usage['total_tokens']}")
    
    return results, failed_tasks, total_token_usage


def save_failed_reasoning_tasks(failed_tasks, output_dir, model_name, batch_id=None, append=False):
    """Save failed reasoning tasks"""
    if not failed_tasks:
        logging.info("No failed tasks to save")
        return
    
    failed_dir = Path(output_dir) / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    if append:
        failed_file = failed_dir / f"failed_reasoning_tasks_{model_name}.json"
        
        existing_tasks = []
        try:
            if os.path.exists(failed_file):
                with open(failed_file, 'r', encoding='utf-8') as f:
                    existing_tasks = json.load(f)
                logging.info(f"Read existing failed task records, containing {len(existing_tasks)} records")
        except Exception as e:
            logging.warning(f"Failed to read existing failed task file: {str(e)}, will create a new file")
        
        task_ids = set(task["task_id"] for task in existing_tasks)
        for task in failed_tasks:
            if task["task_id"] not in task_ids:
                existing_tasks.append(task)
                task_ids.add(task["task_id"])
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(existing_tasks, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Appended {len(failed_tasks)} failed task records to: {failed_file}, total {len(existing_tasks)} records")
    else:
        timestamp = datetime.now().strftime('%m%d%H%M')
        batch_suffix = f"_batch{batch_id}" if batch_id is not None else ""
        failed_file = failed_dir / f"failed_reasoning_tasks_{model_name}{batch_suffix}_{timestamp}.json"
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_tasks, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved {len(failed_tasks)} failed task records to: {failed_file}")


def batch_process_reasoning_tasks(data, model_name, output_dir, batch_id=None, use_pdf=False, use_textpdf=False, pdf_extractor="pdfplumber", use_model_for_pdf=False, more_prompt=False):
    """process a batch of reasoning tasks"""
    client = init_api_client(model_name)
    
    table, instances = extract_reasoning_tasks_and_tables(
        data, 
        client, 
        model_name, 
        use_pdf=use_pdf, 
        use_textpdf=use_textpdf,
        pdf_extractor=pdf_extractor,
        use_model_for_pdf=use_model_for_pdf
    )
    
    logging.info(f"extracted table data length: {len(table) if table else 0}")
    
    if not use_textpdf and table:
        for instance in instances:
            if "table" not in instance or not instance.get("table"):
                instance["table"] = table
    
    if use_textpdf and table is None:
        company = data.get("instances", [])[0].get("company", "Unknown") if data.get("instances", []) else "Unknown"
        company_code = data.get("instances", [])[0].get("company_code", "Unknown") if data.get("instances", []) else "Unknown"
        logging.info(f"skip company {company} ({company_code}), cannot find corresponding PDF text file")
        
        return {
            "results": [],
            "failed_tasks": [{
                "task_id": instance.get("task_id", "unknown"),
                "task_type": instance.get("task_type", "unknown"),
                "company_code": instance.get("company_code", company_code),
                "reason": f"cannot find corresponding PDF text file for company {company_code}"
            } for instance in instances],
            "skipped": True
        }
    
    from model_configs import get_interface_type, INTERFACE_TYPE_VOLCENGINE
    interface_type = get_interface_type(model_name)
    use_volcengine = (interface_type == INTERFACE_TYPE_VOLCENGINE)
    
    max_workers = getattr(config, 'MAX_WORKERS', 1) if use_volcengine else 1
    results, failed_tasks, token_usage = process_reasoning_tasks_parallel(
        instances, client, model_name, max_workers, use_textpdf, more_prompt
    )
    
    if failed_tasks:
        save_failed_reasoning_tasks(failed_tasks, output_dir, model_name, batch_id)
    
    logging.info(f"\n Token usage statistics:")
    if token_usage and all(key in token_usage for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']):
        logging.info(f"  prompt tokens: {token_usage['prompt_tokens']}")
        logging.info(f"  completion tokens: {token_usage['completion_tokens']}")
        logging.info(f"  total tokens: {token_usage['total_tokens']}")
    else:
        logging.warning("cannot get complete token usage information")
        logging.info(f"original token usage data: {token_usage}")
    
    return {
        "results": results,
        "failed_tasks": failed_tasks,
        "token_usage": token_usage
    }


def merge_batch_results(all_batch_results):
    """Merge results from multiple batches"""
    all_results = []
    failed_tasks_count = 0
    
    total_token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }
    
    for batch_result in all_batch_results:
        if batch_result and "results" in batch_result:
            all_results.extend(batch_result["results"])
        
        if batch_result and "failed_tasks" in batch_result:
            failed_tasks_count += len(batch_result["failed_tasks"])
        
        if batch_result and "token_usage" in batch_result:
            token_usage = batch_result["token_usage"]
            if isinstance(token_usage, dict) and all(key in token_usage for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']):
                total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
                total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
                total_token_usage["total_tokens"] += token_usage["total_tokens"]
            else:
                logging.warning(f"token usage format exception in batch: {token_usage}")
    
    overall_results = {
        "total_tasks": len(all_results) + failed_tasks_count,
        "processed_tasks": len(all_results),
        "failed_tasks": failed_tasks_count
    }
    
    logging.info(f"\n Total token usage statistics:")
    logging.info(f"  Prompt tokens: {total_token_usage['prompt_tokens']}")
    logging.info(f"  Completion tokens: {total_token_usage['completion_tokens']}")
    logging.info(f"  Total tokens: {total_token_usage['total_tokens']}")
    
    return {
        "results": all_results,
        "overall_results": overall_results,
        "token_usage": total_token_usage
    }


def process_reasoning_tasks(model_name, input_file, output_dir, batch_size=3, use_pdf=False, use_textpdf=False, pdf_extractor="pdfplumber", use_model_for_pdf=False, more_prompt=False, task_types="reasoning"):
    """process reasoning tasks evaluation main function"""
    try:
        setup_logging()
        
        all_data = load_reasoning_tasks(input_file, use_textpdf, pdf_extractor, task_types)
        if not all_data:
            logging.error("cannot find valid task data")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        from model_configs import get_interface_type, INTERFACE_TYPE_VOLCENGINE, INTERFACE_TYPE_OPENAI
        interface_type = get_interface_type(model_name)
        
        if interface_type == INTERFACE_TYPE_VOLCENGINE:
            logging.info(f"use volcengine API: {model_name}")
        elif interface_type == INTERFACE_TYPE_OPENAI:
            logging.info(f"use openai compatible API: {model_name}")
        else:
            logging.info(f"use custom API: {model_name}")
        
        batch_results = []
        for i, data in enumerate(all_data):
            batch_id = i + 1
            logging.info(f"\n==== process batch {batch_id}/{len(all_data)} ====")
            
            batch_result = batch_process_reasoning_tasks(
                data,
                model_name,
                output_dir,
                batch_id=batch_id,
                use_pdf=use_pdf,
                use_textpdf=use_textpdf,
                pdf_extractor=pdf_extractor,
                use_model_for_pdf=use_model_for_pdf,
                more_prompt=more_prompt
            )
            
            if batch_result:
                batch_results.append(batch_result)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_file = Path(output_dir) / f"reasoning_results_batch{batch_id}_{timestamp}.json"
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_result["results"], f, ensure_ascii=False, indent=2)
                
                logging.info(f"batch {batch_id} results saved to: {result_file}")
        
        final_results = merge_batch_results(batch_results)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_result_file = Path(output_dir) / f"reasoning_results_all_{timestamp}.json"
        
        with open(final_result_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"all results saved to: {final_result_file}")
        
        return final_results
        
    except Exception as e:
        logging.error(f"error processing reasoning tasks: {str(e)}")
        logging.error(traceback.format_exc())
        return None 