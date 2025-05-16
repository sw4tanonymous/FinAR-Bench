import json
import re
import os
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.root.addHandler(console_handler)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from deplot.metrics import table_datapoints_precision_recall, table_number_accuracy_extext, _parse_table
    logging.info("Success importing deplot evaluation metrics")
except ImportError as e:
    logging.warning(f"Failed to import deplot evaluation metrics: {str(e)}")


def extract_table_content(text):
    """extract table content from text, extract content before \n\n"""
    split_content = text.split('\n\n', 1)
    return split_content[0].strip() if len(split_content) > 1 else text.strip()

def process_prediction_with_prefix_suffix(prediction, prefix_list=None, suffix_list=None):
    """process prediction results, extract content based on specified prefixes and suffixes
    
    Args:
        prediction (str): prediction result text
        prefix_list (list): list of prefix strings, if None, no processing
        suffix_list (list): list of suffix strings, if None, use all content
        
    Returns:
        str: processed prediction result
    """
    result = prediction
    
    prefix_empty = not prefix_list
    suffix_empty = not suffix_list
    
    has_calculation = "计算" in prediction
    
    has_project = "\n项目" in prediction
    
    if prefix_empty and suffix_empty and not has_calculation and not has_project:
        return prediction
    
    if has_calculation:
        calculation_indices = []
        current_pos = 0
        while True:
            formula_pos = prediction.find("计算", current_pos)
            if formula_pos < 0:
                break
            calculation_indices.append(formula_pos)
            current_pos = formula_pos + 2  
        
        if calculation_indices:
            formula_pos = calculation_indices[0]
            if formula_pos > 0:
                newline_pos = prediction.rfind('\n', 0, formula_pos)
                if newline_pos >= 0:
                    prediction = prediction[:newline_pos]
                    result = prediction
    
    if has_project:
        project_pos = prediction.find("\n项目")
        if project_pos >= 0:
            prediction = "项目" + prediction[project_pos+3:]
            result = prediction
    
    if prefix_list or suffix_list:
        last_prefix_pos = -1
        last_prefix = None
        if prefix_list:
            for prefix in prefix_list:
                pos = prediction.find(prefix)
                if pos >= 0 and (last_prefix_pos < 0 or pos > last_prefix_pos):
                    last_prefix_pos = pos
                    last_prefix = prefix
        
        if last_prefix_pos >= 0:
            content_start = last_prefix_pos + len(last_prefix)
            content_end = len(prediction)
            
            if suffix_list:
                for suffix in suffix_list:
                    suffix_pos = prediction.find(suffix, content_start)
                    if suffix_pos >= 0:
                        content_end = suffix_pos
                        break
            
            result = prediction[content_start:content_end]
            
            lines = result.split('\n', 1)
            if lines:
                first_line = lines[0]
                def replace_number(match):
                    num = match.group(0)
                    if num.endswith('.0'):
                        return num[:-2]
                    return num
                
                first_line = re.sub(r'\b\d+\.0\b', replace_number, first_line)
                
                if len(lines) > 1:
                    result = first_line + '\n' + lines[1]
                else:
                    result = first_line
        # no prefix but has suffix
        elif suffix_list:
            first_suffix_pos = len(prediction)  # default to text end
            first_suffix = None
            
            for suffix in suffix_list:
                suffix_pos = prediction.find(suffix)
                if suffix_pos >= 0 and suffix_pos < first_suffix_pos:
                    first_suffix_pos = suffix_pos
                    first_suffix = suffix
            
            if first_suffix_pos < len(prediction) and first_suffix is not None:
                result = prediction[:first_suffix_pos]
    
    # final check if result contains "计算" two words, if so, truncate
    if "计算" in result:
        calc_pos = result.find("计算")
        if calc_pos > 0:
            # find the nearest newline before "计算"
            nl_pos = result.rfind('\n', 0, calc_pos)
            if nl_pos >= 0:
                # only keep the content before the newline
                result = result[:nl_pos]
            else:
                # if no newline is found, it is probably in the first line, truncate
                result = result[:calc_pos].strip()
    
    # process the first line of the table, remove .0 suffix
    lines = result.split('\n')
    if lines and '|' in lines[0]:
        # process the first line
        cells = lines[0].split('|')
        processed_cells = []
        
        for cell in cells:
            # use regex to find numbers and remove .0 suffix
            def replace_number(match):
                num = match.group(0)
                if num.endswith('.0'):
                    return num[:-2]
                return num
            
            processed_cell = re.sub(r'\b\d+\.0\b', replace_number, cell)
            processed_cells.append(processed_cell)
        
        lines[0] = '|'.join(processed_cells)
        
        result = '\n'.join(lines)
    
    return result.strip()

def normalize_markdown(markdown_text):
    if not markdown_text:
        return "| |"
        
    lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
    if not lines:
        return "| |"
        
    normalized_lines = []
    for i, line in enumerate(lines):
        if i > 0 and all(c in ['-', '|', ':', ' '] for c in line):
            continue
            
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if parts:
                processed_parts = []
                for j, part in enumerate(parts):
                    if i == 0 or j == 0:
                        processed_parts.append(part)
                    else:
                        if '%' in part:
                            try:
                                num = float(part.replace('%', '')) / 100
                                processed_parts.append(str(num))
                            except ValueError:
                                processed_parts.append(part)
                        elif 'e' in part.lower():
                            try:
                                num = float(part)
                                processed_parts.append(str(num))
                            except ValueError:
                                processed_parts.append(part)
                        elif any(c.isdigit() for c in part):
                            try:
                                num = float(part.replace(',', ''))
                                processed_parts.append(str(num))
                            except ValueError:
                                processed_parts.append(part)
                        else:
                            processed_parts.append(part)
                            
                normalized_lines.append(' | '.join(processed_parts))
        else:
            normalized_lines.append(line)
            
    if not normalized_lines:
        return "| |"
        
    return '\n'.join(normalized_lines)

def evaluate_prediction(prediction, ground_truth, number_theta=None):
    try:
        gt_markdown = normalize_markdown(ground_truth)
        pred_markdown = normalize_markdown(prediction)

        targets = [[gt_markdown]]   
        predictions = [pred_markdown]   

        result = {}
        
        if number_theta is not None:
            n_theta = float(number_theta)
            result = table_datapoints_precision_recall(targets, predictions, number_theta=n_theta)
            try:
                result.update(table_number_accuracy_extext(targets, predictions, number_theta=n_theta))
            except Exception as e:
                logging.warning(f"Error calculating number accuracy: {str(e)}")
        else:
            result = table_datapoints_precision_recall(targets, predictions)
            try:
                result.update(table_number_accuracy_extext(targets, predictions))
            except Exception as e:
                logging.warning(f"Error calculating number accuracy: {str(e)}")
        
        return result
    
    except Exception as e:
        logging.error(f"Error evaluating prediction results: {str(e)}")
        return {
            "table_datapoints_precision": 0.0,
            "table_datapoints_recall": 0.0,
            "table_datapoints_f1": 0.0,
            "numbers_match_extext": 0.0
        }

def calculate_overall_metrics(results, number_theta=None):
    """calculate overall metrics
    
    Args:
        results (list): processed results list
        number_theta (float): threshold for number metrics
        
    Returns:
        dict: overall metrics
    """
    # get all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    
    task_types = {}
    task_nums = {}
    task_num_types = {}
    
    for result in results:
        prediction = result.get("prediction", "")
        ground_truth = result.get("ground_truth", "")
        
        if not prediction or not ground_truth:
            continue
            
        all_predictions.append(prediction)
        all_ground_truths.append(ground_truth)
        
        task_type = result.get("task_type", "unknown")
        if task_type not in task_types:
            task_types[task_type] = {"predictions": [], "ground_truths": []}
        task_types[task_type]["predictions"].append(prediction)
        task_types[task_type]["ground_truths"].append(ground_truth)
        
        task_num = result.get("task_num", 0) 
        task_num_key = f"task_num_{task_num}"
        if task_num_key not in task_nums:
            task_nums[task_num_key] = {"predictions": [], "ground_truths": []}
        task_nums[task_num_key]["predictions"].append(prediction)
        task_nums[task_num_key]["ground_truths"].append(ground_truth)
        
        task_num_type_key = f"task_num_{task_num}_{task_type}"
        if task_num_type_key not in task_num_types:
            task_num_types[task_num_type_key] = {"predictions": [], "ground_truths": []}
        task_num_types[task_num_type_key]["predictions"].append(prediction)
        task_num_types[task_num_type_key]["ground_truths"].append(ground_truth)

    overall_metrics = {}
    
    if not all_predictions or not all_ground_truths:
        logging.warning("Cannot find valid prediction and ground truth pairs, cannot calculate metrics")
        overall_metrics["all"] = {
            "table_datapoints_precision": 0.0,
            "table_datapoints_recall": 0.0,
            "table_datapoints_f1": 0.0,
            "count": 0
        }
        return overall_metrics
        
    try:
        all_targets = [[gt] for gt in all_ground_truths]
        
        if number_theta is not None:
            n_theta = float(number_theta)
            overall_metrics["all"] = table_datapoints_precision_recall(all_targets, all_predictions, number_theta=n_theta)
            try:
                overall_metrics["all"].update(table_number_accuracy_extext(all_targets, all_predictions, number_theta=n_theta))
            except Exception as e:
                logging.warning(f"Error calculating overall number accuracy: {str(e)}")
        else:
            overall_metrics["all"] = table_datapoints_precision_recall(all_targets, all_predictions)
            try:
                temp_result = table_number_accuracy_extext(all_targets, all_predictions)
                overall_metrics["all"].update(temp_result)
            except Exception as e:
                logging.warning(f"Error calculating number accuracy: {str(e)}")
            
        overall_metrics["all"]["count"] = len(all_predictions)
    except Exception as e:
        logging.error(f"Error calculating overall metrics: {str(e)}")
        overall_metrics["all"] = {
            "table_datapoints_precision": 0.0,
            "table_datapoints_recall": 0.0,
            "table_datapoints_f1": 0.0,
            "count": len(all_predictions)
        }
    
    # calculate metrics by task type
    for task_type, data in task_types.items():
        try:
            targets = [[gt] for gt in data["ground_truths"]]
            predictions = data["predictions"]
            
            # apply different number matching logic based on number_theta value
            if number_theta is not None:
                n_theta = float(number_theta)
                overall_metrics[task_type] = table_datapoints_precision_recall(targets, predictions, number_theta=n_theta)
                try:
                    overall_metrics[task_type].update(table_number_accuracy_extext(targets, predictions, number_theta=n_theta))
                except Exception as e:
                    logging.warning(f"Error calculating number accuracy for task type {task_type}: {str(e)}")
            else:
                overall_metrics[task_type] = table_datapoints_precision_recall(targets, predictions)
                try:
                    overall_metrics[task_type].update(table_number_accuracy_extext(targets, predictions))
                except Exception as e:
                    logging.warning(f"Error calculating number accuracy for task type {task_type}: {str(e)}")
            
            overall_metrics[task_type]["count"] = len(predictions)
        except Exception as e:
            logging.error(f"Error calculating metrics for task type {task_type}: {str(e)}")
            overall_metrics[task_type] = {
                "table_datapoints_precision": 0.0,
                "table_datapoints_recall": 0.0,
                "table_datapoints_f1": 0.0,
                "count": len(data["predictions"])
            }
    
    # calculate metrics by task number
    for task_num_key, data in task_nums.items():
        try:
            targets = [[gt] for gt in data["ground_truths"]]
            predictions = data["predictions"]
            
            # apply different number matching logic based on number_theta value
            if number_theta is not None:
                n_theta = float(number_theta)
                overall_metrics[task_num_key] = table_datapoints_precision_recall(targets, predictions, number_theta=n_theta)
                try:
                    overall_metrics[task_num_key].update(table_number_accuracy_extext(targets, predictions, number_theta=n_theta))
                except Exception as e:
                    logging.warning(f"Error calculating number accuracy for task number {task_num_key}: {str(e)}")
            else:
                overall_metrics[task_num_key] = table_datapoints_precision_recall(targets, predictions)
                try:
                    overall_metrics[task_num_key].update(table_number_accuracy_extext(targets, predictions))
                except Exception as e:
                    logging.warning(f"Error calculating number accuracy for task number {task_num_key}: {str(e)}")
            
            overall_metrics[task_num_key]["count"] = len(predictions)
        except Exception as e:
            logging.error(f"Error calculating metrics for task number {task_num_key}: {str(e)}")
            overall_metrics[task_num_key] = {
                "table_datapoints_precision": 0.0,
                "table_datapoints_recall": 0.0,
                "table_datapoints_f1": 0.0,
                "count": len(data["predictions"])
            }
    
    # calculate metrics by task number and task type combination
    for task_num_type_key, data in task_num_types.items():
        try:
            targets = [[gt] for gt in data["ground_truths"]] 
            predictions = data["predictions"]
            
            # apply different number matching logic based on number_theta value
            if number_theta is not None:
                n_theta = float(number_theta)
                overall_metrics[task_num_type_key] = table_datapoints_precision_recall(targets, predictions, number_theta=n_theta)
                try:
                    overall_metrics[task_num_type_key].update(table_number_accuracy_extext(targets, predictions, number_theta=n_theta))
                except Exception as e:
                    logging.warning(f"Error calculating number accuracy for task number and type combination {task_num_type_key}: {str(e)}")
            else:
                overall_metrics[task_num_type_key] = table_datapoints_precision_recall(targets, predictions)
                try:
                    overall_metrics[task_num_type_key].update(table_number_accuracy_extext(targets, predictions))
                except Exception as e:
                    logging.warning(f"Error calculating number accuracy for task number and type combination {task_num_type_key}: {str(e)}")
            
            overall_metrics[task_num_type_key]["count"] = len(predictions)
        except Exception as e:
            logging.error(f"Error calculating metrics for task number and type combination {task_num_type_key}: {str(e)}")
            overall_metrics[task_num_type_key] = {
                "table_datapoints_precision": 0.0,
                "table_datapoints_recall": 0.0,
                "table_datapoints_f1": 0.0,
                "count": len(data["predictions"])
            }
    
    if number_theta is not None:
        logging.info(f"Using number_theta={number_theta} as the matching threshold")
    
    return overall_metrics

def process_results(input_file, output_file, prefix_list=None, suffix_list=None, number_theta=None, company_list=None):
    """process model results file
    
    Args:
        input_file (str): input file path
        output_file (str): output file path
        prefix_list (list): list of prefix strings
        suffix_list (list): list of suffix strings
        number_theta (float): threshold for number metrics
        company_list (list): list of company codes to filter
        
    Returns:
        bool: whether the processing is successful
    """
    try:
        # read input file
        logging.info(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Successfully loaded input file (size: {os.path.getsize(input_file)} bytes)")
        
        results = data.get('results', [])
        if not results:
            logging.error(f"No valid results found in {input_file}")
            return False
        
        logging.info(f"Found {len(results)} results in input file")
        
        if company_list:
            original_count = len(results)
            results = [result for result in results if result.get('company_code') not in company_list]
            filtered_count = original_count - len(results)
            if filtered_count > 0:
                logging.info(f"Deleted {filtered_count} tasks for specified companies")
        
        for result in results:
            if 'prediction' in result:
                prediction = extract_table_content(result['prediction'])
                prediction = process_prediction_with_prefix_suffix(prediction, prefix_list, suffix_list)
                result['prediction'] = prediction
            
            if 'ground_truth' in result and 'prediction' in result:
                gt_markdown = normalize_markdown(result['ground_truth'])
                pred_markdown = normalize_markdown(result['prediction'])
                
                result['ground_truth'] = gt_markdown
                result['prediction'] = pred_markdown
                result['processed_prediction'] = pred_markdown
                
                try:
                    evaluation = evaluate_prediction(pred_markdown, gt_markdown, number_theta)
                    result['metrics'] = evaluation
                except Exception as e:
                    logging.error(f"Error evaluating single result: {str(e)}")
                    result['metrics'] = {
                        "table_datapoints_precision": 0.0,
                        "table_datapoints_recall": 0.0,
                        "table_datapoints_f1": 0.0,
                        "numbers_match_extext": 0.0
                    }
        
        # calculate overall metrics
        overall_metrics = calculate_overall_metrics(results, number_theta)
        
        # update overall metrics and results
        data['overall_metrics'] = overall_metrics
        data['results'] = results
        
        # save processed results
        logging.info(f"Writing processed results to: {output_file}")
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(output_file):
                logging.info(f"Successfully wrote output file: {output_file}")
                return True
            else:
                logging.error(f"File writing operation failed")
                return False
        except Exception as e:
            logging.error(f"Error writing output file: {str(e)}")
            return False
        
    except Exception as e:
        logging.error(f"Error processing results file: {str(e)}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process table evaluation results')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--number_theta', '-nt', type=float, default=None,
                        help='Number matching threshold for precision/recall calculation')
    args = parser.parse_args()
    
    # input and output file paths
    input_file = args.input
    output_file = args.output
    number_theta = args.number_theta
    
    # prefix and suffix lists for result extraction
    prefix_list = ["结果：\n", "markdown\n",'结果如下\n','财务指标\n','财务指标计算\n','数据：\n']
    suffix_list = ["\n检查", "\n注",'\n### 计算','\n计算','\n* 注：','\n解释：','\n说明','\n请注意','\n**注意','\n这']
    
    # ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # process results
    success = process_results(
        input_file=input_file,
        output_file=output_file,
        prefix_list=prefix_list,
        suffix_list=suffix_list,
        number_theta=number_theta
    )
    
    if success:
        logging.info(f"Processing completed successfully, results saved to: {output_file}")
    else:
        logging.error(f"Processing failed")

if __name__ == '__main__':
    main()