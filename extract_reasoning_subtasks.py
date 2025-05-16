import os
import json
import re
import glob
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from process_table_results import normalize_markdown

# Define the input and output directories
input_dir = "experiments_reasoning/text/without_more_prompt"
output_dir_sub1 = "experiments_reasoning/text/without_more_prompt/reasoning_text_sub1"
output_dir_sub2 = "experiments_reasoning/text/without_more_prompt/reasoning_text_sub2"

# Create the output directories if they don't exist
os.makedirs(output_dir_sub1, exist_ok=True)
os.makedirs(output_dir_sub2, exist_ok=True)

def normalize_prediction(prediction):
    """
    normalize the model prediction results to 0/1/2/3 format
    handle various expressions: yes/no/true/false/satisfied/unsatisfied, etc.
    0: negative
    1: positive
    2: special value (None, NaN, "-", etc.)
    3: other (neither positive nor negative nor special value)
    """
    if not prediction:
        return 2  # special mark, representing None/NaN/"-" etc.
    
    pred = prediction.lower().strip()
    
    positive_expressions = {'是', '成立', 'true', 'yes', '1', '1.0', '满足','√','✓','✔','✅','✔️'}
    negative_expressions = {'否', '不成立', 'false', 'no', '0', '0.0', '不满足','×','✕','✗','✘','❌'}
    
    special_values = {'none', 'nan', 'null', 'na', '-', '/', 'n/a', '--'}
    
    if pred in positive_expressions:
        return 1
    elif pred in negative_expressions:
        return 0
    elif pred in special_values:
        return 2  
    
    return 3  

def parse_markdown_table(table_text):
    """
    parse the markdown table, return the index and the prediction
    """
    if not table_text:
        return {}
    
    result = {}
    lines = table_text.strip().split('\n')
    if not lines:
        return {}
    
    index_col_pos = -1   
    pred_col_pos = -1  
    
    if len(lines) > 0 and '|' in lines[0]:
        header_parts = [p.strip().lower() for p in lines[0].split('|') if p.strip()]
        for idx, part in enumerate(header_parts):
            # find the index column
            if any(col_name in part for col_name in ['序号', '编号', 'index', 'no', 'num']):
                index_col_pos = idx
            # find the prediction column (whether satisfied/unsatisfied)
            if any(col_name in part for col_name in ['是否', '成立', '满足', 'yes', 'no']):
                pred_col_pos = idx
    
    if (index_col_pos == -1 or pred_col_pos == -1) and len(lines) > 1:
        for i in range(1, min(3, len(lines))):
            parts = [p.strip() for p in lines[i].split('|') if p.strip()]
            if len(parts) >= 3:
                # assume the first column is index, the third column is prediction (yes/no)
                index_col_pos = 0
                pred_col_pos = 2
                break
    
    if index_col_pos == -1:
        index_col_pos = 0
    if pred_col_pos == -1:
        if len(lines) > 1:
            parts = [p.strip() for p in lines[1].split('|') if p.strip()]
            pred_col_pos = 1 if len(parts) <= 2 else 2
        else:
            pred_col_pos = 1
    
    print(f"Identified columns - Index column: {index_col_pos}, Prediction column: {pred_col_pos}")
    
    for i, line in enumerate(lines):
        if i == 0 or all(c in ['-', '|'] for c in line.replace(' ', '')):
            continue
            
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 2: 
            try:
                index = None
                if index_col_pos < len(parts):
                    try:
                        index_text = parts[index_col_pos]
                        digits = ''.join(c for c in index_text if c.isdigit())
                        if digits:
                            index = int(digits)
                    except (ValueError, IndexError):
                        pass
                
                if index is None:
                    for part in parts:
                        digits = ''.join(c for c in part if c.isdigit())
                        if digits:
                            try:
                                index = int(digits)
                                break
                            except ValueError:
                                continue
                
                if index is not None:
                    prediction_text = None
                    if pred_col_pos >= 0 and pred_col_pos < len(parts):
                        prediction_text = parts[pred_col_pos]
                    elif len(parts) >= 2:
                        prediction_text = parts[min(pred_col_pos, len(parts)-1)]
                    
                    if prediction_text:
                        normalized = normalize_prediction(prediction_text)
                if normalized is not None:
                    result[index] = normalized
            except (ValueError, IndexError) as e:
                print(f"Error processing line: {line} - {e}")
                continue
    
    print(f"Parsed {len(result)} prediction values from table")
    return result

def calculate_metrics(ground_truth_dict, prediction_dict):
    """
    calculate precision and recall
    for special value (None, NaN, "-", etc.):
    1. if ground truth is special value (2), as long as prediction is also special value (2) or other text (3), it is considered correct
    2. if ground truth is not special value, prediction must match ground truth exactly to be considered correct
    """
    correct_predictions = 0
    total_predictions = len(prediction_dict)
    total_ground_truth = len(ground_truth_dict)
    
    # calculate the number of correct predictions
    for index, pred_value in prediction_dict.items():
        if index in ground_truth_dict:
            gt_value = ground_truth_dict[index]
            
            # rule 1: if ground truth is special value (2), as long as prediction is also special value (2) or other text (3), it is considered correct
            if gt_value == 2 and (pred_value == 2 or pred_value == 3):
                correct_predictions += 1
            # rule 2: if ground truth is not special value, prediction must match ground truth exactly to be considered correct
            elif gt_value == pred_value:
                correct_predictions += 1
    
    # calculate precision and recall
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'total_ground_truth': total_ground_truth
    }

def extract_subtasks_from_prediction(prediction):
    """
    Extract the two subtasks from the prediction field:
    1. The judgment table (whether conditions are satisfied) - from first | to last | without \n\n break
    2. The analysis of the conditions - everything after subtask1
    """
    if not prediction or len(prediction.strip()) == 0:
        print("Warning: Prediction is empty or whitespace only")
        return None, None
    
    print(f"Processing prediction: {prediction[:100]}...")
    
    # first check if there is a task1/table/judgment condition/index | whether marker
    task_markers = ["任务1", "表格", "判断条件", "序号 | 是否"]
    for marker in task_markers:
        if marker in prediction:
            marker_pos = prediction.find(marker)
            print(f"Found task marker '{marker}' at position {marker_pos}")
            # find the first line with "|" after the marker
            newline_after_marker = prediction.find('\n', marker_pos)
            if newline_after_marker != -1:
                # find the first line with "|"
                pipe_line_start = newline_after_marker
                while True:
                    next_newline = prediction.find('\n', pipe_line_start + 1)
                    if next_newline == -1:
                        break
                    line = prediction[pipe_line_start+1:next_newline]
                    
                    # check if the line contains at least 3 "|" to ensure it is a table line rather than random "|"
                    if '|' in line and line.count('|') >= 3:
                        # find the start of the table
                        table_start = pipe_line_start + 1
                        # continue to find the end of the table (no "|" in the line)
                        table_end = next_newline
                        while True:
                            next_line_end = prediction.find('\n', table_end + 1)
                            if next_line_end == -1:
                                table_end = len(prediction)
                                break
                            if '|' not in prediction[table_end+1:next_line_end]:
                                # if the next line does not contain "|", then the current line is the end of the table
                                break
                            table_end = next_line_end
                        
                        # extract the table content
                        table_content = prediction[table_start:table_end].strip()
                        print(f"Extracted table from task marker: {table_content[:100]}...")
                        
                        # check if it is a valid table content
                        if '|' in table_content and table_content.count('|') >= 4 and '\n' in table_content:
                            # extract the content after the table as subtask2
                            subtask2_content = prediction[table_end:].strip()
                            return table_content, subtask2_content
                        break
                    pipe_line_start = next_newline
    
    # try method 1: find the content wrapped by ```markdown or ```
    markdown_pattern = re.compile(r"```(?:markdown)?\s*([\s\S]*?)```")
    markdown_matches = markdown_pattern.findall(prediction)
    
    if markdown_matches and len(markdown_matches) > 0:
        print(f"Found {len(markdown_matches)} markdown blocks")
        
        # check if the first match contains a table
        subtask1_content = markdown_matches[0].strip()
        print(f"First markdown block: {subtask1_content[:100]}...")
        
        # check if the extracted content contains a valid table structure
        if '|' in subtask1_content and '\n' in subtask1_content and subtask1_content.count('|') >= 4:
            # check if the extracted content is not pure numbers
            if not is_numeric_content(subtask1_content):
                print("Valid table found in markdown block")
                
                end_of_first_block = prediction.find("```", prediction.find("```") + 3) + 3
                if end_of_first_block < len(prediction):
                    subtask2_content = prediction[end_of_first_block:].strip()
                else:
                    subtask2_content = None
                
                return subtask1_content, subtask2_content
            else:
                print("Warning: Markdown block contains only numeric content")
        else:
            print("Warning: Markdown block does not contain valid table structure")
    
    print("Falling back to pipe character detection")
    
    # try method 2: find the table pattern with sequence (the first column is index, the second column is yes/no)
    table_lines = []
    found_table = False
    for line in prediction.split('\n'):
        line = line.strip()
        if '|' in line:
            # ensure the line contains enough "|" (at least 3, forming at least 2 columns)
            if line.count('|') >= 3:
                if not found_table:
                    # check if it is the header or the first row of data
                    parts = [p.strip().lower() for p in line.split('|') if p.strip()]
                    if len(parts) >= 2:
                        # check if the first column is related to index, or a number
                        if any(col_name in parts[0] for col_name in ['序号', '编号', 'index', 'no', 'num']) or parts[0].isdigit():
                            # check if the second column contains "是否" related words
                            if len(parts) > 1 and any(col_name in parts[1] for col_name in ['是否', '成立', '满足', '是', '否']):
                                found_table = True
                                table_lines.append(line)
                                continue
                table_lines.append(line)
        elif found_table and len(table_lines) > 2:
            break
    
    if found_table and len(table_lines) >= 3: 
        subtask1_content = '\n'.join(table_lines)
        print(f"Found table with pattern matching: {subtask1_content[:100]}...")
        
        table_end_pos = prediction.find(table_lines[-1]) + len(table_lines[-1])
        subtask2_content = prediction[table_end_pos:].strip() if table_end_pos < len(prediction) else None
        
        return subtask1_content, subtask2_content
    
    # try method 3: find the text block with consecutive "|"
    lines = prediction.split('\n')
    potential_table_start = -1
    
    for i, line in enumerate(lines):
        # check if the line contains enough "|" (at least 3, forming at least 2 columns)
        if '|' in line and line.count('|') >= 3:
            # check if the subsequent lines also contain "|"
            if i + 2 < len(lines) and '|' in lines[i+1] and '|' in lines[i+2]:
                potential_table_start = i
                break
    
    if potential_table_start != -1:
        # find the potential start of the table, continue to find the end of the table
        potential_table_end = potential_table_start
        for i in range(potential_table_start + 1, len(lines)):
            if '|' in lines[i]:
                potential_table_end = i
            else:
                # if the line without "|" is found, and at least 3 lines of table content are collected, consider the table ended
                if i - potential_table_start >= 3:
                    break
        
        if potential_table_end - potential_table_start >= 2:
            table_content = '\n'.join(lines[potential_table_start:potential_table_end+1])
            print(f"Found consecutive pipe characters forming a table: {table_content[:100]}...")
            
            subtask2_content = '\n'.join(lines[potential_table_end+1:]).strip() if potential_table_end + 1 < len(lines) else None
            
            return table_content, subtask2_content
    
    current_pos = 0
    valid_start = -1
    
    # ensure the found is a table composed of consecutive "|", not isolated "|"
    pipe_count_in_window = 0
    min_pipes_required = 6  # at least 6 "|" (forming a table with at least 3 columns and 2 rows)
    
    while current_pos < len(prediction):
        pipe_index = prediction.find('|', current_pos)
        if pipe_index == -1:
            break
        
        window_end = min(pipe_index + 200, len(prediction))
        pipe_count_in_window = prediction[pipe_index:window_end].count('|')
        
        if pipe_count_in_window >= min_pipes_required:
            valid_start = pipe_index
            print(f"Found valid pipe sequence at position {valid_start}, with {pipe_count_in_window} pipes in window")
            break
        
        current_pos = pipe_index + 1
    
    if valid_start == -1:
        print("No valid pipe character sequence found")
        return None, None
    
    current_pos = valid_start
    
    end_pos = current_pos
    while True:
        next_newline = prediction.find('\n', end_pos + 1)
        if next_newline == -1:
            if '|' in prediction[end_pos + 1:]:
                end_pos = len(prediction)
            break
        
        if '|' in prediction[end_pos + 1:next_newline]:
            end_pos = next_newline
        else:
            break
    
    # extract subtask1 and subtask2
    subtask1_content = prediction[valid_start:end_pos].strip()
    subtask2_content = prediction[end_pos:].strip() if end_pos < len(prediction) else None
    
    # ensure the extracted content is a valid table, containing multiple lines and enough "|"
    if '\n' in subtask1_content and subtask1_content.count('|') >= min_pipes_required:
        print(f"Subtask1 content (first 100 chars): {subtask1_content[:100]}...")
        if subtask2_content:
            print(f"Subtask2 content (first 100 chars): {subtask2_content[:100]}...")
        
        return subtask1_content, subtask2_content
    else:
        print("Extracted content is not a valid table")
        return None, None

def is_numeric_content(text):
    """
    check if the text only contains numbers, symbols and whitespace
    """
    # remove all spaces, negative sign, comma, decimal point and carriage return/line feed
    cleaned_text = re.sub(r'[\s\-,.\n\r]', '', text)
    # check if the remaining text only contains numbers
    return cleaned_text.isdigit() or not cleaned_text

def normalize_reasoning_table(table_text):
    """
    specifically for reasoning task tables, ensuring the header content is retained
    """
    if not table_text:
        return ""
    
    lines = table_text.strip().split('\n')
    normalized_lines = []
    
    # 处理每一行
    for i, line in enumerate(lines):
        # skip the header separator line (contains "-----" )
        if i > 0 and all(c in ['-', '|', ' ',':'] for c in line):
            continue
            
        # split the line, keep the non-empty parts
        parts = [p.strip() for p in line.split('|')]
        
        # if it is the first line (header line), fully retain
        if i == 0:
            normalized_line = ' | '.join(p for p in parts if p)
            if normalized_line:
                normalized_lines.append(normalized_line)
            continue
        
        # process the data line
        # for the data line, standardize the processing
        valid_parts = [p for p in parts if p]
        if len(valid_parts) >= 2:  # at least contains index and a value
            try:
                normalized_parts = [valid_parts[0]]
                
                if len(valid_parts) > 1:
                    if len(valid_parts) > 2 and (
                        "是" in valid_parts[2] or 
                        "否" in valid_parts[2] or 
                        "成立" in valid_parts[2] or 
                        "不成立" in valid_parts[2]
                    ):
                        normalized_parts.append(valid_parts[1])
                        
                        is_value = normalize_prediction(valid_parts[2])
                        if is_value == 1:
                            normalized_parts.append("是")
                        elif is_value == 0:
                            normalized_parts.append("否")
                        else:
                            normalized_parts.append(valid_parts[2])
                    else:
                        for j in range(1, len(valid_parts)):
                            value = valid_parts[j]
                            is_value = normalize_prediction(value)
                            if is_value == 1:
                                normalized_parts.append("是")
                            elif is_value == 0:
                                normalized_parts.append("否")
                            else:
                                normalized_parts.append(value)
                
                normalized_line = ' | '.join(normalized_parts)
                normalized_lines.append(normalized_line)
            except (ValueError, IndexError) as e:
                normalized_line = ' | '.join(valid_parts)
                normalized_lines.append(normalized_line)
        elif valid_parts:
            normalized_line = ' | '.join(valid_parts)
            normalized_lines.append(normalized_line)
    
    return '\n'.join(normalized_lines)

def process_model_directory(model_dir):
    """Process all result files in a model directory."""
    model_name = os.path.basename(model_dir)
    
    results_files = glob.glob(os.path.join(model_dir, "*reasoning_results_all*.json"))
    if not results_files:
        print(f"No result files found in {model_dir}")
        return
    
    sub1_results = []
    sub2_results = []
    failed_task_ids = []
    
    total_metrics = {
        'correct_predictions': 0,
        'total_predictions': 0,
        'total_ground_truth': 0
    }
    
    for file_path in results_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for result in data.get('results', []):
                prediction = result.get('prediction', '')
                
                if not prediction:
                    task_id = result.get('task_id', '')
                    failed_task_ids.append(task_id)
                    continue
                
                task_id = result.get('task_id', '')
                company_code = result.get('company_code', '')
                ground_truth = result.get('ground_truth', '')
                
                subtask1_content, subtask2_content = extract_subtasks_from_prediction(prediction)
                
                if subtask1_content:
                    normalized_ground_truth = normalize_reasoning_table(ground_truth)
                    normalized_subtask1 = normalize_reasoning_table(subtask1_content)
                    
                    # parse ground truth and prediction result
                    ground_truth_dict = parse_markdown_table(normalized_ground_truth)
                    prediction_dict = parse_markdown_table(normalized_subtask1)
                    
                    metrics = calculate_metrics(ground_truth_dict, prediction_dict)
                    
                    total_metrics['correct_predictions'] += metrics['correct_predictions']
                    total_metrics['total_predictions'] += metrics['total_predictions']
                    total_metrics['total_ground_truth'] += metrics['total_ground_truth']
                    
                    sub1_results.append({
                        'task_id': task_id,
                        'company_code': company_code,
                        'ground_truth': normalized_ground_truth,
                        'subtask1_result': normalized_subtask1,
                        'metrics': metrics
                    })
                else:
                    failed_task_ids.append(task_id)
                
                if subtask2_content:
                    sub2_results.append({
                        'task_id': task_id,
                        'company_code': company_code,
                        'ground_truth': ground_truth, 
                        'subtask2_result': subtask2_content
                    })
                else:
                    if subtask1_content and task_id not in failed_task_ids:
                        failed_task_ids.append(task_id)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    overall_precision = total_metrics['correct_predictions'] / total_metrics['total_predictions'] if total_metrics['total_predictions'] > 0 else 0
    overall_recall = total_metrics['correct_predictions'] / total_metrics['total_ground_truth'] if total_metrics['total_ground_truth'] > 0 else 0
    
    # Save subtask1 results
    if sub1_results:
        with open(os.path.join(output_dir_sub1, f"{model_name}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'results': sub1_results,
                'failed_task_ids': failed_task_ids,
                'overall_metrics': {
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'correct_predictions': total_metrics['correct_predictions'],
                    'total_predictions': total_metrics['total_predictions'],
                    'total_ground_truth': total_metrics['total_ground_truth']
                }
            }, f, ensure_ascii=False, indent=2)
    
    # Save subtask2 results
    if sub2_results:
        with open(os.path.join(output_dir_sub2, f"{model_name}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'model': model_name,
                'results': sub2_results,
                'failed_task_ids': failed_task_ids
            }, f, ensure_ascii=False, indent=2)

def main():
    """Main function to process all model directories."""
    model_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    if not model_dirs:
        print(f"No model directories found in {input_dir}")
        return
    
    for model_dir in model_dirs:
        print(f"Processing {model_dir}...")
        process_model_directory(model_dir)
    
    print("Done!")

if __name__ == "__main__":
    main() 