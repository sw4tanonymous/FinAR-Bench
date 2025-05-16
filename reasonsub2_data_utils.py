import os
import json
import glob
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

def load_financial_data(data_path: str) -> List[Dict]:

    if os.path.isdir(data_path):
        print(f"loading data from directory {data_path}")
        results_files = []
        for file in os.listdir(data_path):
            if file.endswith('.json'):
                results_files.append(os.path.join(data_path, file))
    
        if not results_files:
            raise FileNotFoundError(f"cannot find financial data in directory: {data_path}")
        
        all_data_items = set()  # use set to remove duplicates
        for model_file in results_files:
            print(f"loading data from model file {model_file}")
            with open(model_file, 'r', encoding='utf-8') as f:
                model_results = json.load(f)
                
            for item in model_results.get("results", []):
                task_id = item.get("task_id", "")
                company_code = item.get("company_code", "")
                if task_id and company_code:  # Only add valid entries
                    all_data_items.add((task_id, company_code))
        
        data_items = [{"task_id": task_id, "company_code": company_code} 
                     for task_id, company_code in all_data_items]
        
        print(f"loaded {len(data_items)} unique financial data from all model result files")
        return data_items
    
    if not os.path.exists(data_path):
        alt_path = "experiments_reasoning/text/without_more_prompt/reasoning_text_sub2"
        
        results_files = []
        if os.path.exists(alt_path):
            for file in os.listdir(alt_path):
                if file.endswith('.json'):
                    results_files.append(os.path.join(alt_path, file))
        
        if not results_files:
            raise FileNotFoundError(f"Financial data not found: {data_path}")
        
        all_data_items = set()
        for model_file in results_files:
            print(f"loading data from model file {model_file}")
            with open(model_file, 'r', encoding='utf-8') as f:
                model_results = json.load(f)
                
            for item in model_results.get("results", []):
                task_id = item.get("task_id", "")
                company_code = item.get("company_code", "")
                if task_id and company_code:  # Only add valid entries
                    all_data_items.add((task_id, company_code))
        
        data_items = [{"task_id": task_id, "company_code": company_code} 
                     for task_id, company_code in all_data_items]
        
        print(f"loaded {len(data_items)} unique financial data from all model result files")
        return data_items
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
                
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    
    return data

def save_results(results: Dict, output_path: str) -> None:
    """
    save evaluation results
    
    Args:
        results: evaluation results
        output_path: output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"evaluation results saved to {output_path}")

def generate_summary(results: Dict) -> str:
    """
    generate evaluation results summary
    
    Args:
        results: evaluation results
        
    Returns:
        summary text
    """
    summary = "# Evaluation Results Summary\n\n"
    
    if "summary" in results:
        summary += "## Model Rankings\n\n"
        
        stats = results["summary"].get("individual_stats", {})
        sorted_models = sorted(stats.items(), key=lambda x: x[1].get("rank", float('inf')))
        
        for model, model_stats in sorted_models:
            win_rate = model_stats.get("win_rate", 0) * 100
            wins = model_stats.get("wins", 0)
            losses = model_stats.get("losses", 0)
            ties = model_stats.get("ties", 0)
            rank = model_stats.get("rank", "-")
            
            summary += f"{rank}. {model}: Win Rate {win_rate:.1f}% (Wins: {wins}, Losses: {losses}, Ties: {ties})\n"
    
    if "summary" in results and "head_to_head" in results["summary"]:
        h2h = results["summary"]["head_to_head"]
        models = list(h2h.keys())
        
        summary += "\n## Model Comparisons\n\n"
        
        for model_a in models:
            summary += f"### {model_a} vs other models\n\n"
            
            for model_b in models:
                if model_a != model_b and model_b in h2h.get(model_a, {}):
                    stats = h2h[model_a][model_b]
                    wins = stats.get("wins", 0)
                    losses = stats.get("losses", 0)
                    ties = stats.get("ties", 0)
                    total = stats.get("total", 0)
                    win_rate = (wins / total) * 100 if total > 0 else 0
                    
                    summary += f"- vs {model_b}: win rate {win_rate:.1f}% (wins: {wins}, losses: {losses}, ties: {ties})\n"
            
            summary += "\n"
    
    return summary

def export_results_to_excel(results_path: str, output_path: str) -> None:
    """
    Export tournament results to Excel format.
    
    Args:
        results_path: Path to the JSON results file
        output_path: Path to save the Excel file
    """
    try:
        # Load results
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Rankings sheet
            rankings_df = pd.DataFrame(results["rankings"], columns=["Model", "Score"])
            rankings_df.to_excel(writer, sheet_name="Rankings", index=False)
            
            # Match results sheet
            if "match_results" in results:
                # Create a simplified version of match results for Excel
                match_data = []
                for match in results["match_results"]:
                    match_data.append({
                        "Data Item": match.get("data_id", "Unknown"),
                        "Model A": match.get("model_a", ""),
                        "Model B": match.get("model_b", ""),
                        "Winner": match.get("winner", ""),
                        "Justification": match.get("justification", "")
                    })
                
                if match_data:
                    match_df = pd.DataFrame(match_data)
                    match_df.to_excel(writer, sheet_name="Match Results", index=False)
            
            # Pairwise matrix
            if "scores" in results:
                models = list(results["scores"].keys())
                matrix = pd.DataFrame(0, index=models, columns=models)
                
                # Fill matrix from match results
                if "match_results" in results:
                    for match in results["match_results"]:
                        model_a = match.get("model_a", "")
                        model_b = match.get("model_b", "")
                        winner = match.get("winner", "")
                        
                        if winner == "A":
                            matrix.loc[model_a, model_b] += 1
                        elif winner == "B":
                            matrix.loc[model_b, model_a] += 1
                        elif winner == "Tie":
                            matrix.loc[model_a, model_b] += 0.5
                            matrix.loc[model_b, model_a] += 0.5
                
                matrix.to_excel(writer, sheet_name="Pairwise Matrix")
                
        print(f"Results exported to Excel file: {output_path}")
        
    except Exception as e:
        print(f"Error exporting results to Excel: {e}")

