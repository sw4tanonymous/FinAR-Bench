import os
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import sys
import time
import traceback
import copy
import logging

# Import configurations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from model_configs import (
    get_model_params,
    get_interface_type,
    get_model_id,
    get_api_key,
    get_base_url,
    init_api_client,
    INTERFACE_TYPE_VOLCENGINE,
    INTERFACE_TYPE_OPENAI
)

# Import Volcengine SDK (for fallback import checks)
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    print("Failed to import Volcengine Ark SDK. Please install: pip install 'volcengine-python-sdk[ark]'")
    sys.exit(1)

# Import OpenAI SDK (for fallback import checks)
try:
    from openai import OpenAI
except ImportError:
    print("Failed to import OpenAI SDK. Please install: pip install openai")
    sys.exit(1)

from reasonsub2_prompts import UNIFIED_PROMPT, SWAPPED_PROMPT
from reasonsub2_data_utils import load_financial_data, save_results

class TournamentEvaluator:
    def __init__(
        self,
        candidates: List[str],
        judge_model: str,
        data_path: str,
        output_dir: str,
        use_cached: bool = True,
        random_order: bool = False,
        fixed_order_swap: bool = False,
    ):
        """
        Initialize the evaluator
        
        Parameters:
            candidates: List of candidate models to evaluate
            judge_model: Name of the judge model
            data_path: Path to financial data
            output_dir: Directory for saving results
            use_cached: Whether to use cached responses (if available)
            random_order: Whether to randomize the order of analysis A and B
            fixed_order_swap: Whether to always use the swapped order
        """
        self.candidates = candidates
        self.judge_model = judge_model
        self.data_path = data_path
        self.output_dir = output_dir
        self.use_cached = use_cached
        self.random_order = random_order
        self.fixed_order_swap = fixed_order_swap
        
        self.results = None
        
        self._init_judge_client()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        
        if self.random_order:
            logging.info("using random order to evaluate models")
        elif self.fixed_order_swap:
            logging.info("using fixed order swap to evaluate models")
    
    def _init_judge_client(self):
        """initialize the judge model client"""
        try:
            from model_configs import init_api_client
            
            self.interface_type = get_interface_type(self.judge_model)
            self.model_params = get_model_params(self.judge_model)
            
            self.judge_client = init_api_client(self.judge_model)
            
            if self.interface_type == INTERFACE_TYPE_VOLCENGINE:
                self.model_id = get_model_id(self.judge_model)
            else:
                self.model_id = self.judge_model
                
            logging.info(f"initialized judge model: {self.judge_model}")
            
        except Exception as e:
            logging.error(f"failed to initialize judge model client: {str(e)}")
            raise
    
    def _load_model_results(self, model_name: str) -> Dict:
        """Load evaluation results for a model"""
        results_path = os.path.join(self.data_path, f"{model_name}.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Model result file not found: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_judge_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the judge's response to determine the winner
        
        Returns:
            Tuple[str, str]: (winner, reasoning)
        """
        # Default values in case parsing fails
        winner = "tie"
        reasoning = "Could not determine winner from response"
        
        # Parse response text
        try:
            # Try to find key phrases that indicate the winner
            response_lower = response.lower()
            
            # Check for keywords that indicate Model A is better
            if "model a is better" in response_lower or "model a wins" in response_lower:
                winner = "A"
            # Check for keywords that indicate Model B is better
            elif "model b is better" in response_lower or "model b wins" in response_lower:
                winner = "B"
            # Look for explicit preference statements
            elif "prefer model a" in response_lower or "i choose model a" in response_lower:
                    winner = "A"
            elif "prefer model b" in response_lower or "i choose model b" in response_lower:
                    winner = "B"
            # Check for tie statements
            elif "tie" in response_lower or "both models" in response_lower or "equally" in response_lower:
                winner = "tie"
            
            # Extract reasoning
            reasoning_parts = response.split("Reasoning:")
            if len(reasoning_parts) > 1:
                reasoning = reasoning_parts[1].strip()
            else:
                reasoning = response
                
            return winner, reasoning
            
        except Exception as e:
            logging.error(f"Error parsing judge response: {str(e)}")
            return winner, reasoning
    
    def _get_judge_response(self, prompt: str, model_a: str, model_b: str, cache_key: str) -> Tuple[str, Dict]:
        """
        get the judge model's evaluation for two candidate models
        
        Parameters:
            prompt: the prompt sent to the judge model
            model_a: the name of model A
            model_b: the name of model B
            cache_key: the cache key, for identifying this evaluation (may contain order information)
        
        Returns:
            Tuple[str, Dict]: (the response from the judge model, token usage statistics)
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        if self.use_cached:
            try:
                if os.path.exists(cache_file):
                    logging.info(f"using cached evaluation results: {model_a} vs {model_b}")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    token_usage = cache_data.get('token_usage', token_usage)
                    return cache_data.get('response', ''), token_usage
                else:
                    logging.info(f"no cache file found, will perform new evaluation")
            except Exception as e:
                logging.error(f"error reading cache file: {str(e)}")
        
        try:
            logging.info(f"calling the judge model to evaluate: {model_a} vs {model_b}")
            messages = [{"role": "user", "content": prompt}]
            
            response = self.judge_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.model_params.get('temperature', 0.2),
                top_p=self.model_params.get('top_p', 0.7),
                max_tokens=self.model_params.get('max_tokens', 1500)
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if hasattr(response, 'usage'):
                token_usage = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
            
            logging.info(f"Token usage: input={token_usage['prompt_tokens']}, output={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")
            
            if self.use_cached:
                try:
                    cache_dir_path = os.path.dirname(cache_file)
                    if not os.path.exists(cache_dir_path):
                        os.makedirs(cache_dir_path, exist_ok=True)
                    
                    cache_data = {
                        'response': response_text,
                        'token_usage': token_usage,
                        'model_a': model_a,
                        'model_b': model_b
                    }
                    
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    logging.error(f"error caching evaluation results: {str(e)}")
            
            return response_text, token_usage
            
        except Exception as e:
            logging.error(f"error calling the judge model: {str(e)}")
            return "error: cannot get the judge model response", token_usage
    
    def _evaluate_pair(self, model_a: str, model_b: str, data_id: str, company_code: str) -> Dict:
        """Evaluate a pair of models"""
        logging.info(f"evaluating: {model_a} vs {model_b} with task id {data_id}")
        
        token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        try:
            model_a_results = self._load_model_results(model_a)
            model_b_results = self._load_model_results(model_b)
            
            analysis_a = None
            analysis_b = None
            
            for item in model_a_results.get("results", []):
                if item["task_id"] == data_id:
                    analysis_a = item.get("subtask2_result", "")
                    break
            
            for item in model_b_results.get("results", []):
                if item["task_id"] == data_id:
                    analysis_b = item.get("subtask2_result", "")
                    break
            
            if not analysis_a or not analysis_b:
                missing_models = []
                if not analysis_a:
                    missing_models.append(model_a)
                if not analysis_b:
                    missing_models.append(model_b)
                logging.warning(f"cannot find the analysis results for task id {data_id}, missing models: {', '.join(missing_models)}")
                return {
                    "winner": "Tie",
                    "justification": f"Could not find complete analysis results, missing models: {', '.join(missing_models)}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "data_id": data_id,
                    "company_code": company_code,
                    "error": "missing_analysis",
                    "token_usage": token_usage
                }
            
            # Determine if order needs to be swapped
            should_swap = False
            cache_key_suffix = ""
            
            if self.random_order:
                # Randomize order
                should_swap = random.choice([True, False])
                cache_key_suffix = "_random"
            elif self.fixed_order_swap:
                # Fixed swap order
                should_swap = True
                cache_key_suffix = "_swapped"
            
            safe_judge_model = self.judge_model.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.', '-')
            safe_model_a = model_a.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.', '-')
            safe_model_b = model_b.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.', '-')
                
            # Prepare prompt based on swap setting
            if should_swap:
                prompt = SWAPPED_PROMPT.format(
                    analysis_a=analysis_a,
                    analysis_b=analysis_b
                )
                cache_key = f"{safe_judge_model}_{safe_model_a}_vs_{safe_model_b}_{data_id}{cache_key_suffix}"
            else:
                prompt = UNIFIED_PROMPT.format(
                    analysis_a=analysis_a,
                    analysis_b=analysis_b
                )
                cache_key = f"{safe_judge_model}_{safe_model_a}_vs_{safe_model_b}_{data_id}"
            
            # Get judge evaluation and token usage
            judge_response, token_usage = self._get_judge_response(prompt, model_a, model_b, cache_key)
            
            # Parse judge response
            winner, justification = self._parse_judge_response(judge_response)
            
            # If order was swapped, reverse the winner
            if should_swap and winner != "tie":
                winner = "B" if winner == "A" else "A"
            
            # Convert winner from A/B to actual model name
            winner_name = model_a if winner == "A" else (model_b if winner == "B" else "Tie")
            
            # Return results with token usage
            return {
                "winner": winner_name,
                "winner_letter": winner,
                "justification": justification,
                "model_a": model_a,
                "model_b": model_b,
                "data_id": data_id,
                "company_code": company_code,
                "order_swapped": should_swap,
                "judge_response": judge_response,
                "token_usage": token_usage
            }
            
        except FileNotFoundError as e:
            logging.error(f"model result file not found: {str(e)}")
            return {
                "winner": "Tie",
                "justification": f"Could not load model result file: {str(e)}",
                "model_a": model_a,
                "model_b": model_b,
                "data_id": data_id,
                "company_code": company_code,
                "error": "file_not_found",
                "token_usage": token_usage
            }
        except json.JSONDecodeError as e:
            logging.error(f"invalid model result file format: {str(e)}")
            return {
                "winner": "Tie",
                "justification": f"Invalid model result file format: {str(e)}",
                "model_a": model_a,
                "model_b": model_b,
                "data_id": data_id,
                "company_code": company_code,
                "error": "json_decode_error",
                "token_usage": token_usage
            }
        except Exception as e:
            logging.error(f"error evaluating {model_a} vs {model_b}: {str(e)}")
            return {
                "winner": "Tie",
                "justification": f"Evaluation error: {str(e)}",
                "model_a": model_a,
                "model_b": model_b,
                "data_id": data_id,
                "company_code": company_code,
                "error": "evaluation_error",
                "token_usage": token_usage
            }
    
    def run_tournament(self, sample_size: int = None):
        """
        run the evaluation tournament, comparing all candidate models pairwise
        
        Parameters:
            sample_size: Number of data entries to sample, None means use all data
        """
        logging.info(f"starting the evaluation tournament, candidate models: {', '.join(self.candidates)}")
        
        # Load data
        data_items = load_financial_data(self.data_path)
        logging.info(f"loaded {len(data_items)} financial data")
        
        # If sample size is specified, randomly sample
        if sample_size and sample_size < len(data_items):
            data_items = random.sample(data_items, sample_size)
            logging.info(f"randomly sampled {sample_size} data for evaluation")
        
        # Generate all model pair matches
        all_matches = []
        for i, model_a in enumerate(self.candidates):
            for j, model_b in enumerate(self.candidates):
                if i < j:  # Ensure no duplicate comparisons
                    for item in data_items:
                        all_matches.append({
                            "model_a": model_a,
                            "model_b": model_b,
                            "data_id": item["task_id"],
                            "company_code": item["company_code"]
                        })
        
        logging.info(f"total {len(all_matches)} matches need to be evaluated")
        
        # Initialize token usage tracking
        total_token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # Initialize results dictionary with evaluation metadata
        self.results = {
            "meta": {
                "judge_model": self.judge_model,
                "model_id": self.model_id,
                "evaluation_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidates": self.candidates,
                "random_order": self.random_order,
                "fixed_order_swap": self.fixed_order_swap
            },
            "matches": [],
            "summary": {},
            "token_usage": total_token_usage
        }
        
        # Evaluate each match
        for match in tqdm(all_matches, desc="evaluating matches"):
            result = self._evaluate_pair(
                match["model_a"], 
                match["model_b"], 
                match["data_id"], 
                match["company_code"]
            )
            
            # Add to results
            self.results["matches"].append(result)
            
            # Accumulate token usage from this match
            if "token_usage" in result:
                total_token_usage["prompt_tokens"] += result["token_usage"].get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += result["token_usage"].get("completion_tokens", 0)
                total_token_usage["total_tokens"] += result["token_usage"].get("total_tokens", 0)
        
        # Log the total token usage
        logging.info(f"\nToken usage statistics:")
        logging.info(f"  prompt tokens: {total_token_usage['prompt_tokens']}")
        logging.info(f"  completion tokens: {total_token_usage['completion_tokens']}")
        logging.info(f"  total tokens: {total_token_usage['total_tokens']}")
        
        # Update the token usage in results
        self.results["token_usage"] = total_token_usage
        
        # Calculate summary statistics
        self._calculate_summary()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _calculate_summary(self):
        """Calculate tournament summary statistics"""
        # Ensure results are properly initialized
        if self.results is None:
            self.results = {
                "meta": {
                    "judge_model": self.judge_model,
                    "model_id": self.model_id,
                    "evaluation_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "candidates": self.candidates
                },
                "matches": [],
                "summary": {}
            }
        elif "meta" not in self.results:
            # If results initialized but no meta field, add it
            self.results["meta"] = {
                "judge_model": self.judge_model,
                "model_id": self.model_id,
                "evaluation_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidates": self.candidates
            }
        
        # Initialize win, loss, tie counts for each model
        stats = {model: {"wins": 0, "losses": 0, "ties": 0} for model in self.candidates}
        
        # Count results for each match
        for match in self.results["matches"]:
            if match["winner"] == "Tie":
                stats[match["model_a"]]["ties"] += 1
                stats[match["model_b"]]["ties"] += 1
            else:
                stats[match["winner"]]["wins"] += 1
                loser = match["model_b"] if match["winner"] == match["model_a"] else match["model_a"]
                stats[loser]["losses"] += 1
        
        # Calculate win rate for each model
        for model, counts in stats.items():
            total_matches = counts["wins"] + counts["losses"] + counts["ties"]
            counts["win_rate"] = counts["wins"] / total_matches if total_matches > 0 else 0
            counts["total_matches"] = total_matches
        
        # Sort by win rate
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: (x[1]["win_rate"], x[1]["wins"]), 
            reverse=True
        )
        
        # Add rankings
        ranked_stats = {}
        for i, (model, stats) in enumerate(sorted_stats):
            ranked_stats[model] = {**stats, "rank": i + 1}
        
        # Add head-to-head statistics
        head_to_head = {}
        for model_a in self.candidates:
            head_to_head[model_a] = {}
            for model_b in self.candidates:
                if model_a != model_b:
                    wins = 0
                    losses = 0
                    ties = 0
                    
                    for match in self.results["matches"]:
                        if match["model_a"] == model_a and match["model_b"] == model_b:
                            if match["winner"] == model_a:
                                wins += 1
                            elif match["winner"] == model_b:
                                losses += 1
                            else:
                                ties += 1
                        elif match["model_a"] == model_b and match["model_b"] == model_a:
                            if match["winner"] == model_a:
                                wins += 1
                            elif match["winner"] == model_b:
                                losses += 1
                            else:
                                ties += 1
                    
                    head_to_head[model_a][model_b] = {
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "total": wins + losses + ties
                    }
        
        # Save statistics
        self.results["summary"] = {
            "individual_stats": ranked_stats,
            "head_to_head": head_to_head
        }
    
    def _save_results(self):
        """Save the evaluation results to files"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define output paths
        results_path = os.path.join(self.output_dir, f"tournament_results_{timestamp}.json")
        
        # Create a copy of the results with additional metadata
        output_results = copy.deepcopy(self.results)
        
        for match in output_results.get("matches", []):
            if "justification" in match:
                del match["justification"]
        
        # Add extra metadata
        output_results["meta"]["timestamp"] = timestamp
        output_results["meta"]["data_path"] = self.data_path
        output_results["meta"]["output_dir"] = self.output_dir
        
        # Get token usage statistics
        token_usage = output_results.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        
        # Save results to JSON
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"\nresults saved to {results_path}")
        
        # Print summary statistics
        logging.info("\nsummary statistics:")
        for model_name, stats in self.results["summary"]["individual_stats"].items():
            win_rate = stats["win_rate"] * 100
            logging.info(f"  {model_name}:")
            logging.info(f"    - wins: {stats['wins']} ({win_rate:.2f}%)")
            logging.info(f"    - losses: {stats['losses']}")
            logging.info(f"    - ties: {stats['ties']}")
        
        # Print token usage
        logging.info(f"\nToken usage:")
        logging.info(f"  - prompt tokens: {prompt_tokens}")
        logging.info(f"  - completion tokens: {completion_tokens}")
        logging.info(f"  - total tokens: {total_tokens}")
        
        return results_path

def main():
    parser = argparse.ArgumentParser(description="run the financial analysis model evaluation tournament")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="the output directory, default is reason_tournament/[judge model name]_judge")
    parser.add_argument("--data-path", type=str, default="experiments_reasoning/text/without_more_prompt/reasoning_text_sub2", help="the financial data path")
    parser.add_argument("--sample-size", type=int, default=3, help="the number of data entries to sample, default is 3, set to 0 to use all data")
    parser.add_argument("--use-cached", action="store_true", help="use cached evaluation responses (default enabled)")
    parser.add_argument("--no-cache", action="store_true", help="disable cache, call API every time")
    parser.add_argument("--judge-model", type=str, default="volcengine-deepseek-v3", 
                       help="the name of the evaluation model, default is volcengine-deepseek-v3")
    parser.add_argument("--models", type=str, nargs="+", default=None, 
                       help="specify the models to evaluate, separated by spaces, no models specified will use all models in the data directory")
    parser.add_argument("--order", choices=["fixed", "random", "swap"], default="fixed",
                      help="the order mode of analysis: fixed=fixed order, random=random order, swap=swap order")
    args = parser.parse_args()
    
    # set the output directory
    if args.output_dir is None:
        # extract the short name from the judge_model (remove the path and version number)
        judge_name = args.judge_model.split('/')[-1] if '/' in args.judge_model else args.judge_model
        judge_name = judge_name.split('-')[0] if '-' in judge_name else judge_name
        args.output_dir = f"reason_tournament/{judge_name}_judge"
    
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{args.output_dir}/logs/tournament_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"evaluation parameters: judge model={args.judge_model}, data path={args.data_path}")
    
    model_dir = args.data_path
    available_models = []
    
    if args.models:
        for model in args.models:
            model_file = os.path.join(model_dir, f"{model}.json")
            if os.path.exists(model_file):
                available_models.append(model)
            else:
                logging.warning(f"cannot find the specified model file: {model_file}")
    
    if not available_models:
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.json'):
                    model_name = os.path.splitext(file)[0]
                    available_models.append(model_name)
    
    
    if len(available_models) < 2:
        logging.error("cannot find enough model files, at least 2 models are needed for comparison")
        sys.exit(1)
    
    logging.info(f"evaluating models: {', '.join(available_models)}")
    
    use_cached = not args.no_cache  # if --no-cache is specified, disable cache
    if args.no_cache:
        logging.info("cache function is disabled")
    
    random_order = (args.order == "random")
    fixed_order_swap = (args.order == "swap")
    
    evaluator = TournamentEvaluator(
        candidates=available_models,
        judge_model=args.judge_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_cached=use_cached,  # if --no-cache is specified, disable cache
        random_order=random_order,
        fixed_order_swap=fixed_order_swap
    )
    
    sample_size = None if args.sample_size <= 0 else args.sample_size
    
    results = evaluator.run_tournament(sample_size=sample_size)
    
    logging.info("\nfinal ranking:")
    sorted_models = sorted(
        results["summary"]["individual_stats"].items(),
        key=lambda x: x[1]["rank"]
    )
    
    for model, stats in sorted_models:
        win_rate = stats["win_rate"] * 100
        logging.info(f"{stats['rank']}. {model}: win rate {win_rate:.1f}% (wins: {stats['wins']}, losses: {stats['losses']}, ties: {stats['ties']})")

if __name__ == "__main__":
    main() 