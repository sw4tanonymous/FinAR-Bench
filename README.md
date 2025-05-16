# FinAR-Bench

FinAR-Bench is designed to assess the capabilities of Large Language Models (LLMs) in performing financial fundamental analysis.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys and parameters:
Edit the `model_configs.py` file to set your API keys and other configuration settings.

## Dataset

The FinAR-Bench dataset is hosted on Hugging Face and can be accessed using the Hugging Face datasets library.

### Downloading the Dataset

You can download the dataset using the Hugging Face datasets library:

```python
from datasets import load_dataset
dataset = load_dataset("sw4tanonymous/FinAR-Bench")
```

The dataset contains:
1. Company Tables and PDFs (`pdf_data`): Financial statements from 2023 annual reports of 100 companies listed on the Shanghai Stock Exchange (SSE)
2. Extracted Text (`pdf_extractor_result/txt_output`): Text extracted from PDFs using six different PDF extraction tools
3. Development Set (`dev.txt`): Evaluation tasks for 10 companies
4. Test Set (`test.txt`): Evaluation tasks for 90 companies

Each company's evaluation set contains 13 tasks:
- 6 fact extraction tasks
- 6 financial indicator computation tasks
- 1 logical reasoning task

For more details about the dataset structure and usage, please visit the [dataset page on Hugging Face](https://huggingface.co/datasets/sw4tanonymous/FinAR-Bench).


## Project Setup

After downloading the FinAR-Bench repository, you need to properly organize the dataset files downloaded from Hugging Face.

### Directory Structure

Set up your project with the following structure:

```
FinAR-Bench/
├── data/
│   ├── dev.txt         # Development set (from Hugging Face)
│   └── test.txt        # Test set (from Hugging Face)
├── pdf_data/           # PDF files (from Hugging Face)
│   ├── 600000.pdf
│   ├── 600001.pdf
│   └── ...
├── pdf_extractor_result/
│   └── txt_output/     # Extracted text files (from Hugging Face)
│       ├── mineru
│       ├── pdfminer
│       └── ...
├── README.md
├── requirements.txt
├── config.py
├── model_configs.py
└── ... (other script files)
```

### Steps to Set Up

1. Clone the FinAR-Bench repository
2. Download the dataset from [Hugging Face](https://huggingface.co/datasets/sw4tanonymous/FinAR-Bench)
3. Create a `data` folder and place `dev.txt` and `test.txt` inside it
4. Extract `pdf_data.zip` and place its contents in the `pdf_data` folder
5. Extract `pdf_extractor_result.zip` and place its contents in the `pdf_extractor_result` folder

> **Note:** The correct directory structure is crucial for the evaluation scripts to work properly.


## Configuration

Configurations are managed in two main files:
- `config.py`: Contains general system settings, processing options, and task configurations
- `model_configs.py`: Contains model-specific parameters, API endpoints, and interface types

The system supports multiple API interfaces:
- OpenAI-compatible APIs (including NVIDIA and other providers)
- Volcengine APIs
- Custom APIs (configurable through model_configs.py)

## Usage

> **Note:** For all commands, `--model` should be the full model name as defined in `model_configs.py`

### Running Fact and Indicator Evaluation

#### Text Mode
```bash
python table_main_metrics.py --mode text --model full_model_name [--more-prompt]
```

#### Textpdf Mode
```bash
python table_main_metrics.py --mode textpdf --model full_model_name --pdf_extractor pymupdf [--more-prompt]
```

### Running Reasoning Evaluation

To evaluate models on their financial reasoning capabilities:

#### Step 1: Collect Model Responses
Run this to collect model responses (avoiding incorrect regularization extraction):

```bash
python reasoning_main.py --model full_model_name
```

#### Step 2: Extract Reasoning Subtasks
Extract reasoning_subtask1 and reasoning_subtask2:

```bash
python extract_reasoning_subtasks.py
```

The default paths are configured as follows (can be modified in extract_reasoning_subtasks.py):
```bash
input_dir = "experiments_reasoning/text/without_more_prompt"
output_dir_sub1 = "experiments_reasoning/without_more_prompt/reasoning_text_sub1"
output_dir_sub2 = "experiments_reasoning/without_more_prompt/reasoning_text_sub2"
```

The evaluation result of reasoning_subtask1 will be available in `experiments_reasoning/reasoning_text_sub1` after running this step.

#### Step 3: Tournament Evaluation
Run a comparison tournament between multiple models to assess reasoning_subtask2:

```bash
python reasonsub2_tournament_evaluation.py --judge-model full_model_name --sample-size 0
```
> Note: Setting `--sample-size` to 0 means all samples will be evaluated


### Using Batch Processing Mode

For parallel processing of many tasks, you should run this after completing the table_main_metrics.py related operations. This two-step approach is intentional: the second step uses more detailed regex patterns that can be customized for specific models. If everything were processed in a single step and errors occurred, you would need to rerun the entire procedure. This approach allows you to refine the regex patterns(if need) in the second step without repeating the initial processing.

#### Process with predefined file list
You can define specific files to process in the `batch_process_results.py` file by editing the file lists in the dictionary:
```python
# In batch_process_results.py
file_lists = {
    "text": ['experiments_text/without_more_prompt/model-name/results_file.json'],
    "text_more_prompt": [],
    "mineru": [],
    "pdfminer": [],
    # ... other file lists
}
```

Then select which file list to process by setting the `list_type` variable:
```python
# Choose which list to process
list_type = "text"  # Use the "text" list
# list_type = "mineru"  # Or use the "mineru" list
```

The output will be automatically saved to a directory based on the chosen list type (e.g., `results_summary/text/model_name/`).

Then run:
```bash
python batch_process_results.py
```

#### Process a single file
Process a specific file using process_table_results.py directly:
```bash
python process_table_results.py --input path/to/input.json --output path/to/output.json --number_theta 0
```


## Input Data Format

The input file(dev.txt/test.txt) for table metrics evaluation tasks are in the following JSON format:

```json
{
  "table": "# Income Statement\n| Item | Dec 31, 2023 | Dec 31, 2022 |\n|...",
  "instances": [
    {
      "task_id": "ec503ba06eeb4fe88584060770807b52",
      "task": "Extract sales expenses for 2022 and 2023...",
      "ground_truth": "Item,2022,2023\nSales expenses,23.3,13.5",
      "task_type": "fact",
      "task_num": 1,
      "company": "Company Name",
      "company_code": "600234.SH"
    },
    ...
  ]
}
```

## Evaluation Metrics

Model evaluations use the following metrics:
- Precision
- Recall
- F1 Score
- Numerical Accuracy

## Output Results

Evaluation results are saved in the specified output directory, including:
- Detailed results in JSON format
- Evaluation metrics grouped by task type

## Project Structure

- `table_metrics_processor.py`: Core module for processing table data and evaluating LLM tasks
- `batch_process_results.py`: Provides different number theta to process
- `table_main_metrics.py`: Main program for fact and indicator evaluation
- `reasoning_main.py`: Main program for reasoning evaluation
- `reasonsub2_tournament_evaluation.py`: Tournament-style evaluation between multiple models
- `config.py`: Unified configuration management file
- `model_configs.py`: Model-specific configuration parameters
- `reasonsub2_prompts.py`: Prompts for model evaluation in reasoning tasks
- `reasonsub2_data_utils.py`: Utilities for loading and processing reasoning data

## Requirements

- Python 3.8+
- OpenAI Python package
- Volcengine Python SDK for Ark (for Volcengine models)
- Pandas, NumPy, and other data processing libraries
- tqdm for progress bars

