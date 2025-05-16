import os
from pathlib import Path
from model_configs import (
    OPENAI_INTERFACE_MODELS, 
    VOLCENGINE_INTERFACE_MODELS,
    NVIDIA_MODELS,
    OPENAI_MODELS,
    VOLCENGINE_MODELS
)

# set to True to try to read table content from PDF files, set to False to use the table field in JSON
USE_PDF_SOURCE = False
# set to True to use multimodal model to read PDF, False to use PyPDF2 to parse
USE_MODEL_FOR_PDF = True

USE_VOLCENGINE = False
USE_OPENAI = False

BATCH_SIZE = 10
INPUT_FILE_PATH = "data/dev.txt" 

MAX_RETRIES = 3  # max retries when API call fails
MAX_WORKERS = 1  # max number of parallel workers
LOGGING_LEVEL = "INFO"  # logging level: DEBUG, INFO, WARNING, ERROR

PER_TASK_OUTPUT = True  # whether to save JSON files for each task separately

# control the number of data to process, set to None to process all data
PROCESS_DATA_COUNT = None  # can modify this number to control the number of data to process

# control the range of data to process
PROCESS_DATA_RANGE = {
    "start": 1, 
    "end": None,    
    "step": 1       
}

# control to run specific task numbers
TASK_NUM = ['32','64']  # only run these task numbers,such as['32'], empty list or None to run all tasks

# default model settings
NVIDIA_MODEL_NAME = "deepseek-ai/deepseek-r1"
VOLCENGINE_MODEL_NAME = "volcengine-deepseek-v3"
OPENAI_MODEL_NAME = "gpt-4o" 

# task configuration
TASKS = {
    "table_metrics": {
        "enabled": True,
        "description": "process financial table data and evaluate LLM tasks",
        "input_file": INPUT_FILE_PATH,
        "evaluation_metrics": ["precision", "recall", "f1", "number_accuracy"],
        "use_batching": True,
        "batch_size": BATCH_SIZE,  # the number of data to process in each batch
        "use_pdf_source": USE_PDF_SOURCE,  # whether to use PDF as data source
        "use_model_for_pdf": USE_MODEL_FOR_PDF,  # whether to use multimodal model to read PDF
        "use_volcengine": USE_VOLCENGINE,  # whether to use volcengine
        "use_openai": USE_OPENAI,  # use OpenAI API or not
        "task_num": TASK_NUM,  # which tasksize to run
    }
}

for task in TASKS.values():
    if "output_dir" in task:
        os.makedirs(task["output_dir"], exist_ok=True)

