"""
Author  : Vian Sebastian B 
Version : 1
Date    : 4/11/2024

"main_pipeline.py" 
This module initializes and runs a TFX pipeline for a stress detection model based on sleep quality. 

Key Components: 
- Data ingestion and preprocessing 
- Model transformation, tuning, and training
- Model Evaluation and serving

Usage: 
- Serves as the main script for pipeline execution
"""

# import dependencies
import os
from absl import logging
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import init_components
from modules.pipeline import init_local_pipeline

# initialize constants
# pipeline name
SCHEMA_PIPELINE_NAME = 'sleep-stress-tfdv-schema'
PIPELINE_NAME = 'sleep-stress-pipeline'

# pipeline schema and root directory
SCHEMA_PIPELINE_ROOT = os.path.join('pipelines', SCHEMA_PIPELINE_NAME)
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)

# metadata schema and path directory
SCHEMA_METADATA_PATH = os.path.join('metadata', SCHEMA_PIPELINE_NAME, 'metadata.db')
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

# model serving directory
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

# data and module paths
DATA_ROOT = 'dataset'
TRANSFORM_MODULE_FILE = 'modules/stress_sleep_transform.py'
TUNER_MODULE_FILE = 'modules/stress_sleep_tuner.py'
TRAINER_MODULE_FILE = 'modules/stress_sleep_trainer.py'

# define component args dictionary
components_args = {
    "data_dir": DATA_ROOT,
    "transform_module": TRANSFORM_MODULE_FILE,
    "tuner_module": TUNER_MODULE_FILE,
    "train_module": TRAINER_MODULE_FILE,
    "train_steps": 1000,
    "eval_steps": 800,
    "serving_model_dir": SERVING_MODEL_DIR,
}

logging.set_verbosity(logging.INFO)

# initialize components
components = init_components(components_args)

# create pipeline
pipeline = init_local_pipeline(
    components,
    PIPELINE_ROOT,
    PIPELINE_NAME,
    METADATA_PATH)
BeamDagRunner().run(pipeline=pipeline)
