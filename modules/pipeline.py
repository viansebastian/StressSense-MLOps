"""
Author  : Vian Sebastian B
Version : 1
Date    : 4/11/2024

"pipeline.py"
This module instantiates a TFX-BEAM pipeline.

Usage:
- Serves to instantiate a TFX-BEAM pipeline
"""

# import dependencies
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline

# pipeline initialization function
def init_local_pipeline(
        components,
        pipeline_root: Text,
        pipeline_name,
        metadata_path):
    """
    Initializes a local TFX pipeline for execution with Apache Beam.

    Args:
        components    (list): A list of TFX components that make up the pipeline.
        pipeline_root (Text): The root directory where the pipeline's output will be stored.
        pipeline_name (Text): The name of the pipeline.
        metadata_path (Text): The path to the SQLite database for metadata storage.

    Returns:
        pipeline.Pipeline: The instantiated TFX pipeline object configured for local execution.
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )
