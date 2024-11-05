# pylint: disable=no-member
"""
Author  : Vian Sebastian B
Version : 1
Date    : 4/11/2024

"components.py "
This module contains all components constructing the TFX Pipeline.

Key Components:
- Ingestion
- Validation
- Preprocessing
- Analysis
- Interpretation
- Tuning and Training
- Pushing

Usage:
- Serves as the components base of the pipeline
"""

# import dependencies
import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)

# main initialization function


def init_components(args):
    """
    Initializes and returns the components required to construct a TFX pipeline.

    Args:
        args (dict): A dictionary containing the following keys:
            - 'data_dir'          (str): Path to the directory containing input data.
            - 'transform_module'  (str): Path to the transform module.
            - 'tuner_module'      (str): Path to the tuning module.
            - 'train_module'      (str): Path to the training module.
            - 'train_steps'       (int): Number of steps for training.
            - 'eval_steps'        (int): Number of steps for evaluation.
            - 'serving_model_dir' (str): Directory where the trained model 
                                        will be pushed for serving.

    Returns:
        Tuple: A tuple containing the TFX components:
            - example_gen: Ingests the raw CSV data and splits it into training and evaluation sets.
            - statistics_gen: Generates statistics for the ingested examples.
            - schema_gen: Infers the schema of the data using the generated statistics.
            - example_validator: Validates the data against the inferred schema to detect anomalies.
            - transform: Perform preprocessing based on transform module.
            - tuner: Tunes the model based on the tuner module.
            - trainer: Fetch tuned params and perform training 
                        based on the training module
            - model_resolver: Resolves the latest blessed model for comparison during evaluation.
            - evaluator: Evaluates the trained model using the provided evaluation configuration.
            - pusher: Pushes the blessed model to the specified serving directory.

    """
    # exampleGen component
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
        ])
    )
    example_gen = CsvExampleGen(
        input_base=args['data_dir'],
        output_config=output)

    # statisticsGen component
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # schemaGen component
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )

    # exampleValidator component
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # transform component
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(args['transform_module'])
    )

    # tuner component
    tuner = Tuner(
        module_file=os.path.abspath(args['tuner_module']),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=args['train_steps']
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=args['eval_steps']
        )
    )

    # trainer component
    trainer = Trainer(
        module_file=os.path.abspath(args['train_module']),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=args['train_steps']
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=args['eval_steps']
        )
    )

    # resolver component
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # evaluator component
    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['stress_rate_xf'])
    ]

    metrics_specs = [
        tfma.MetricsSpec(per_slice_thresholds={
            'accuracy':
                tfma.PerSliceMetricThresholds(thresholds=[
                    tfma.PerSliceMetricThreshold(
                        slicing_specs=[tfma.SlicingSpec()],
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.6}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10}
                            )
                        )
                    )
                ])
        }),

        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='Precision'),
            tfma.MetricConfig(class_name='Recall'),
            tfma.MetricConfig(class_name='ExampleCount'),
        ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='stress_rate_xf')
        ],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # pusher component
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=args['serving_model_dir']
            )
        )
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
