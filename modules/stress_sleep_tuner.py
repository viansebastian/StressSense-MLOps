"""
Author  : Vian Sebastian B
Version : 1
Date    : 4/11/2024

"stress_sleep_tuner.py "
This module contains tuning components for the model.

Key Components:
- Get inputs
- Instantiate Keras Tuner
- Tunes the model

Usage:
- Serves as the model tuning script
"""

# import dependencies
from typing import Any, Dict, NamedTuple, Text
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from keras_tuner.engine import base_tuner
from stress_sleep_transform import transformed_name, LABEL_KEY, FEATURE_KEYS

LABEL_KEY = 'stress_rate'
FEATURE_KEYS = [
    'snoring_rate', 'respiration_rate', 'body_temp', 'limb_movement',
    'blood_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate'
]

NUM_EPOCHS = 5

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    patience=10,
)


def gzip_reader_fn(filenames):
    """Loads compressed data from TFRecord files.

    Args:
        filenames (Text): The pattern of TFRecord files to read.

    Returns:
        tf.data.Dataset: A dataset object for the TFRecord files.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=32):
    """Creates a batched dataset from the transformed features.

    Args:
        file_pattern        (Text) : The pattern for input files.
        tf_transform_output        : The transformed output from TensorFlow Transform.
        num_epochs          (int)  : The number of epochs for which to iterate over the dataset.
        batch_size          (int)  : The size of the batches to create.

    Returns:
        tf.data.Dataset: A batched dataset of transformed features and labels.
    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        shuffle=False,
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset


def get_model_tuner(params):
    """Builds and tunes the Keras model based on hyperparameters.

    Args:
        params: The hyperparameters provided by Keras Tuner.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    input_features = []
    for feature in FEATURE_KEYS:
        input_features.append(
            tf.keras.layers.Input(shape=(1,), name=transformed_name(feature))
        )

    dense_units = params.Int(
        'units',
        min_value=32,
        max_value=512,
        step=32
    )

    dropout_rate = params.Float(
        'drop_rate',
        min_value=0.0,
        max_value=0.5,
        step=0.1
    )

    learning_rate = params.Choice(
        'lr',
        values=[
            0.001,
            0.0001,
            0.00001
        ]
    )

    inputs = tf.keras.layers.concatenate(input_features)
    layer_1 = tf.keras.layers.Dense(dense_units, activation='relu')(inputs)
    dropout_1 = tf.keras.layers.Dropout(dropout_rate)(layer_1)
    layer_2 = tf.keras.layers.Dense(dense_units, activation='relu')(dropout_1)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(layer_2)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def tuner_fn(fn_args):
    """Tunes the model to find the best hyperparameters.

    Args:
        fn_args: The arguments containing input and tuning specifications.

    Returns:
        TunerFnResult: A named tuple containing the tuner and fitting arguments.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5)

    tuner = kt.RandomSearch(
        get_model_tuner,
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='kt_tuner'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "callbacks": [early_stop]
        },
    )
