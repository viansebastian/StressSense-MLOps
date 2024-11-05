"""
Author  : Vian Sebastian B
Version : 1
Date    : 4/11/2024

"stress_sleep_trainer.py "
This module contains training components for the data.

Key Components:
- Reads and parses files
- Build the model from tuned params
- Train, save, and serve the model

Usage:
- Serves as the model training script
"""

# import dependencies
import os
import tensorflow as tf
import tensorflow_transform as tft
from stress_sleep_transform import transformed_name
from stress_sleep_tuner import input_fn

# label-feature constants
LABEL_KEY = 'stress_rate'
FEATURE_KEYS = [
    'snoring_rate', 'respiration_rate', 'body_temp', 'limb_movement',
    'blood_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate'
]


def gzip_reader_fn(filenames):
    """Loads compressed data from TFRecord files.

    Args:
        filenames (str or list of str): The path or list of paths to the TFRecord files to read.

    Returns:
        tf.data.TFRecordDataset: A dataset object for the TFRecord files with GZIP compression.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example for serving.

    Args:
        model (tf.keras.Model): The trained Keras model to be used for inference.
        tf_transform_output (tft.TFTransformOutput): The output of TensorFlow Transform used for
                                                        feature transformation.

    Returns:
        function: A function that takes serialized tf.Example and returns model predictions.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Parses serialized tf.Example and makes predictions using the model.

        Args:
            serialized_tf_examples (tf.Tensor): A batch of serialized tf.Example.

        Returns:
            dict: A dictionary containing the model's output predictions.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def get_model(params):
    """Builds and compiles the Keras model based on the given hyperparameters.

    Args:
        params (dict): A dictionary of hyperparameters used to configure the model.

    Returns:
        tf.keras.Model: The compiled Keras model ready for training.
    """
    print(params)
    input_features = []
    for feature in FEATURE_KEYS:
        input_features.append(
            tf.keras.layers.Input(shape=(1,), name=transformed_name(feature))
        )

    inputs = tf.keras.layers.concatenate(input_features)
    layer_1 = tf.keras.layers.Dense(params['units'], activation='relu')(inputs)
    dropout_1 = tf.keras.layers.Dropout(params['drop_rate'])(layer_1)
    layer_2 = tf.keras.layers.Dense(
        params['units'], activation='relu')(dropout_1)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(layer_2)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def run_fn(fn_args):
    """Trains the model using the TFX Trainer.

    This function reads the training and evaluation data, constructs the model,
    and trains it based on the specified hyperparameters. It also sets up various
    callbacks for monitoring and saves the trained model.

    Args:
        fn_args (Namespace): An object containing input arguments including file paths,
                             transformation graph path, hyperparameters, and
                             serving model directory.

    Returns:
        None: The function saves the trained model to the specified directory and
        prints a message upon completion.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=10)
    eval_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        num_epochs=5)

    params = fn_args.hyperparameters['values']
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq="batch"
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        patience=10
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(fn_args.serving_model_dir, 'best_model.h5'),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True
    )

    callbacks = [
        tensorboard_callback,
        early_stop_callback,
        model_checkpoint_callback]

    model = get_model(params)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=params.get("tuner/initial_epoch", 10),
        verbose=1
    )

    signatures = {
        'serving_default': get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))}
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures)
    print("Model training complete and saved to:", fn_args.serving_model_dir)
