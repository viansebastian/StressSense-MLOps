"""
Author  : Vian Sebastian B
Version : 1
Date    : 4/11/2024

"stress_sleep_transform.py "
This module contains transform components for the data.

Key Components:
- Transform names
- Perform label encoding
- Perform feature normalization
- Provide label mapping results

Usage:
- Serves as the data transform script
"""

# import dependencies
import tensorflow as tf
import tensorflow_transform as tft

# label-feature constants
LABEL_KEY = 'stress_rate'
FEATURE_KEYS = [
    'snoring_rate', 'respiration_rate', 'body_temp', 'limb_movement',
    'blood_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate'
]


def transformed_name(feature_key):
    """Renames transformed features by appending a suffix.

    Args:
        feature_key (str): The original feature name.

    Returns:
        str: The transformed feature name with '_xf' appended.
    """
    return feature_key + '_xf'


def preprocessing_fn(inputs):
    """Preprocesses input features for TensorFlow Transform.

    This function performs label encoding for the stress_rate label
    and normalizes the feature inputs using z-score scaling.

    Args:
        inputs (dict): A dictionary of input feature tensors where keys
                       are feature names and values are the corresponding
                       tensors.

    Returns:
        dict: A dictionary of transformed features where keys are transformed
              feature names and values are the corresponding tensors after
              processing.
    """
    # define keys
    encoding_keys = [
        'high/unhealthy',
        'low/normal',
        'medium',
        'medium_high',
        'medium_low']

    # encoding
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=encoding_keys,
        values=tf.cast(tf.range(len(encoding_keys)), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64
    )
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)

    stress_rate = inputs[LABEL_KEY]
    stress_rate_encoded = table.lookup(stress_rate)

    # normalization
    outputs = {}
    for feature_key in FEATURE_KEYS:
        outputs[transformed_name(feature_key)] = tft.scale_to_z_score(inputs[feature_key])

    # add the encoded label to the outputs
    outputs[transformed_name(LABEL_KEY)] = stress_rate_encoded

    return outputs


# print mapping
mapping_keys = [
    'high/unhealthy',
    'low/normal',
    'medium',
    'medium_high',
    'medium_low']
for index, key in enumerate(mapping_keys):
    print(f"'{key}' is encoded as {index}")
