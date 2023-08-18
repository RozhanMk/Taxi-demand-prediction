import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
from preprocessing import prepare_dataset, separate_features_and_target
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input ,Dense, Flatten, Add, Activation

def train(input_shape, X_train, Y_train, epochs=10, batch_size=128, learning_rate=0.0001):
    """
    Create and train a demand prediction model using a specified architecture and evaluation metrics.

    Parameters:
        input_shape (tuple): The shape of the input data.
        X_train (numpy.ndarray): Training feature data.
        Y_train (numpy.ndarray): Training label data.
        epochs (int, optional): Number of training epochs. Default is 10.
        batch_size (int, optional): Batch size for training. Default is 128.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 0.0001.

    Returns:
        tensorflow.keras.Model: The trained demand prediction model.
        dict: The training history containing loss and metric values during training.
    """
    # Step 1: Define the model architecture
    input_ = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(input_)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input_, outputs=x)

    # Step 2: Define the optimizer with the given learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Step 3: Compile the model with the specified loss and evaluation metrics
    model.compile(
        loss=tf.keras.losses.Huber(),  # Use Huber loss function
        optimizer=optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
    )

    # Step 4: Train the model with the provided training data
    model_history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # Turn off verbosity during training
    )

    return model, model_history


def retrain(model, X_train, Y_train, epochs=10, batch_size=128, learning_rate=0.0001):
    """
    Retrain a demand prediction model.

    Parameters:
        model: The model will be retrained with the new data.
        X_train (numpy.ndarray): Training feature data.
        Y_train (numpy.ndarray): Training label data.
        epochs (int, optional): Number of training epochs. Default is 10.
        batch_size (int, optional): Batch size for training. Default is 128.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 0.0001.

    Returns:
        tensorflow.keras.Model: The trained demand prediction model.
        dict: The training history containing loss and metric values during training.
    """

    # Step 1: Define the optimizer with the given learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Step 3: Compile the model with the specified loss and evaluation metrics
    model.compile(
        loss=tf.keras.losses.Huber(),  # Use Huber loss function
        optimizer=optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
    )

    # Step 4: Train the model with the provided training data
    model_history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0  # Turn off verbosity during training
    )

    return model, model_history


def save_model(model, model_history, model_name='default'):
    """
    Save the trained model and its training history.

    Args:
        model (tf.keras.Model): The trained model to be saved.
        model_history (History): The training history of the model.
        model_name (str): Name of the model. Default is 'default'.

    Returns:
        None
    """

    dir = '../artifacts/'
    model_dir = os.path.join(dir, model_name)

    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model in the TensorFlow SavedModel format
    model_path = os.path.join(model_dir, model_name + '.keras')
    model.save(model_path)

    # Save the model history using pickle
    history_path = os.path.join(model_dir, model_name + '_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(model_history, f)



def load_model(model_name, load_history=False):
    """
    Load a trained Keras model and optionally its training history.

    Args:
        model_name (str): Name of the model to be loaded.
        load_history (bool, optional): Whether to load training history. Default is False.

    Returns:
        model (tf.keras.Model or None): Loaded Keras model. None if model file doesn't exist.
        history (dict or None): Loaded training history if 'load_history' is True, otherwise None.
    """
    model = None
    history = None

    # Construct paths for the model and history files
    model_dir = f'./artifacts/{model_name}'
    model_path = os.path.join(model_dir, f'{model_name}.keras')
    history_path = os.path.join(model_dir, f'{model_name}_history.pkl')

    # Load the model if it exists
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

        # Load training history if requested
        if load_history:
            with open(history_path, 'rb') as file:
                history = pickle.load(file)

    return model, history



def forecast(model_name: str, batch: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Generate predictions using a specified model.

    Args:
        model_name (str): The name of the model to be loaded.
        batch (pd.DataFrame): The input data batch for prediction.

    Returns:
        Tuple[np.array, np.array]: A tuple containing two numpy arrays.
            - The first array contains the predicted values.
            - The second array contains the corresponding target values.
    """
    # Load the specified model
    models = load_model(model_name)
    
    # Select the first model from the list
    selected_model = models[0]

    # Check if the selected model exists
    assert selected_model is not None, f"Model '{model_name}' not found"

    # Prepare the batch data for prediction
    processed_df = prepare_dataset(batch, training=False)
    processed_array = processed_df.to_numpy()
    
    # Separate features and target variables from the processed data
    features_set, target_set = separate_features_and_target(processed_array)
    
    # Generate predictions using the selected model
    predictions = selected_model.predict(features_set, verbose=0)
    
    # Return the predicted values and the corresponding target values
    return predictions, target_set

