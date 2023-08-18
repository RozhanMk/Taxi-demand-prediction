"""
Description:
    Achieve Latent space by mapping the number of pickups in timestamp  and location to the number of drop-offs.
"""

"""Import the most essential packages"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input ,Dense, Flatten, Add, Activation

"""Ignore the packages' warnings """
warnings.filterwarnings('ignore')


def reshape_data(df, locations=263, location='PULocationID'):
    """Reshape the dataframe into a 2D numpy array.

    Args:
        df (pd.DataFrame): Input dataframe.
        locations (int): Total number of location columns (default is 263).
        location (str): Column name to consider for location (default is 'PULocationID').

    Returns:
        numpy.ndarray: Reshaped 2D numpy array.
    """
    # Filter out rows with specific location IDs (265 and 264 in this case)
    temp_df = df.loc[~((df[location] == 265) | (df[location] == 264))]

    # Drop the 'location' column from the dataframe
    temp_df.drop([location], axis=1, inplace=True)

    # Convert the filtered dataframe to a numpy array and reshape it
    values = temp_df.to_numpy()
    values = np.reshape(values, (-1, locations))
    return values


def make_latent_location_features(pu_dir, do_dir, latent_space=7, epochs=1000, learning_rate=0.0001, verbose=False):
    """Preprocess the data and train a latent model to extract location features.

    Args:
        pu_dir (str): File path for the data related to PULocationID.
        do_dir (str): File path for the data related to DOLocationID.
        verbose (bool): If True, display the summary of the model (default is True).

    Returns:
        pd.DataFrame: DataFrame containing the extracted location features.
    """
    # Read the data from Parquet files into dataframes
    df_p = pd.read_parquet(pu_dir)
    df_d = pd.read_parquet(do_dir)

    # Sorting dataframes based on timestamp and location IDs
    df_p = df_p.sort_values(['timestamp', 'PULocationID'])
    df_d = df_d.sort_values(['timestamp', 'DOLocationID'])

    # Set 'timestamp' column as the index for both dataframes
    df_p.set_index('timestamp', inplace=True)
    df_d.set_index('timestamp', inplace=True)

    # Reshape the data for input and output
    data_in = reshape_data(df_p)
    data_out = reshape_data(df_d, location='DOLocationID')[:-1]

    # Get the input shape for the neural network
    input_shape = np.shape(data_in)[1:]

    # Define the input layer of the neural network
    input_ = Input(input_shape)

    # Build the neural network architecture
    x = Dense(latent_space)(input_)
    x = Dense(263)(x)
    latent_model = Model(inputs=input_, outputs=x)

    # Display the summary of the model if verbose is True
    if verbose:
        latent_model.summary()

    # Use Adam as the optimizer with a learning rate of 1e-6 (2.5e-3 in the comment seems incorrect)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Use Huber loss function and RMSE and MAPE as evaluation metrics
    latent_model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error'])

    # Train the model
    latent_model_history = latent_model.fit(data_in, data_out, epochs=epochs, verbose=verbose)

    # Get the weights of the trained model
    weights = latent_model.get_weights()

    # Extract location features from the weights
    locations_feature = weights[2].T

    # Create column names for location features
    locations_feature_columns = [f'lf{i+1}' for i in range(np.shape(locations_feature)[1])]

    # Create a dataframe for location features with corresponding column names
    locations_df = pd.DataFrame(locations_feature, columns=locations_feature_columns)

    # Create a list of locations from 1 to 263 (total locations)
    locations = [i for i in range(1, 264)]

    # Add the 'location' column to the dataframe with location IDs
    locations_df['PULocationID'] = locations

    return locations_df


if __name__ == '__main__':
    pu_dir = "../Demand/demand_based_on_PU.parquet"
    do_dir = "../content/Demand/demand_based_on_DO.parquet"
    location_features_df = make_latent_location_features(pu_dir, do_dir, verbose=True)
    location_features_df.to_parquet('../artifacts/locations_latent_features.parquet')






