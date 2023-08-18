import numpy as np
import pandas as pd


def prepare_dataset(df, training=True)->pd.DataFrame:
    """
    Process the dataset by reading from a Parquet file, filtering rows with specific PULocationID values,
    adding a 'target' column, and removing unnecessary columns.

    Parameters:
        dataset_dir (str): The directory path to the Parquet file.
        latent_features (bool): Add locations' latent space to the dataset
    Returns:
        pandas.DataFrame: The processed DataFrame with the 'target' column.
    """

    df = df.reset_index()

    # Step 1: Filter out rows with PULocationID equal to 265 or 264.
    df = df.loc[~((df['PULocationID'] == 265) | (df['PULocationID'] == 264))]

    # Step 2: Get the list of column names in the DataFrame.
    columns = df.columns.to_list()

    # Step 3: Add a 'target' column and initialize it with 0.
    df['target'] = 0

    # Step 4: Sort the DataFrame by 'timestamp' and 'PULocationID'.
    df.sort_values(['timestamp', 'PULocationID'], inplace=True)

    # Step 5: Calculate the 'target' values for each PULocationID using shift(-1) function.
    for i in range(1, 264):
        temp = df[df.PULocationID == i]
        df['target'][df.PULocationID == i] = temp.demand.shift(-1)

    if training==True:
        # Step 6-1: Drop rows with NaN values in the DataFrame.
        df.dropna(inplace=True)
    else:
        # Step 6-2: When processing data for forecasting, only the latest timestamp record for each district should be utilized.
        df = df[df['target'].isnull()]
        df.sort_values(['timestamp', 'PULocationID'], inplace=True)

    # Step 7: Drop the 'PULocationID' column as it is no longer needed.
    df = df.drop(['PULocationID'], axis=1)

    # Return the processed DataFrame.
    return df




def separate_features_and_target(processed_array):
    """
    Separates the features and the target from the processed_array.

    Parameters:
        processed_array (numpy.ndarray): The input array containing both features and target.

    Returns:
        numpy.ndarray: The features array.
        numpy.ndarray: The target array.
    """
    features_set = processed_array[:, :-1]
    target_set = np.reshape(processed_array[:, -1], (-1, 1))
    return features_set, target_set

