
"""
    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # Replace null_values with zero
    df['Valencia_pressure'] = df['Valencia_pressure'].fillna(0)
    # Extract the numbers from the string
    df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.extract('(\d+)')
    
    # Transform the numbers to numeric data type
    df['Valencia_wind_deg'] = pd.to_numeric(df['Valencia_wind_deg'])
    # Extract the numbers from the string
    df['Seville_pressure'] = df['Seville_pressure'].str.extract('(\d+)')
    
    # Transform the numbers to numeric data type
    df['Seville_pressure'] = pd.to_numeric(df['Seville_pressure'])
    # transform to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # extract features from the time column and insert them at the begining of the dataframe 
    df.insert(1, 'Year', df['time'].dt.year)
    df.insert(2, 'Month', df['time'].dt.month)
    df.insert(4, 'Day', df['time'].dt.day)
    df.insert(5, 'Weekday', df['time'].dt.weekday + 1)
    df.insert(6, 'Hour', df['time'].dt.hour)
    #drop unwanted noise
    identifiers = ['Unnamed: 0', 'Barcelona_weather_id', 'Madrid_weather_id', 'Seville_weather_id', 'Bilbao_weather_id']
    multicol_features = ['Madrid_temp_min', 'Seville_temp_min', 'Barcelona_temp_min', 'Bilbao_temp_min', 'Valencia_temp_min', 
                   'Bilbao_temp_max', 'Madrid_temp_max', 'Barcelona_temp_max', 'Valencia_temp_max', 'Seville_temp_max',
                   'Seville_clouds_all', 'Bilbao_clouds_all', 'Madrid_clouds_all']
    one_hourly = ['Bilbao_rain_1h', 'Seville_rain_1h', 'Madrid_rain_1h', 'Barcelona_rain_1h', 'time']
    df = df.drop(identifiers + multicol_features + one_hourly, axis = 1)

    
    # ------------------------------------------------------------------------

    return df

def load_model(path_to_model:str):
    
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.
      
        

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
