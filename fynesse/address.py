
# This file contains code for suporting addressing questions in the data
from . import access
from . import assess

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# utility for prediction

def feature_column_list_from_category_list(category_list):
    """ Returns the names of features from generated from a list
        of categories.
    :param category_list: list of strings for category names
    :return feature_list: list of strings for names of features
    """
    feature_list = ["price"]
    for category in category_list:
        feature_list.append(f"n_{category}")
        feature_list.append(f"shortest_distance_{category}")
    return feature_list



def fetch_all_training_data(target_lat, target_long, feature_range_width, feature_range_height, property_range_width, property_range_height, category_list, property_type, date, days_since=365):
    """ Fetches all of the training data needed for the model using osm
        data as features.
    :param target_lat: float lattitude
    :param target_long: float longitude
    :param feature_range_width: float degree width to collect features from
    :param feature_range_height: float degree height to collect features from
    :param property_range_width: float degree width to collect properties from
    :param property_range_height: float degree height to collect properties from
    :param category_list: list of strings of categories
    :param property_type: string property type
    :param date: datetime object date
    :param days_since: int days since both sides
    :return training_df: pandas dataframe training set
    :return category_pois_dict: dictionary of string->geo dataframe category->feature dataframe
    """
    category_pois_dict = access.fetch_features_outer_bbox(target_lat, target_long, feature_range_width, feature_range_height, property_range_width, property_range_height, category_list)
    print("fetched all features")
    price_df = access.fetch_pp_and_pc_joined_area(target_lat, target_long, date, property_range_height, property_range_width, days_since)
    print("fetched pricing data")
    price_df = assess.remove_price_tails(price_df, 0.025)
    price_df = price_df[price_df["type"] == property_type]

    # every row will consist of a lat, long, price, and features
    training_df = price_df
    #print(len(training_df))
    #print(training_df.shape[0])
    #print("abc")

    for category in category_list:
        training_df[f"n_{category}"] = -1.0
        training_df[f"shortest_distance_{category}"] = -1.0
    #print(len(training_df))
    #print(training_df.shape[0])

    for category in category_list:
        for i in range(len(training_df)):
            #print(type(training_df.at[i, "lattitude"]))
            #print(training_df.at[i, "lattitude"])
            #asdasd
            #features = fetch_geo_features_nearby_from_df(training_df["lattitude"], training_df["longitude"], feature_range_width, feature_range_height, category_pois_dict[category])
            #feature_df = fetch_geo_features_nearby_from_df(float(training_df.at[i, "lattitude"]), float(training_df.at[i, "longitude"]), feature_range_width, feature_range_height, category_pois_dict[category])
            feature_df = access.fetch_geo_features_nearby_from_df(float(training_df["lattitude"].iloc[i]), float(training_df["longitude"].iloc[i]), feature_range_width, feature_range_height, category_pois_dict[category])
            training_df[f"n_{category}"].iloc[i] = len(feature_df)
            training_df[f"shortest_distance_{category}"].iloc[i] = float(assess.calculate_closest_feature(float(training_df["lattitude"].iloc[i]), float(training_df["longitude"].iloc[i]), feature_df, feature_range_width*(2**(1/2))))
            #training_df.at[i, f"n_{category}"] = len(feature_df)
            #training_df.at[i, f"shortest_distance_{category}"] = float(calculate_closest_feature(float(training_df.at[i, "lattitude"]), float(training_df.at[i, "longitude"]), feature_df, feature_range_width*(2**(1/2))))
            #print(f"added features for category: {category}")
            #print(len(training_df))
            #print(training_df.shape[0])

    return training_df, category_pois_dict



def fetch_prediction_data(target_lat, target_long, feature_range_width, feature_range_height, category_pois_dict, date, days_since=365):
    """ Fetches all of the training data needed for the model using osm
        data as features.
    :param target_lat: float lattitude
    :param target_long: float longitude
    :param feature_range_width: float degree width to collect features from
    :param feature_range_height: float degree height to collect features from
    :param category_pois_dict: dictionary of string->geo dataframe category->feature dataframe
    :param date: datetime object date
    :param days_since: int days since both sides
    :return prediction_df: pandas dataframe prediction set
    """
    #category_pois_dict = {}

    #for category in category_list:
    #  category_pois = fetch_pois_from_bounding_box(target_lat, target_long, feature_range_width, feature_range_height, fetch_tags(category))
    #  pois_coords_df = fetch_pois_coordinates(category_pois)[["lattitude", "longitude"]]
    #  category_pois_dict[category] = pois_coords_df.dropna()
    

    pred_row = {}

    for category in category_pois_dict.keys():
        feature_df = access.fetch_geo_features_nearby_from_df(float(target_lat), float(target_long), feature_range_width, feature_range_height, category_pois_dict[category])
        pred_row[f"n_{category}"] = len(feature_df)
        pred_row[f"shortest_distance_{category}"] = float(assess.calculate_closest_feature(float(target_lat), float(target_long), feature_df, feature_range_width*(2**(1/2))))

    prediction_df = pd.DataFrame(pred_row, index=[0])

    return prediction_df


def prepare_training_data(pred_lattitude, pred_longitude, property_type, date, category_list, days_since=365, property_box_length=0.08, feature_box_length=0.02, training_df=None, category_pois_dict=None):
    """ Fetches all of the training data needed for model training and augments
        it into the form which is required for training.
    :param pred_lattitude: float lattitude
    :param pred_longitude: float longitude
    :param property_type: string property type
    :param date: datetime object date
    :param category_list: list of strings of categories
    :param days_since: int days since both sides
    :param property_box_length: float degree width to collect properties from
    :param feature_box_length: float degree width to collect features from
    :param training_df: pandas dataframe training set
    :param category_pois_dict: dictionary of string->geo dataframe category->feature dataframe
    :return ys: numpy float array of prices 
    :return xs: numpy float array of features
    :return training_df: pandas dataframe training set
    :return category_pois_dict: dictionary of string->geo dataframe category->feature dataframe
    """
    if ((training_df is None) or (category_pois_dict is None)):  
        training_df, category_pois_dict = fetch_all_training_data(pred_lattitude, pred_longitude, feature_box_length, feature_box_length, property_box_length, property_box_length, category_list, property_type, date, days_since=days_since)
        training_df = training_df[feature_column_list_from_category_list(category_list)]
    ys = np.array(training_df["price"], dtype=float)
    xs = np.array(training_df.drop("price", axis=1), dtype=float)
    return ys, xs, training_df, category_pois_dict


def fit_OLS_model(ys, xs, training_df):
    """ Fits an Ordinary Least Squares model to training data and returns
        the fitted model.
    :param ys: numpy float array of prices 
    :param xs: numpy float array of features
    :param training_df: pandas dataframe training set
    :return fit_model: statsmodels results fitted model
    """
    xs = sm.add_constant(xs)
    linear_model = sm.OLS(ys, xs)
    fit_model = linear_model.fit()
    return fit_model


def fit_regularised_OLS_model(ys, xs, training_df, alpha=1, L1_wt=1):
    """ Fits an Ordinary Least Squares model to training data with an L1
        regularisation (Lasso regression) constraint and returns the fitted model.
    :param ys: numpy float array of prices 
    :param xs: numpy float array of features
    :param alpha: float positive for penalty from L1 regularisation
    :param L1_wt: float between 0 and 1 weighting towards L1 regularisation vs L2
    :return fit_model: statsmodels reguralised results fitted model
    """
    xs = sm.add_constant(xs)
    linear_model = sm.OLS(ys, xs)
    fit_model = linear_model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
    return fit_model


def validate_OLS_model(validation_df, fit_model):
    """ Validates the performance of a fitted model. Provides graph plots and 
        summary statistics on its performance.
    :param validation_df: pandas dataframe validation data
    :param fit_model: statsmodels results fitted model
    """
    validation_df = validation_df.sort_values(by=['price'])
    print(f"fetched {len(validation_df)} properties for this location, check this isnt too low, if it is: prediction = N.A.")
    t_ys = np.array(validation_df["price"], dtype=float)
    t_xs = np.array(validation_df.drop("price", axis=1), dtype=float)
    t_xs = sm.add_constant(t_xs)
    t_pred_ys = fit_model.get_prediction(t_xs).summary_frame(alpha=0.05)
    data_point_indexes = range(len(t_ys))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(data_point_indexes, t_ys, c='b', label='actual', alpha=1)
    ax.scatter(data_point_indexes, t_pred_ys["mean"], c='r', label='prediction', alpha = 0.3)
    ax.plot(data_point_indexes, t_pred_ys["obs_ci_lower"], linestyle="-", c='r', label='prediction lower confidence', alpha = 0.5)
    ax.plot(data_point_indexes, t_pred_ys["obs_ci_upper"], linestyle="-", c='r', label='prediction upper confidence', alpha = 0.5)
    ax.fill_between(data_point_indexes, t_pred_ys["obs_ci_lower"], t_pred_ys["obs_ci_upper"], color="red", alpha=0.1)
    plt.xlabel("Properties Sorted By Price")
    plt.ylabel("Price")
    plt.title("Validation Performance")
    plt.legend()
    plt.show()
    print(fit_model.summary())



def predict_with_OLS_model(fit_model, prediction_df):
    """ Validates the performance of a fitted model. Provides graph plots and 
        summary statistics on its performance.
    :param fit_model: statsmodels results fitted model
    :param prediction_df: pandas dataframe of features to predict from
    :return price_prediction: numpy array of predictions
    """
    pred_xs = np.array(prediction_df)
    pred_xs = np.insert(pred_xs, 0, 1)
    print(pred_xs)
    price_prediction = fit_model.get_prediction(pred_xs).summary_frame(alpha=0.05)
    print(price_prediction)
    return price_prediction


def prediction_ols(pred_lattitude, pred_longitude, property_type, date, category_list, days_since=365, property_box_length=0.08, feature_box_length=0.02, training_df=None, category_pois_dict=None, perform_prediction=True):
    """ Shows training performance and also has capacity to perform predictions on other data points
        if provided.
    :param pred_lattitude: float lattitude
    :param pred_longitude: float longitude
    :param property_type: string property type
    :param date: datetime object date
    :param category_list: list of strings of categories
    :param days_since: int days since both sides
    :param property_box_length: float degree width to collect properties from
    :param feature_box_length: float degree width to collect features from
    :param training_df: pandas dataframe training set
    :param category_pois_dict: dictionary of string->geo dataframe category->feature dataframe
    :param perform_prediction: boolean perform prediction
    :return fit_model: statsmodels results fitted model
    :return price_prediction: (optional) numpy float array of features
    """
    ys, xs, training_df, category_pois_dict = prepare_training_data(pred_lattitude, pred_longitude, property_type, date, category_list, days_since=days_since, property_box_length=property_box_length, feature_box_length=feature_box_length, training_df=training_df, category_pois_dict=category_pois_dict)
    fit_model = fit_OLS_model(ys, xs, training_df)
    validation_df = training_df
    validate_OLS_model(validation_df, fit_model)
    if perform_prediction:
        prediction_df = fetch_prediction_data(pred_lattitude, pred_longitude, feature_box_length, feature_box_length, category_pois_dict, date, days_since)
        price_prediction = predict_with_OLS_model(fit_model, prediction_df)
        return price_prediction, fit_model
    return fit_model


def prediction_ols_L1_regularised(pred_lattitude, pred_longitude, property_type, date, category_list, days_since=365, property_box_length=0.08, feature_box_length=0.02, training_df=None, category_pois_dict=None, perform_prediction=True, alpha=1, L1_wt=1):
    """ Shows training performance and reveals the params L1 regularisation has prioritised.
    :param pred_lattitude: float lattitude
    :param pred_longitude: float longitude
    :param property_type: string property type
    :param date: datetime object date
    :param category_list: list of strings of categories
    :param days_since: int days since both sides
    :param property_box_length: float degree width to collect properties from
    :param feature_box_length: float degree width to collect features from
    :param training_df: pandas dataframe training set
    :param category_pois_dict: dictionary of string->geo dataframe category->feature dataframe
    :param perform_prediction: boolean perform prediction
    :param alpha: float positive for penalty from L1 regularisation
    :param L1_wt: float between 0 and 1 weighting towards L1 regularisation vs L2
    :return fit_model: statsmodels reguralised results fitted model
    """
    ys, xs, training_df, category_pois_dict = prepare_training_data(pred_lattitude, pred_longitude, property_type, date, category_list, days_since=days_since, property_box_length=property_box_length, feature_box_length=feature_box_length, training_df=training_df, category_pois_dict=category_pois_dict)
    fit_model = fit_regularised_OLS_model(ys, xs, training_df, alpha=alpha, L1_wt=L1_wt)
    print(f"Parameters for regularised model are: {fit_model.params}")
    #validation_df = training_df
    #validate_OLS_model_regularised(validation_df, fit_model)
    #if perform_prediction:
    #  prediction_df = fynesse.address.fetch_prediction_data(pred_lattitude, pred_longitude, feature_box_length, feature_box_length, category_pois_dict, date, days_since)
    #  price_prediction = predict_with_OLS_model(fit_model, prediction_df)
    #  return price_prediction, fit_model
    return fit_model