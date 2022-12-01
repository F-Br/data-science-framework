from .config import *

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