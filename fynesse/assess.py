from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


import datetime
import numpy as np
from scipy.spatial import distance
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import pymysql




# ---------------------------------------------------------------------
# OSM data operations
# ---------------------------------------------------------------------


def remove_price_tails(price_df, tail_amount_cut):
    """ Removes from the price outliers from a dataframe.
    :param price_df: pandas dataframe with price column
    :param tail_amount_cut: float between 0 and 0.5 for proportion cut on each tail
    :return price_df: pandas dataframe with price column
    """
    if tail_amount_cut >= 0.5:
        raise ValueError(f"tail_amount_cut must be less than 0.5, argument value given was {tail_amount_cut}")
    if tail_amount_cut < 0:
        raise ValueError(f"tail_amount_cut must be positive, argument value given was {tail_amount_cut}")
    lower_value = price_df["price"].quantile(tail_amount_cut)
    upper_value = price_df["price"].quantile(1 - tail_amount_cut)
    print(f"lower bound: {lower_value} \nupper bound: {upper_value}")
    price_df = price_df[(price_df["price"] >= lower_value) & (price_df["price"] <= upper_value)]
    return price_df


def calculate_closest_feature(lat, long, feature_df, max_distance):
    """ Calculates the euclidean distance to the closest feature.
    :param lat: float lattitude
    :param long: float longitude
    :param feature_df: pandas dataframe of features and coordinates
    :param max_distance: float maximum distance value
    :return: float shortest distance from location to nearest feature
    """
    if len(feature_df) == 0:
        return max_distance * 2
    property_location = np.array([(lat, long)])
    feature_locations = np.array(list(zip(feature_df.lattitude, feature_df.longitude)))
    return distance.cdist(property_location, feature_locations).min(axis=1)[0]


# ---------------------------------------------------------------------
# OSM plotting
# ---------------------------------------------------------------------

def plot_category_maps(category_list, name_location, lattitude, longitude, box_width, box_height):
    """ Plots local street map around location and overlays it
        with the category features from osm present nearby.
    :param category_list: list of strings of categories
    :param name_location: string name of location
    :param lattitude: float lattitude
    :param longitude: float longitude
    :param box_width: float degree width of map
    :param box_height: float degree height of map
    """

    north, south, west, east = access.calculate_bounding_box_dimensions(lattitude, longitude, box_width, box_height)

    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(name_location)#"Cambridge, UK")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Local Map With Category Features")

    # Plot all POIs 
    for category in category_list:
        category_pois = access.fetch_pois_from_bounding_box(lattitude, longitude, box_width, box_height, access.fetch_tags(category))
        category_pois.plot(ax=ax, alpha=0.3, label=category)
    plt.tight_layout()
    plt.legend()



def plot_local_price_map(name_location, lattitude, longitude, box_width, box_height, price_df=None, date=None, days_since=365):
    """ Plots local street map around location and overlays it
        with the properties sold nearby with a colour proportional
        to their price.
    :param name_location: string name of location
    :param lattitude: float lattitude
    :param longitude: float longitude
    :param box_width: float degree width of map
    :param box_height: float degree height of map
    :param price_df: pandas dataframe pricing data
    :param date: datetime object date
    :param days_since: int days since the date to include properties from
    """
    if date is None:
        date = datetime.date(2019, 1, 1)
    if price_df is None:
        price_df = access.fetch_pp_and_pc_joined_area(lattitude, longitude, date, box_height, box_width, days_since)
    
    north, south, west, east = access.calculate_bounding_box_dimensions(lattitude, longitude, box_width, box_height)

    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(name_location)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Local Map With Prices")

    # Plot all POIs 
    price_df.plot.scatter(ax=ax, x='longitude', y='lattitude', alpha=0.3, zorder=2, c='price', colormap='viridis')
    plt.tight_layout()



def plot_UK_price_map(price_df=None, date=None, days_since=100):
    """ Plots map of UK and overlays it with the properties sold 
        with a colour proportional to their price.
    :param price_df: pandas dataframe pricing data
    :param date: datetime object date
    :param days_since: int days since the date to include properties from
    """
    UK_max_lattitude = 58.696977
    UK_min_lattitude = 49.921544
    UK_max_longitude = 1.830806
    UK_min_longitude = -10.493171
    UK_lattitude_mean = (UK_max_lattitude + UK_min_lattitude)/2
    UK_lattitude_range = abs(UK_max_lattitude - UK_min_lattitude)
    UK_longitude_mean = (UK_max_longitude + UK_min_longitude)/2
    UK_longitude_range = abs(UK_max_longitude - UK_min_longitude)

    if date is None:
        date = datetime.date(2019, 1, 1)
    if price_df is None:
        price_df = access.fetch_pp_and_pc_joined_area(UK_lattitude_mean, UK_longitude_mean, date, lat_height=UK_lattitude_range, long_width=UK_longitude_range, days_since=days_since)

    north, south, west, east = access.calculate_bounding_box_dimensions(UK_lattitude_mean, UK_longitude_mean, UK_longitude_range, UK_lattitude_range)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    world[(world.name == "United Kingdom")].plot(ax=ax, color='white', edgecolor='black')

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Property Prices Across The UK")

    price_df.plot.scatter(ax=ax, x='longitude', y='lattitude', alpha=0.1, c='price', colormap='viridis', zorder=2)
    plt.tight_layout()


def plot_type_distribution(price_df, bins=15):
    """ Plots the price distribution for different property types in
        a dataframe.
    :param price_df: pandas dataframe pricing data
    """
    detached = price_df[price_df["type"] == "D"]["price"]
    semi_detached = price_df[price_df["type"] == "S"]["price"]
    terraced = price_df[price_df["type"] == "T"]["price"]
    flat = price_df[price_df["type"] == "F"]["price"]
    other = price_df[price_df["type"] == "O"]["price"]

    plt.hist(detached, bins, alpha=0.5, label='detached')
    plt.hist(semi_detached, bins, alpha=0.5, label='semi_detached')
    plt.hist(terraced, bins, alpha=0.5, label='terraced')
    plt.hist(flat, bins, alpha=0.5, label='flat')
    plt.hist(other, bins, alpha=0.5, label='other')
    
    plt.xlabel("Price")
    plt.ylabel("No. Properties")
    plt.title("Property Type and Price Distribution")

    plt.legend(loc='upper right')
    plt.show()


# ---------------------------------------------------------------------
# Misc plotting
# ---------------------------------------------------------------------


def plot_rolling_average(df, x_col, y_col, window_size, alpha=0.1, label="", color="b", fig=None, ax=None):
    """ Plots the rolling average of a dataframe with a column respect to another column.
    :param df: pandas dataframe
    :param x_col: string column on x axis
    :param y_col: string column on y axis
    :param window_size: int window size
    :param alpha: float between 0 and 1 alpha of line plot
    :param label: string line label
    :param color: string line colour
    :param fig: matplotlib fig object figure
    :param ax: matplotlib ax object axes
    """
    rolling_price_array = df.sort_values(by=[x_col])[y_col].rolling(window=window_size).mean()
    plt.xlabel(x_col)
    plt.ylabel("rolling average price")
    plt.title(f"Rolling average price along {x_col} over the UK")
    plt.plot(df.sort_values(by=[x_col])[x_col], rolling_price_array, alpha=alpha, label=label, color=color)



def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
