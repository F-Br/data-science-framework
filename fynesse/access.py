from .config import *


"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

# https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads#using-or-publishing-our-price-paid-data for information on data rights with property price data
# https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/  http://www.ordnancesurvey.co.uk/docs/licences/os-opendata-licence.pdf   postcode data license
# 


import datetime
import numpy as np
from scipy.spatial import distance
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import pymysql

import yaml
from ipywidgets import interact_manual, Text, Password


@interact_manual(username=Text(description="Username:"), 
                 password=Password(description="Password:"))
def write_credentials(username, password):
    """ Asks for credentials to be used for AWS with MariaDB,
        and stores them on a temporary local file.
    :param username: string username
    :param password: string password
    """
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username, 
                            'password': password}
        yaml.dump(credentials_dict, file)

def fetch_database_credentials():
    """ Uses the stored credentials for AWS with MariaDB to fetch
        the username, the password, and the URL to the database. 
    :return username: string username
    :return password: string password
    :return url: string url
    """
    database_details = {"url": "database-fb531.cgrre17yxw11.eu-west-2.rds.amazonaws.com", 
                      "port": 3306}

    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = database_details["url"]
    return username, password, url

usn, pwd, url = fetch_database_credentials()
db_name = "assessment_db"

def create_connection(database=db_name, user=usn, password=pwd, host=url, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: string username
    :param password: string password
    :param host: string host url
    :param database: string database name
    :param port: int port number
    :return conn: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database,
                               autocommit=True)
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def upload_csv_to_db(file, table):
    """ Uploads a csv file to a table in the database.
    :param file: string csv file name
    :param table: string table database table name
    """
    conn = create_connection()
    cur = conn.cursor()
    query = f"""
            LOAD DATA LOCAL INFILE %s INTO TABLE `{table}`
            FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' 
            LINES STARTING BY '' TERMINATED BY '\n';
            """
    cur.execute(query, file)
    conn.close()



def fetch_pp_and_pc_joined_area(lat, long, date, lat_height=0.05, long_width=0.05, days_since=365):
    """ Joins the property price and postcode data along the postcode column.
        Then fetches data from the join which lies within a bounding box of 
        lattitude and longitude and also days since on either side of the 
        date given.
    :param lat: float lattitude
    :param long: float longitude
    :param date: datetime object date
    :param lat_height: float lattitude height
    :param long_width: float longitude width
    :param days_since: int days since both sides
    :return joined_df: pandas dataframe joined data returned
    """
    conn = create_connection(database="assessment_db")

    lat_max = lat + lat_height/2
    lat_min = lat - lat_height/2
    long_max = long + long_width/2
    long_min = long - long_width/2

    date_max = (date + datetime.timedelta(days=days_since)).strftime("%Y%m%d")
    date_min = (date - datetime.timedelta(days=days_since)).strftime("%Y%m%d")

    try:
        cur = conn.cursor()
    
        # https://www.gov.uk/guidance/about-the-price-paid-data contains information on the columns and what they mean (and their possible values)
        cur.execute(f"""
                    SELECT 
                      pp.price as price, pp.date_of_transfer as date, pp.postcode as postcode, pp.property_type as type, pp.new_build_flag as new, pp.tenure_type as tenure, pp.county as county,
                      pc.usertype as size, pc.country as country, pc.lattitude as lattitude, pc.longitude as longitude
                    FROM
                      (SELECT * FROM pp_data
                      WHERE  CAST(date_of_transfer AS date) <= {date_max}
                      AND CAST(date_of_transfer AS date) >= {date_min}) pp
                    INNER JOIN 
                      (SELECT * FROM postcode_data
                      WHERE lattitude BETWEEN {lat_min} AND {lat_max}
                      AND longitude BETWEEN {long_min} AND {long_max}) pc
                    ON
                      pp.postcode = pc.postcode
                    """)

        rows = cur.fetchall()
        col_names = [col[0] for col in cur.description]
        joined_df = pd.DataFrame(rows, columns=col_names)
        print("joined data fetched")
        return joined_df
    finally:
        conn.close()



def setup_pp_table():
    """ Creates the property price data table and adds indexes.
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
                """)

    cur.execute(f"""
                USE `{db_name}`;
                """)

    cur.execute("""
                --
                -- Table structure for table `pp_data`
                --
                DROP TABLE IF EXISTS `pp_data`;
                """)

    cur.execute("""
                CREATE TABLE IF NOT EXISTS `pp_data` (
                  `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
                  `price` int(10) unsigned NOT NULL,
                  `date_of_transfer` date NOT NULL,
                  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
                  `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
                  `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
                  `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
                  `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
                  `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
                  `street` tinytext COLLATE utf8_bin NOT NULL,
                  `locality` tinytext COLLATE utf8_bin NOT NULL,
                  `town_city` tinytext COLLATE utf8_bin NOT NULL,
                  `district` tinytext COLLATE utf8_bin NOT NULL,
                  `county` tinytext COLLATE utf8_bin NOT NULL,
                  `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
                  `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
                  `db_id` bigint(20) unsigned NOT NULL
                ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
                """)

    cur.execute("""
                --
                -- Indexes for table `pp_data`
                --
                ALTER TABLE `pp_data`
                ADD PRIMARY KEY (`db_id`);
                """)
    cur.execute("""
                ALTER TABLE `pp_data`
                MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
                """)

    cur.execute("""
                CREATE INDEX `pp.transaction_unique_identifier` USING HASH
                  ON `pp_data`
                    (transaction_unique_identifier);
                """)

    cur.execute("""
                CREATE INDEX `pp.date` USING HASH
                  ON `pp_data` 
                    (date_of_transfer);
                """)
  
    conn.close()
    print("pp_data table setup")


def setup_pc_table():
    """ Creates the postcode data table and adds indexes.
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
                """)

    cur.execute(f"""
                USE `{db_name}`;
                """)

    cur.execute("""
                --
                -- Table structure for table `postcode_data`
                --
                DROP TABLE IF EXISTS `postcode_data`;
                """)

    cur.execute("""
                CREATE TABLE IF NOT EXISTS `postcode_data` (
                  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
                  `status` enum('live','terminated') NOT NULL,
                  `usertype` enum('small', 'large') NOT NULL,
                  `easting` int unsigned,
                  `northing` int unsigned,
                  `positional_quality_indicator` int NOT NULL,
                  `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
                  `lattitude` decimal(11,8) NOT NULL,
                  `longitude` decimal(10,8) NOT NULL,
                  `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
                  `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
                  `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
                  `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
                  `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
                  `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
                  `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
                  `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
                  `db_id` bigint(20) unsigned NOT NULL
                ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
                """)

    cur.execute("""
                ALTER TABLE `postcode_data`
                ADD PRIMARY KEY (`db_id`);
                """)
  
    cur.execute("""
                ALTER TABLE `postcode_data`
                MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
                """)
  
    cur.execute("""
                CREATE INDEX `po.postcode` USING HASH
                  ON `postcode_data`
                    (postcode);
                """)
  
    conn.close()
    print("postcode_data table setup")



def upload_pp_data(first_year, last_year):
    """ Downloads and then uploads the property price data from
        the .gov website to the database instance.
    :param first_year: int first year to fetch pricing data (inclusive)
    :param last_year: int last year to fetch pricing data (exclusive)
    """
    if (first_year > last_year):
        raise ValueError("first_year should be less than or equal to last_year")
    if (first_year < 1995 or first_year > 2023):
        raise ValueError("first_year is an illegal argument, should be between 1995 and 2023 inclusive")
    if (last_year < 1995 or last_year > 2023):
        raise ValueError("last_year is an illegal argument, should be between 1995 and 2023 inclusive")
  
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"
    try:
        os.mkdir("pp_data")
    except FileExistsError:
        pass
    
    # download files
    for year in range(first_year, last_year):
        year_data_url = base_url + "pp-" + str(year) + ".csv"
        data_csv_name = str(year) + "_PPData.csv"
        if os.path.exists("pp_data/" + str(year) + "_PPData.csv"):
            continue
        year_data_url = base_url + "pp-" + str(year) + ".csv"
        data_csv_name = str(year) + "_PPData.csv"
        pd.read_csv(year_data_url).to_csv("pp_data/" + data_csv_name, index = False)
        print(f"{data_csv_name} downloaded")
  
    # upload files
    for year in range(first_year, last_year):
        data_csv_name = str(year) + "_PPData.csv"
        data_csv_path = "pp_data/" + data_csv_name
        upload_csv_to_db(data_csv_path, "pp_data")
        print(f"{data_csv_name} uploaded to pp_data")
  
  
def upload_pc_data():
    """ Download and upload postcode data from the getthedata website 
        to the database instance.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"} # copied from https://medium.com/@speedforcerun/python-crawler-http-error-403-forbidden-1623ae9ba0f

    # download data
    unzip_location = "."
    req = Request(url="https://www.getthedata.com/downloads/open_postcode_geo.csv.zip", headers=headers) # need headers to allow for webscraping
    resp = urlopen(req)
    zipfile = ZipFile(BytesIO(resp.read()))
    zipfile.extractall(path=unzip_location)
    print("ONS data downloaded")

    # upload data
    upload_csv_to_db("open_postcode_geo.csv", "postcode_data")
    print("ONS data uploaded")


# db data cleaning/sanity checks

def fetch_table_head(table, limit):
    """ Fetches the head of a table up to a given number of rows.
    :param table: string table name
    :param limit: int number of rows
    :return rows: tuple of tuple of row values
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                SELECT * FROM {table} LIMIT {limit};
                """)
    rows = cur.fetchall()
    conn.close()
    return rows

def table_row_count(table):
    """ Fetches the row count in a table.
    :param table: string table name
    :return rows: tuple of tuple of row values
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                SELECT count(*) FROM {table};
                """)
    rows = cur.fetchall()
    conn.close()
    return rows


def check_column_for_null(table, column_name):
    """ Checks for a column, how many rows contain a null value. 
    :param table: string table name
    :param column_name: string column name
    :return rows: tuple of tuple of row values
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                SELECT count(*) FROM {table}
                WHERE {column_name} IS NULL;
                """)
    rows = cur.fetchall()
    conn.close()
    return rows


def check_column_for_value(table, column_name, check_value):
    """ Checks for a column, how many rows contain a specific value. 
    :param table: string table name
    :param column_name: string column name
    :param check_value: string or int value
    :return rows: tuple of tuple of row values
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                SELECT count(*) FROM {table}
                WHERE {column_name} = "{check_value}";
                """)
    rows = cur.fetchall()
    conn.close()
    return rows


def check_for_no_longitude_and_lattitude():
    """ Gives how many rows don't have a value for both lattitude and longitude.
    :return rows: tuple of tuple of row values
    """
    conn = create_connection()

    cur = conn.cursor()

    cur.execute(f"""
                SELECT count(*) FROM postcode_data
                WHERE longitude = "" and lattitude = "";
                """)
    rows = cur.fetchall()
    conn.close()
    return rows



# osm functions


def fetch_pois_from_bounding_box(lattitude, longitude, box_width, box_height, tags):
    """ Fetches all of the points of interest (pois) contained within the 
        box at the coordinates which match the tags given.
    :param lattitude: float lattitude
    :param longitude: float longitude
    :param box_width: float box width
    :param box_height: float box_height
    :param tags: dictionary string->dictionary category->tags for osm to match with
    :return: geo dataframe pois found
    """
    north, south, west, east = calculate_bounding_box_dimensions(lattitude, longitude, box_width, box_height)
    return ox.geometries_from_bbox(north, south, east, west, tags)


def calculate_bounding_box_dimensions(lattitude, longitude, box_width, box_height):
    """ Helper function to transfer box specification and coordinates
        into coordinates for max and min box height and width.
    :param lattitude: float lattitude
    :param longitude: float longitude
    :param box_width: float box width
    :param box_height: float box_height
    :return north: float north
    :return south: float south
    :return west: float west
    :return east: float east
    """
    north = lattitude + box_height/2
    south = lattitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    return north, south, west, east


def fetch_tags(category):
    """ Mapping from category name to the set of osm tags which 
        correspond to that category. More details available here:
        https://wiki.openstreetmap.org/wiki/Map_features
    :param category: string category
    :return: dictionary string->dictionary category->tags for osm to match with
    """
    # https://wiki.openstreetmap.org/wiki/Map_features for a list of tags and what they correspond to
    return {
        "schooling": {"amenity": ["kindergarten", "music_school", "school"]},
        "further_education": {"amenity": ["college", "library", "university"]},
        "children": {"amenity": ["toy_library", "library", "childcare"],
                     "leisure": ["amusement_arcade", "pitch", "playground", "summer_camp", "water_park"],
                     "shop": ["baby_goods"]},
        "healthcare": {"amenity": ["clinic", "dentist", "doctors", "hospital", "nursing_home", "pharmacy", "social_facility", "veterinary"],
                       "healthcare": ["yes"]},
        "restaurants": {"amenity": ["bar", "biergarten", "cafe", "food_court", "ice_cream", "pub", "restaurant"]},
        "essential_shops": {"shop": ["convenience", "general", "kiosk", "supermarket"],
                            "amenity": ["fuel", "atm", "bank", "pharmacy"]},
        "shopping": {"amenity": ["marketplace"],
                     "shop": True},
        "creative": {"amenity": ["music_school", "arts_centre", "cinema", "community_centre", "events_venue", "social_centre", "studio", "theatre", "marketplace"],
                     "craft": ["atelier", "bakery", "basket_maker", "confectionery", "embroiderer", "goldsmith", "handicraft",  "jeweller", "joiner", "musical_instrument", "photographer", "photographic_laboratory", "piano_tuner", "pottery", "sculptor"],
                     "leisure": ["bandstand"],
                     "shop": ["art", "craft", "music", "musical_instrument", "photo"],
                     "tourism": ["artwork", "attraction", "gallery"]},
        "leisure": {"amenity": ["casino", "cinema", "gambling", "nightclub", "planetarium", "theatre"],
                    "leisure": True,
                    "tourism": ["aquarium", "attraction", "attraction", "attraction", "zoo"]},
        "exercise": {"amenity": ["dive_centre"],
                     "building": ["grandstand", "sports_hall", "stadium"],
                     "leisure": ["dance", "fishing", "fitness_centre", "fitness_station", "horse_riding", "pitch", "sports_centre", "swimming_area", "swimming_pool", "track"],
                     "sport": True},
#       "public_services": {"amenity": ["fire_station", "police", "post_box", "post_depot", "post_office", "ranger_station", "townhall", "prison", "courthouse"]}, TODO: Removed because feature seem to be more misleading than anything (prison vs postoffice, close to police being annoying) - can test this theory using data analysis which should suggest no correlation
        "public_transport": {"amenity": ["bus_station", "ferry_terminal"],
                             "public_transport": ["stop_position", "station"],
                             "highway": ["bus_stop"]},
        "transport_utility": {"amenity": ["bicycle_parking", "bicycle_repair_station", "bicycle_rental", "boat_rental", "boat_sharing", "car_rental", "car_sharing", "car_wash", "charging_station", "fuel", "grit_bin", "	motorcycle_parking", "parking", "parking_entrance", "parking_space", "taxi"]},
        "green_spaces": {"leisure": ["bird_hide", "dog_park", "	fishing", "garden", "nature_reserve", "park"],
                         "landuse": ["allotments", "farmland", "farmyard", "flowerbed", "forest", "meadow", "orchard", "vineyard", "grass", "greenhouse_horticulture", "plant_nursery", "recreation_ground", "village_green"],
                         "natural": True},
        "commercial": {"landuse": ["commercial"], 
                       "office": True},
        "industrial": {"amenity": ["sanitary_dump_station", "recycling", "waste_disposal", "waste_transfer_station", ],
                       "landuse": ["construction", "industrial", "depot", "landfill", "port", "quarry"],
                       "man_made": ["adit", "mineshaft", "pier", "pipeline", "pumping_station", "silo", "wastewater_plant", "water_works", "works"],
                       "power": ["plant", "substation", "switchgear", "transformer"]}
    }[category]


def fetch_features_outer_bbox(centre_lat, centre_long, feature_range_width, feature_range_height, property_range_width, property_range_height, category_list):
    """ Determines a bounding box which covers all properties and all the
        possible features these properties might want to draw on as features.
        Then fetches the osm pois within this box.
    :param centre_lat: float centre lattitude
    :param centre_long: float centre longitude
    :param feature_range_width: float feature range width
    :param feature_range_height: float feature range height
    :param property_range_width: float property range width
    :param property_range_height: float property range height
    :param category_list: list of strings categories to fetch
    :return category_pois_dict: dictionary string->geo dataframe  category->pois in category
    """
    total_width = property_range_width + feature_range_width
    total_height = property_range_height + feature_range_height

    category_pois_dict = {}
    for category in category_list:
        category_pois = fetch_pois_from_bounding_box(centre_lat, centre_long, total_width, total_height, fetch_tags(category))
        category_pois_dict[category] = fetch_pois_coordinates(category_pois)[["lattitude", "longitude"]]
  
    return category_pois_dict


def fetch_geo_features_nearby_from_df(lattitude, longitude, feature_box_width, feature_box_height, feature_df):
    """ Fetches all pois which are within a feature box distance of point
        by searching a dataframe covering the pois in an outer bounding box.
    :param lattitude: float centre lattitude
    :param longitude: float centre longitude
    :param feature_box_width: float feature range width
    :param feature_box_height: float feature range height
    :param feature_df: pandas dataframe of features in larger bounding box
    :return feature_df: pandas dataframe of features within feature box
    """
    north, south, west, east = calculate_bounding_box_dimensions(lattitude, longitude, feature_box_width, feature_box_height)
    return feature_df[(feature_df["lattitude"] <= north) &
                      (feature_df["lattitude"] >= south) &
                      (feature_df["longitude"] >= west) &
                      (feature_df["longitude"] <= east)]


def fetch_pois_coordinates(pois):
    """ Calculates a longitude and lattitude for each pois feature
        and assigns it a value in the new longitude and lattitude columns.
    :param pois: geo dataframe pois
    :return pois: geo dataframe pois
    """
    
    pois["lattitude"] = pois["geometry"].centroid.y #pois.apply(lambda feature: fetch_central_lattitude(feature["geometry"]), axis=1)
    pois["longitude"] = pois["geometry"].centroid.x #pois.apply(lambda feature: fetch_central_longitude(feature["geometry"]), axis=1)
    return pois


def fetch_test_town_pp_data(town_tests_dict, width=0.08, height=0.08, date=datetime.date(2019, 1, 1)):
    """ Fetches the pricing data for the towns which have been selected 
        as the validation/testing set.
    :param town_tests_dict: list of list pandas dataframe, string pair. list of pairs of towns and their dataframes
    :param width: float box width around town
    :param height: float box height around town
    :param date: datetime object date
    :return north: float north
    """
    town_dfs = []
    for town in town_tests_dict.keys():
        lat = town_tests_dict[town][0]
        long = town_tests_dict[town][1]
    
        price_df = fetch_pp_and_pc_joined_area(lat, long, date, height, width, days_since=365)
        town_dfs.append([price_df, town])
        print(f"fetched {town} pricing data")
    return town_dfs


# ----------------------


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

