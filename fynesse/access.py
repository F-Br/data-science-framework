from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

import yaml
from ipywidgets import interact_manual, Text, Password


@interact_manual(username=Text(description="Username:"), 
                 password=Password(description="Password:"))
def write_credentials(username, password):
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username, 
                            'password': password}
        yaml.dump(credentials_dict, file)

def fetch_database_credentials():
  database_details = {"url": "database-fb531.cgrre17yxw11.eu-west-2.rds.amazonaws.com", 
                      "port": 3306}
                      
  with open("credentials.yaml") as file:
    credentials = yaml.safe_load(file)
  username = credentials["username"]
  password = credentials["password"]
  url = database_details["url"]
  return username, password, url

username, password, url = fetch_database_credentials()




def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

