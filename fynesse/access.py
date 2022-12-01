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
    """ Asks for credentials to be used for AWS with MariaDB,
        and stores them on a temporary local file.
    :param username: username
    :param password: password
    """
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username, 
                            'password': password}
        yaml.dump(credentials_dict, file)

def fetch_database_credentials():
    """ Uses the stored credentials for AWS with MariaDB to fetch
        the username, the password, and the URL to the database. 
    :return username: string of username
    :return password: string of password
    :return url: string of url
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
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
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
  conn = create_connection()
  cur = conn.cursor()
  query = f"""
          LOAD DATA LOCAL INFILE %s INTO TABLE `{table}`
          FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' 
          LINES STARTING BY '' TERMINATED BY '\n';
          """
  cur.execute(query, file)
  conn.close()


def upload_pp_data(first_year, last_year):
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

def fetch_table_head(table, limit):
  conn = create_connection()

  cur = conn.cursor()

  cur.execute(f"""
              SELECT * FROM {table} LIMIT {limit};
              """)
  rows = cur.fetchall()
  conn.close()
  print(rows)

def table_row_count(table):
  conn = create_connection()

  cur = conn.cursor()

  cur.execute(f"""
              SELECT count(*) FROM {table};
              """)
  rows = cur.fetchall()
  conn.close()
  print(rows)


def check_column_for_null(table, column_name):
  conn = create_connection()

  cur = conn.cursor()

  cur.execute(f"""
              SELECT count(*) FROM {table}
              WHERE {column_name} IS NULL;
              """)
  rows = cur.fetchall()
  conn.close()
  print(rows)


def check_column_for_value(table, column_name, check_value):
  conn = create_connection()

  cur = conn.cursor()

  cur.execute(f"""
              SELECT count(*) FROM {table}
              WHERE {column_name} = "{check_value}";
              """)
  rows = cur.fetchall()
  conn.close()
  print(rows)



def check_for_no_longitude_and_lattitude():
  conn = create_connection()

  cur = conn.cursor()

  cur.execute(f"""
              SELECT count(*) FROM postcode_data
              WHERE longitude = "" and lattitude = "";
              """)
  rows = cur.fetchall()
  conn.close()
  print(rows)

def setup_pp_table():
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

# ----------------------


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

