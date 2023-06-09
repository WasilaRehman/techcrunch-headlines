# Importing important libraries

import re  # For preprocessing
import pickle
import pandas as pd  # For data handling
pd.set_option('display.max_colwidth', 1000)
import numpy as np # linear algebra

import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Get Kaggle API credentials from environment variables
kaggle_username = os.environ['KAGGLE_USERNAME']
kaggle_key = os.environ['KAGGLE_KEY']

# Set up Kaggle API credentials
api = KaggleApi()
api.authenticate()

# Download the dataset as a zipfile
dataset_name = 'thibalbo/techcrunch-posts-compilation'
api.dataset_download_files(dataset_name)

# Unzip the downloaded file
with zipfile.ZipFile(dataset_name + '.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

# Load the data from the unzipped file
data_path = './data/techcrunch-posts-compilation.csv'
data = pd.read_csv(data_path)






def main():
    st.set_page_config(layout="wide")
    # Giving title
    st.title('Relevant News Headline Finder')

    #Introduction to Web app
    st.write('This is a web app created to fetch relevant news headlines\
             Please provide a query and our different models will fetch its relevant result')

    st.write(data.head())

   


if _name_ == '_main_':
    main()
