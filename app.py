# Importing important libraries

import re  # For preprocessing
import pickle
import pandas as pd  # For data handling
pd.set_option('display.max_colwidth', 1000)
import numpy as np # linear algebra
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
#import spacy  # For preprocessing
from spacy.lang.en import English
#from spacy import displacy
#import textacy
from textacy import extract, text_stats
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from operator import contains
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, MeanShift
from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.decomposition import TruncatedSVD

#import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from gensim.models import KeyedVectors
import multiprocessing
from gensim.models import Word2Vec
import gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

stemmer = PorterStemmer()

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
