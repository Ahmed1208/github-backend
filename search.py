import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from scipy.spatial import distance
from featuresextractor import FeatureExtractor
import pickle
from searcher import Searcher


pickle_path = "features_caltech101.p"

searcher = Searcher(pickle_path)

query_path = "C:/Users/Mohamed/cbir-vgg16/flowers-recognition/flowers/daisy/5547758_eea9edfd54_n.jpg"
query, similar_images = searcher.similar_images(query_path, 5)