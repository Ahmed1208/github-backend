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

class Searcher:
    __init__(self, ):
        
        
    def get_closest_images(query_image_idx, num_results=10):
        feat_extractor = FeatureExtractor()
        query_features = feat_extractor.extract(image_path)
        distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
        idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
        return idx_closest

    def get_concatenated_images(indexes, thumb_height):
        thumbs = []
        for idx in indexes:
            img = image.load_img(images[idx])
            img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
            thumbs.append(img)
            print(images[idx])
        concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
        return concat_image
        
    