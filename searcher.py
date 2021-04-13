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

class Searcher:
    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        self.objects = []
        with (open(pickle_path, "rb")) as openfile:
            while True:
                try:
                    self.objects.append(pickle.load(openfile))
                except EOFError:
                    break
        self.images= self.objects[0][0]
        self.pca_features = self.objects[0][1]
        self.pca = self.objects[0][2]
        
    def get_concatenated_images(self, indexes, thumb_height):
        thumbs = []
        for idx in indexes:
            img = image.load_img(self.images[idx])
            img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
            thumbs.append(img)
            print(self.images[idx])
        concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
        return concat_image
    
    def similar_images(self, image_path, no_of_similar_images):
       feat_extractor = FeatureExtractor()
       query_features = feat_extractor.extract_query(image_path)
       query_features = np.array(query_features)
       query_pca_features = self.pca.transform(query_features)[0]
       
       distances = [ distance.cosine(query_pca_features, feat) for feat in self.pca_features ]
       idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:no_of_similar_images]  # grab first n
       results_image = self.get_concatenated_images(idx_closest, 200)
        
       #display the results
       query_image = image.load_img(image_path, target_size=feat_extractor.model.input_shape[1:3])
       plt.figure(figsize = (5,5))
       plt.imshow(query_image)
       plt.title("query image")
        
       # display the resulting images
       plt.figure(figsize = (16,12))
       plt.imshow(results_image)
       plt.title("result images")
       return query_image, results_image
            
           
      
        
        
    
    
