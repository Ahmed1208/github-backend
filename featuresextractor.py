import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    def  __init__(self):
        print("hello")
        self.model = keras.applications.VGG16(weights='imagenet', include_top=True)
        self.feat_extractor = Model(inputs=self.model.input, outputs=self.model.get_layer("fc2").output)
        
    def extract(self, image_path):
        img = image.load_img(image_path, target_size=self.feat_extractor.input_shape[1:3])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.feat_extractor.predict(x)[0]   
        return features
    
    def extract_pca_features(self, features):
        features = np.array(features)    
        pca = PCA(n_components=300)
        pca.fit(features)
        pca_features = pca.transform(features)
        return pca_features,pca
                                     
    
    
    
