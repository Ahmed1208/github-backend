# import the necessary packages
from featuresextractor import FeatureExtractor
import argparse
import glob
import os
import time
import random
import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required = True,
# 	help = "Path to the directory that contains the images to be indexed")
# args = vars(ap.parse_args())




# initialize the feature extractor
feat_extractor = FeatureExtractor()



images_path = 'C:/Users/Mohamed/cbir-vgg16/flowers-recognition/flowers/'
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]


tic = time.process_time()

features = []
for i, image_path in enumerate(images):
    if i % 500 == 0:
        toc = time.process_time()
        elap = toc-tic;
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images),elap))
        tic = time.process_time()
    feat = feat_extractor.extract(image_path)
    features.append(feat)
    
# features = np.array(features)    
# pca = PCA(n_components=300)
# pca.fit(features)
# pca_features = pca.transform(features)

pca_features,pca = feat_extractor.extract_pca_features(features)
   
import pickle
pickle.dump([images, pca_features, pca], open('C:/Users/Mohamed/cbir-vgg16/features_caltech101.p', 'wb'))

print('finished extracting features for %d images' % len(images))


