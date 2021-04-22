import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance
import random

pickle_file_path = ''
images = list()             #list where images path are stored ex: airplanes/image_0007.jpg
pca_features  = list()    #list where feature vectors of each image is stored, # each of size : 300 ( dataset used : caltech101 )
pca = list()
model = None
feat_extractor = None


def init_model(images_input , pca_features_input,pickle_file_path_input):
    global images
    global pca
    global pca_features
    global model
    global feat_extractor
    global pickle_file_path

    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    images = images_input
    pca_features = pca_features_input
    pickle_file_path = pickle_file_path_input

    objects = []
    with (open(pickle_file_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    pca = objects[0][2]

def get_history_data(pickle_path):
    global images
    global pca_features
    global pca

    objects = []
    with (open(pickle_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    images = objects[0][0]
    pca_features = objects[0][1]
    pca = objects[0][2]

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(query_image_idx, num_results=5):
    distances = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results + 1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height,images_foler_path):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images_foler_path + images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def get_similar_images(img_path,images_foler_path):
    # load image and extract features
    new_image, x = load_image(img_path)
    new_features = feat_extractor.predict(x)
    new_pca_features = pca.transform(new_features)[0]

    # calculate its distance to all the other images pca feature vectors
    distances = [distance.cosine(new_pca_features, feat) for feat in pca_features]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:5]  # grab first 5
    results_image = get_concatenated_images(idx_closest, 200,images_foler_path)

    # display the results
    plt.figure(figsize=(5, 5))
    plt.imshow(new_image)
    plt.title("query image")

    # display the resulting images
    plt.figure(figsize=(16, 12))
    plt.imshow(results_image)
    plt.title("result images")