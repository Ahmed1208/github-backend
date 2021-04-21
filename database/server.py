from elasticsearch import Elasticsearch
from os import listdir
from os.path import isfile, join
import pprint as pp
import base64
import pickle
import numpy
import sklearn
from elasticsearch_dsl import *

es = Elasticsearch()

analyzer = {
  "settings": {
    "analysis": {
      "filter": {
        "english_stop": {
          "type":       "stop",
          "stopwords":  "_english_"
        },
        "english_keywords": {
          "type":       "keyword_marker",
          "keywords":   ["example"]
        },
        "english_stemmer": {
          "type":       "stemmer",
          "language":   "english"
        },
        "english_possessive_stemmer": {
          "type":       "stemmer",
          "language":   "possessive_english"
        }
      },
      "analyzer": {
        "rebuilt_english": {
          "tokenizer":  "standard",
          "filter": [
            "english_possessive_stemmer",
            "lowercase",
            "english_stop",
            "english_keywords",
            "english_stemmer"
          ]
        }
      }
    }
  }
}

# images_foler_path = 'C:/Users/ahmed/Desktop/ahmed folders/progamming/final project/github backend/101_ObjectCategories/'
pickle_file_path = '/home/bondok/github-backend/features_caltech101.p'
images = list()  # list where images path are stored ex: airplanes/image_0007.jpg
pca_features = list()  # list where feature vectors of each image is stored, # each of size : 300                                                                      # dataset used : caltech101
pca = list()


class Imj:
    def __init__(self, path: str, base: str = None, tags: list = None, features: list = None, *args, **kwargs):
        self.path = path
        self.base = base
        self.tags = tags
        self.features = features


def create_doc(index, doc: dict):
    # used to create an index and manually set the body as dict to be a doc
    # not used
    es.index(index=index, body=doc)


def delete_index(index):
    # used to delete an index
    es.indices.delete(index=index, ignore=[400, 404])


def get_indices():
    # used to retrieve all indices in elasticsearch cluster
    indices = es.indices.get_alias().keys()
    index_0 = list(indices)
    index_1 = []
    for i in range(len(index_0)):
        if index_0[i][0] != '.':
            index_1.append(index_0[i])
    print(index_1)


def get_docs(index, field: str = None):
    # used to retrieve docs, if field is set will be retrieve and saved in images and pca_features
    # field values (features,tags, not set) base is not supported yet
    # _source_includes="name"
    if field:
        results = es.search(index=index, _source_includes=field, size=9999)['hits']['hits']
        modify_output(results)
    else:
        results = es.search(index=index)['hits']['hits']
        modify_output(results, field)  # use pp.pprint
    pp.pprint(results)



def delete_all_docs(index):
    # delete all docs in a certain index
    es.delete_by_query(index=index, body={"query": {"match_all": {}}})


def get_files(folder_path):
    # not used
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    print(onlyfiles)


def input_docs(index: str, base: str = None, path: str = None, tags: list = None, pickle_path: str = None,
               features: list = None, *args, **kwargs):
    # index: database name, base : not used yet, path and tags and features : used if input just one image, pickle_path: import data from pickle file
    global images
    global pca_features
    global pca
    if pickle_path:
        get_history_data(pickle_path)
        for i in range(len(images)):
            es.index(index=index, body={"tags": [images[i].split("/")[1]], "features": pca_features[i]}, id=images[i])
    else:
        es.index(index=index, body={"tags": tags, "features": features, "base": base}, id=path)


def image_to_base(image: Imj):
    # not used
    with open(image.path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
        image.base = str(b64_string)


def add_tags():
    pass


def add_feature():
    pass


def search_by_feature():
    pass


def search_by_tags(tag: str):
    # search by a certain tag and receive results in global lists images and pca
    # TODO english_analyzer
    query_body = {
        "query": {
            "fuzzy": {
                "tags": {"value": tag, "fuzziness": 2}
            }
        }
    }
    result = es.search(index="images", body=query_body, size=9999)
    modify_output(result["hits"]["hits"], "features")


def modify_output(results, field: str = "features", *args):
    # dont call this func
    global images
    global pca_features
    global pca
    images.clear()
    pca_features.clear()
    for i in results:
        images.append(i["_id"])
        pca_features.append(i["_source"][field])
    #pp.pprint(len(images))
    #pp.pprint(pca_features)


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


# get_files("/home/bondok/Downloads/kaggle")
# get_docs("images", "tags")
# delete_all_docs("images")
# get_indices()
# delete_index("test-index")
# input_docs(index="images",pickle_path=pickle_file_path)
search_by_tags("airplanes")
#pp.pprint(len(res))



