from elasticsearch import Elasticsearch
import pickle


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


def get_docs(index, field: str = None):
    # used to retrieve docs, if field is set will be retrieve and saved in images and pca_features
    # field values (features,tags, not set) base is not supported yet
    # _source_includes="name"
    images = list()
    pca_features = list()
    if field:
        results = es.search(index=index, _source_includes=field, size=9999)['hits']['hits']
        images , pca_features = modify_output(results)
    else:
        results = es.search(index=index)['hits']['hits']
        images , pca_features = modify_output(results, field)  # use pp.pprint

    return images , pca_features

def input_docs(index: str, base: str = None, path: str = None, tags: list = None, pickle_path: str = None, features: list = None, *args, **kwargs):
    # index: database name, base : not used yet, path and tags and features : used if input just one image, pickle_path: import data from pickle file

    if pickle_path:
        images , pca_features = get_history_data(pickle_path)
        for i in range(len(images)):
            es.index(index=index, body={"tags": [images[i].split("/")[1]], "features": pca_features[i]}, id=images[i])
    else:
        es.index(index=index, body={"tags": tags, "features": features, "base": base}, id=path)

def search_by_tags(index:str, tag: str):
    # search by a certain tag and receive results in global lists images and pca
    # TODO english_analyzer
    query_body = {
        "query": {
            "fuzzy": {
                "tags": {"value": tag, "fuzziness": 2}
            }
        }
    }
    result = es.search(index=index, body=query_body, size=9999)
    return modify_output(result["hits"]["hits"], "features")

def modify_output(results, field: str = "features", *args):
    # dont call this func
    images = list()
    pca_features = list()

    for i in results:
        images.append(i["_id"])
        pca_features.append(i["_source"][field])
    return images, pca_features

def get_history_data(pickle_path):
    objects = []
    with (open(pickle_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    return objects[0][0] , objects[0][1]
