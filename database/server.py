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

#images_foler_path = 'C:/Users/ahmed/Desktop/ahmed folders/progamming/final project/github backend/101_ObjectCategories/'
pickle_file_path = '/home/bondok/github-backend/features_caltech101.p'
images = list()             #list where images path are stored ex: airplanes/image_0007.jpg
pca_features = list()       #list where feature vectors of each image is stored, # each of size : 300                                                                      # dataset used : caltech101
pca = list()

class Imj:
    def __init__(self, path: str, base: str = None, tags: list = None, features: list = None, *args, **kwargs):
        self.path = path
        self.base = base
        self.tags = tags
        self.features = features

def create_doc(index,doc:dict):
    es.index(index=index,body=doc)

def delete_index(index):
    es.indices.delete(index=index, ignore=[400, 404])

def get_indecies():
    indices = es.indices.get_alias().keys()
    index_0 = list(indices)
    index_1 = []
    for i in range(len(index_0)):
        if index_0[i][0] != '.':
            index_1.append(index_0[i])
    print(index_1)

def get_docs(index,field:str=None):
    #_source_includes="name"
    if field:
        results = es.search(index=index, _source_includes=field)['hits']['hits']
    else:
        results = es.search(index=index)['hits']['hits']
    modify_output(results,field) #use pp.pprint

def delete_all_docs(index):
    es.delete_by_query(index=index, body={"query": {"match_all": {}}})

def get_files(folder_path):
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    print(onlyfiles)

def input_docs(index:str,base:str=None,path:str = None,tags:list=None,pickle_path:str=None,features:list = None,*args,**kwargs):
    global images
    global pca_features
    global pca
    if pickle_path:
        get_history_data(pickle_path)
        for i in range(len(images)):
            es.index(index=index, body={"tags": [images[i].split("/")[1]], "features": pca_features[i]},id=images[i])
    else:
        es.index(index=index, body={"tags": tags, "features": features, "base": base},id=path)

def image_to_base(image: Imj):
    with open(image.path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())
        image.base = str(b64_string)

def add_tags():
    pass

def add_feature():
    pass


def search_by_feature():
    pass

def search_by_tags(tag:str):
    #TODO english_analyzer
    query_body = {
        "query": {

            "fuzzy": {
                "tags": {"value": tag, "fuzziness": "auto"}
            }
        }
    }
    result = es.search(index="images",body= query_body,size=999)
    modify_output(result["hits"]["hits"])

def modify_output(results,field:str):
    global images
    global pca_features
    global pca
    images.clear()
    pca_features.clear()
    for i in results:
        images.append(i["_id"])
        pca_features.append(i["_source"][field])



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


#get_files("/home/bondok/Downloads/kaggle")
res = get_docs("images")
pp.pprint(res)
#delete_all_docs("images")
#get_indecies()
#delete_index("test-index")

#input_docs(index="images",pickle_path=pickle_file_path)
#res = search_by_tags("yin_yang")
#pp.pprint(len(res))