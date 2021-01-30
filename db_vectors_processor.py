"""

Process collection of database vectors every certain times (scheduled by db_vectors_processor_scheduler)

"""

import pickle
import json
import numpy as np
from sklearn.preprocessing import normalize
import faiss
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_json
from vector_comparison import get_features
from utils import flatten
from global_var import NUMBER_OF_CLUSTER

if __name__ == "__main__" :

    # initializing spark instance
    spark = SparkSession.builder.master("local").appName("plagiat-detector-db-processor").getOrCreate()
    databases = spark.read.option("multiLine",True).json("hdfs://localhost:9000/database_embeddings/*", recursiveFileLookup=True).select("features").toJSON().collect()
    # writing databases to database_vector.pkl
    list_all_db_vectors = []
    list_all_db_vectors_index = []
    i = 0
    for data in databases :
        data = json.loads(data)["features"]
        data = get_features(data, "database")
        data_index = [i] * len(data)
        list_all_db_vectors += data
        list_all_db_vectors_index.append(data_index)
    list_all_db_vectors = np.asarray(list_all_db_vectors).astype('float32')
    list_all_db_vectors = normalize(list_all_db_vectors, axis=1, norm='l2')
    list_all_db_vectors_index = flatten(list_all_db_vectors_index)
    with open("database_vector_index.pkl","wb") as f :
        pickle.dump(list_all_db_vectors_index, f)
    # initializing database vectors
    # with open("database_vector.pkl",'rb') as f : 
    #     database_vector = pickle.load(f)
    # initialization for FAISS algorithm
    cluster = NUMBER_OF_CLUSTER 
    dimension = list_all_db_vectors[0].shape[0]
    quantiser = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, cluster, faiss.METRIC_INNER_PRODUCT)
     # training index on database vectors
    index.train(list_all_db_vectors)
    index.add(list_all_db_vectors)
    faiss.write_index(index,"database_faiss.index")


    
