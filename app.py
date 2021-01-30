
"""

REST-ful API of this project

"""
import sys
import os
import time
import pickle
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
import faiss
from tika import parser
from plagiarism_checker import search_database
from global_var import DB_LOCATION, GPU_NUMBER, N_JOBS_MULTIPROCESSING
# from vector_comparison import get_features
# from utils import batch, split

# import multiprocessing
# import psutil

# initializing service
app = Flask(__name__)
api = Api(app)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# initializing API
plagiarism_put_args = reqparse.RequestParser()
plagiarism_put_args.add_argument("uploaded_pdf", type=str, required=True, help="Uploaded pdf")

# initiating list features for database file

# index = 0
# list_all_db_features = []
# list_all_db_features_index = [index]
# database = os.listdir(DB_LOCATION)
# num_files_db = len(database)

# do the same for each file in database
# start = time.time()
# for data in database : 
#     with open(f'{DB_LOCATION}/{data}', 'r') as json_file_database:
#        reference_results = list(json_file_database)
         
#     list_database_features = get_features(reference_results)
#     del reference_results
#     number_of_features = len(list_database_features)
#     index += number_of_features
#     list_all_db_features_index.append(index)
#     list_all_db_features.append(list_database_features)
# list_all_db_features = [feature for db_features in list_all_db_features for feature in db_features]
# end = time.time()
# print("Database reading time: ",end-start )

# batching reference database for multiprocessing in vector comparison
# batched_all_db_features = batch(list_all_db_features, N_JOBS_MULTIPROCESSING)

# batching reference database for multiprocessing gpu
# batched_all_db_features = [list(split(x, N_JOBS_MULTIPROCESSING)) for x in list(split(list_all_db_features, GPU_NUMBER))]

# initializing database FAISS index
database_faiss_index = faiss.read_index("database_faiss.index")
# initializing database vectors
# with open("database_vector.pkl","rb") as f :
#     database_vector = pickle.load(f)
# initializing database vector indexes
with open("database_vector_index.pkl","rb") as f :
    database_vector_index = pickle.load(f)
database_vector_index = np.asarray(database_vector_index)

class DetectPlagiarism(Resource) :

    def post(self):

        # Comment this for sanity check
        # print("CPU CORE: ", multiprocessing.cpu_count())
        # print("MEMORY STATUS: ", psutil.virtual_memory())

        start = time.time()

        args = plagiarism_put_args.parse_args()
        uploaded_pdf = args["uploaded_pdf"]
        # max_similarity, plagiarism_status = check(uploaded_pdf, batched_all_db_features)
        # max_similarity, plagiarism_status = check(uploaded_pdf, list_all_db_features, list_all_db_features_index, num_files_db)
        max_similarity, plagiarism_status, _ = search_database(parser, uploaded_pdf, database_faiss_index, database_vector_index)
        end = time.time()
        print(f"Processing time: {end-start}")

        return {"max_similarity_percentage" : max_similarity, "plagiarism_status" : plagiarism_status}
        # return {"uploaded_pdf":uploaded_pdf}

api.add_resource(DetectPlagiarism, "/")

if __name__ == "__main__" :
    app.run(debug=True, host="0.0.0.0", port="6000")
