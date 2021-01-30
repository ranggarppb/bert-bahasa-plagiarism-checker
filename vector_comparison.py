import os
import numpy as np
# from joblib import Parallel, delayed
# from multiprocessing import Process, Manager
import json
from itertools import repeat
# import concurrent.futures
from utils import dictfilt, flatten, batch
# from scipy.spatial.distance import cdist
from fastdist import fastdist
from global_var import COSIM_THRESHOLD, GPU_NUMBER, N_JOBS_MULTIPROCESSING, MAX_WORKERS_CONCURRENCY, CONJUNCTION, NUMBER_OF_CLUSTER, PLAGIARISM_PERC, WORDS_PER_FILE_ASSUMP
import faiss
from sklearn.preprocessing import normalize

# res = faiss.StandardGpuResources()

# def get_exceed_cosim(matrix1, matrix2, list_exceed_cosim) :
def get_exceed_cosim(matrix1, matrix2) :
    """
    Checking cosine similarity between vector (2-d)
    Input :
        - vector 1 (list 2-d) : input vector
        - vector 2 (list 2-d) : reference vector
    Output :
        - exceed_percentage (float) : percentage of word exceeding cosine similarity threshold
    """
    len_input = len(matrix1)
    len_reference = len(matrix2)
    all_combinations = len_input * len_reference
    cosim = fastdist.matrix_to_matrix_distance(np.array(matrix1), np.array(matrix2), fastdist.cosine, "cosine").reshape(-1)
    # cosim = cdist(np.array(matrix1),np.array(matrix2),metric="cosine").reshape(-1)
    # cosim = cosim[cosim < COSIM_THRESHOLD]
    cosim = cosim[cosim > COSIM_THRESHOLD]
    exceed = cosim.size
    exceed_percentage = exceed / all_combinations * 100

    # list_exceed_cosim.append(exceed_percentage)
    return exceed_percentage

def get_exceed_cosim_concurrent(vector1, batch_vector2) :
    """
    Apply concurrency to get_exceed_cosim function
    Input :
        - vector1 (list-1-d): input vector
        - batch_vector2 (list-2-d) : batch of vectors per workers in joblib (batch of database pdf features)
    Output :
        - list_exceed_cosim (list-1-d) : percentage of word exceeding cosine similarity for every comparison between input file and database file features
    """
    list_exceed_cosim = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS_CONCURRENCY) as executor :
        for exceed_cosim in executor.map(get_exceed_cosim, batch_vector2, repeat(vector1)) :
            list_exceed_cosim.append(exceed_cosim)

    return list_exceed_cosim

def working_instance(list_input_features, batched_all_db_features, gpu_id, job, list_exceed_cosim) :
    
    # initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    list_exceed_cosim_per_worker = []
    for matrix_feature in batched_all_db_features[gpu_id][job] :
        exceed_cosim = get_exceed_cosim(list_input_features, matrix_feature)
        list_exceed_cosim_per_worker.append(exceed_cosim)

    list_exceed_cosim.append(list_exceed_cosim_per_worker)

def get_features(list_features, source) :
    """
    Get list of feature vectors (4 vector for each word)
    Input :
        - list_features (list) : list of json object from extract_feature's output file
    Output :
        - list_result (list) : list of feature vectors
    """
    # loading the json string of each list element
    if source == "input":
        list_features = [json.loads(x) for x in list_features]

        # only take the features 
        # it results in multi-level list [[{"tokens": ...., "layers": .....},{"tokens": ...., "layers": .....},...], ...]
        list_features = [dictfilt(x,"features")['features'] for x in list_features]
    
        # flatten multi-level list
        list_features = flatten(list_features)

    # take out the conjunction tokens
    list_features = [x for x in list_features if x["token"] not in CONJUNCTION]

    # take only the last layers value
    list_features = [dictfilt(x,"layers")["layers"] for x in list_features]
    list_features = flatten(list_features)
    list_features = [dictfilt(x,"values")["values"] for x in list_features]

    return list_features

# def check_similarity(input, batched_all_db_features) :
def check_similarity(input, list_all_db_features, list_all_db_features_index, num_files_db) :

    """
    Get cosine similarity between two vectors
    Input :
        - input1 (jsonl) : file of the input file's embedding
        - list_all_db_features (list) : list of file of reference embedding batched by gpu and number of workers
    Output :
        - max_cosim (float) : maximum of cosine similarity between input file and refence database
    """
    # initiating features of input file
    list_input_features = get_features(input)
    list_input_features = np.array(list_input_features).astype('float32')
    # faiss.normalize_L2(list_input_features)
    list_input_features = normalize(list_input_features, axis=1, norm='l2')
    # initiating db features
    list_all_db_features = np.array(list_all_db_features).astype('float32')
    list_all_db_features = normalize(list_all_db_features, axis=1, norm='l2')
    # faiss.normalize_L2(list_all_db_features)
    list_all_db_features_index = np.array(list_all_db_features_index)

    # compare cosine similarity of input file with all db file with faiss
    # initialization
    cluster = NUMBER_OF_CLUSTER # number of cluster 
    dimension = len(list_all_db_features[0]) # dimension of one veector representing each word
    quantiser = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, cluster, faiss.METRIC_INNER_PRODUCT)
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    # training index on database vectors
    index.train(list_all_db_features)
    index.add(list_all_db_features)
    faiss.write_index(index, "database_vector.index")
    # querying words in database with most similar context with input files
    nprobe = NUMBER_OF_CLUSTER  # number of clusters to be checked
    number_of_similar_words = int((PLAGIARISM_PERC / 100) * WORDS_PER_FILE_ASSUMP * num_files_db)
    distances, indices = index.search(list_input_features, number_of_similar_words)
    # filtering distance and index which exceed cosine similarity angle threshold
    distances, indices = distances.flatten(), indices.flatten()
    distances = distances[np.where(distances>COSIM_THRESHOLD)]
    indices = indices[np.where(distances>COSIM_THRESHOLD)]
    # matching which index come from which file
    digitized_index = np.digitize(indices, list_all_db_features_index)
    # counting occurance of each file in similarity checking
    unique_elements_index, counts_elements_index = np.unique(digitized_index, return_counts=True)
    list_exceed_cosim = counts_elements_index / (len(list_input_features) * number_of_similar_words) * 100

    # # compare cosine similarity of input file with all db file with multiprocessing from joblib
    # list_exceed_cosim = Parallel(n_jobs=N_JOBS_MULTIPROCESSING)(delayed(get_exceed_cosim_concurrent)(list_input_features, batched_db_features) for batched_db_features in batched_all_db_features)    
    # list_exceed_cosim = flatten(list_exceed_cosim)

    # # initiating list of exceeding cosim percentage
    # with Manager() as manager :
    #     list_exceed_cosim = manager.list()
    #     # compare cosine similarity of input file with all db file with multiprocessing
    #     processes = []
    #     for job in range(N_JOBS_MULTIPROCESSING) :
    #         for gpu_id in range(GPU_NUMBER) :
    #             p = Process(target=working_instance, args=(list_input_features, batched_all_db_features, gpu_id, job, list_exceed_cosim))
    #             p.start()
    #             processes.append(p)

    #         for process in processes :
    #             process.join()
    #list_exceed_cosim = list(list_exceed_cosim)
    #print(list_exceed_cosim)
    #list_exceed_cosim = flatten(list_exceed_cosim)

    return list_exceed_cosim

def check_similarity_from_db(input_features, database_faiss_index, database_vector_index, num_files_db) :
    
    # querying words in database with most similar context with input files
    nprobe = NUMBER_OF_CLUSTER # number of clusters to be checked
    # number_of_similar_words = int((PLAGIARISM_PERC / 100) * WORDS_PER_FILE_ASSUMP * num_files_db)
    number_of_similar_words = 1000
    distances, indices = database_faiss_index.search(input_features, number_of_similar_words)
    # filtering distance and index which exceed cosine similarity angle threshold
    distances, indices = distances.flatten(), indices.flatten()
    distances = distances[np.where(distances>COSIM_THRESHOLD)]
    indices = indices[np.where(distances>COSIM_THRESHOLD)]
    # matching which index come from which file
    digitized_index = np.digitize(indices, database_vector_index)
    # counting occrance of each file in similarity checking
    unique_elements_index, counts_elements_index = np.unique(digitized_index, return_counts=True)
    list_exceed_cosim = counts_elements_index / (len(input_features) * number_of_similar_words) * 100

    return list_exceed_cosim
