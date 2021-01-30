"""

Main functionalities of this project

"""
import json
import os
import time
import numpy as np
from model import extract_features
# from text_processing import get_required_part, organize_per_paragraph
from text_processing import get_text_data
from vector_comparison import get_features, check_similarity_from_db
from sklearn.preprocessing import normalize
from global_var import MODEL_DIR, PLAGIARISM_PERC

# def check(parser, file, list_all_db_features, list_all_db_features_index, num_files_db):

#     # parsing the input
#     parsed_file = parser.from_file(file)['content']

#     # get required chapters of document  (the default is 3 and 4)
#     needed_text = get_required_part(parsed_file)

#     # organize the text per paragraph
#     paragraphs = organize_per_paragraph(needed_text)

#     # do feature extraction per paragraphs
#     list_output_json = extract_features.extract(paragraphs, f'{MODEL_DIR}/vocab.txt', f'{MODEL_DIR}/bert_config.json', f'{MODEL_DIR}/bert_model.ckpt')

#     # check whether there is a file similar or not with input file
#     similarity = check_similarity(list_output_json, list_all_db_features, list_all_db_features_index, num_files_db)

#     # get maximum similarity
#     max_similarity = max(similarity)

#     if max_similarity >= PLAGIARISM_PERC :
#         plagiarism_status = True

#     else :
#         plagiarism_status = False
    
#     return max_similarity, plagiarism_status



def search_database(parser, file_title, database_faiss_index, database_vector_index):
    
    # process file to become a ready to process text per paragraph
    text_per_paragraphs = get_text_data(parser, file_title)
    # do feature extraction per paragraphs
#     input_features = extract_features.extract(text_per_paragraphs, f'{MODEL_DIR}/vocab.txt', f'{MODEL_DIR}/bert_config.json', f'{MODEL_DIR}/bert_model.ckpt', to_json=True, output_file=f"sample_database_embeddings_{file_title}_output.jsonl")
    input_features = extract_features.extract(text_per_paragraphs, f'{MODEL_DIR}/vocab.txt', f'{MODEL_DIR}/bert_config.json', f'{MODEL_DIR}/bert_model.ckpt', to_json=False, output_file=None)
    input_features = get_features(input_features, source="input")
    input_features = np.array(input_features).astype('float32')
    input_features = normalize(input_features)

    # check whether there is a file similar or not with input file
    num_files_db = len(database_vector_index) - 1
    similarity = check_similarity_from_db(input_features, database_faiss_index, database_vector_index, num_files_db)
    # get maximum similarity
    max_similarity = max(similarity)

    if np.isnan(max_similarity) == True :
        plagiarism_status = "Error in checking: file format not supported"
    elif max_similarity >= PLAGIARISM_PERC :
        plagiarism_status = "True"
    elif max_similarity < PLAGIARISM_PERC :
        plagiarism_status = "False"

    return max_similarity, plagiarism_status, input_features


