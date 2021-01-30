import os
import subprocess
from glob import glob
from tika import parser
from tokenizers import BertWordPieceTokenizer
from text_processing import get_text_data
from model import create_pretraining_data, run_pretraining
from global_var import FILE_DB_LOCATION, TXT_DB_LOCATION, MODEL_DIR 


files = os.listdir(FILE_DB_LOCATION)

if __name__ == "__main__" :

    # preparing all files in txt
    # for file in files :
    #    if not os.path.exists(f'{TXT_DB_LOCATION}/{file}.txt') : 
    #        try : 
    #            text = get_text_data(parser, f'{FILE_DB_LOCATION}/{file}')
    #            with open(f'{TXT_DB_LOCATION}/{file}.txt','w') as f :
    #                f.write('\n'.join(text))
    #            with open('training_data.txt','a') as t :
    #                t.write('\n'.join(text)+'\n\n')
    #        except :
    #            pass

    # training new tokenizer based on new files
    # all_txt = glob('./sample_database_txt/*.txt*')
    # tokenizer = BertWordPieceTokenizer(clean_text=True, strip_accents=True, lowercase=True)
    # tokenizer.train(all_txt, vocab_size=32000, min_frequency=2, show_progress=True, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], limit_alphabet=1000, wordpieces_prefix="##")
    # tokenizer.save_model('./','bert-oceanography-uncased')

    # preparing text data in tfrecord format
    # create_pretraining_data.prepare_pretraining_data('training_data.txt', 'training_data.tfrecord', 'bert-oceanography-uncased-vocab.txt')
    
    # create new languange model based on new tfrecords
    run_pretraining.run_training('training_data.tfrecord', 'training_output', f'{MODEL_DIR}/bert_config.json', None, 1e-4, 90000, 100, 32, 8, False, 128, 20, 100, 1000, 1000, None, True, True)
