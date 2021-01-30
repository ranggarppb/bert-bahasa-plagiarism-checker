"""

Collection of configuration variables

"""
from utils import config_reader
CONFIG_FILE="config.ini"


config_database = config_reader(CONFIG_FILE, section="database")
config_document = config_reader(CONFIG_FILE, section="full_document_setting")
config_model = config_reader(CONFIG_FILE, section="model")
config_matching = config_reader(CONFIG_FILE, section="matching")

# configuration for database
DB_LOCATION = config_database["location"]
FILE_DB_LOCATION = config_database["file_location"]
TXT_DB_LOCATION = config_database["txt_location"]

# configuration for documents
OPENING_BORDER = config_document["opening_border"].split(',')
CHAPTERS = config_document["chapters"].split(',')
PARAGRAPH_SEPARATOR = config_document["paragraph_separator"].split(',')
MIN_SPACE = int(config_document["min_space"])
CONJUNCTION = config_document["conjunction"].split(',')

# configuration for models
MODEL_DIR = config_model["dir"]
LAYERS    = config_model["layers"]
BATCH_SIZE = int(config_model["batch_size"])
MAX_SEQ_LENGTH = int(config_model["max_seq_length"])

# configuration for matching process
WORDS_PER_FILE_ASSUMP = int(config_matching["words_per_file_assump"])
COSIM_THRESHOLD = float(config_matching["cosim_threshold"])
PLAGIARISM_PERC = float(config_matching["plagiarism_perc"])
GPU_NUMBER = int(config_matching["gpu"])
N_JOBS_MULTIPROCESSING = int(config_matching["n_jobs_multiprocessing"])
MAX_WORKERS_CONCURRENCY = int(config_matching["max_workers_concurrency"])
NUMBER_OF_CLUSTER = int(config_matching["faiss_cluster"])
