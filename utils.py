"""

Collection of utility function

"""

from configparser import ConfigParser
from functools import reduce

# Function for reading configuration
def config_reader(config, section):
    """
    Reading configuration file
    Input :
        - config (str) : path to the configuration file
        - section (str) : section to be read
    Output :
        - config_dict (dictionary) : configuration dictionary
    """
    # Create parser
    parser = ConfigParser()

    # Read config file
    parser.read(config)

    # Get database section
    config_dict = {}
    params = parser.items(section)
    for param in params:
        config_dict[param[0]] = param[1]

    return config_dict

# Function to filter list
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i == y ])

# Function to flatten multilevel list
def flatten(x):
    """
    Flatten a multi-level list
    Input :
        - x (list) : a multi-level list to be flattened 
        - section (str) : section to be read
    Output :
        - config_dict (dictionary) : configuration dictionary
    """
    if isinstance(x, list):
        flattened = [a for i in x for a in flatten(i)]
    else:
        flattened = [x]

    return flattened

# Function to batch a list
def batch(input_list, batch_size):
    """
    Batch a list
    Input :
        - input_list (list) : one dimensional list to be batched
    Output :
        - batch_list (list) = result of batched list (two dimensional) 
    """
    def reducer(cumulator, item):
        if len(cumulator[-1]) < batch_size:
            cumulator[-1].append(item)
        else:
            cumulator.append([item])
        return cumulator
    batch_list = reduce(reducer, input_list, [[]])
    
    return batch_list

def split(input_list, n):
    """
    Splitting a list to n part
    Input :
        - input_list (list) : one dimensional list to be splitted
        - n (int) : number of partition
    Output :
        - split_result (list) : result of splitting
    """
    k, m = divmod(len(input_list), n)
    split_result = (input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    return split_result
