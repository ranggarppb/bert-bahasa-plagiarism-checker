"""

Produce clean text to be sent to the input.txt and used for similarity calculation 
then sent it to training.txt to update the model

It consists of 3 function :
- get_required_part
- organize_per_paragraph  
- clean_up

"""
import numpy as np
import re
from global_var import OPENING_BORDER, CHAPTERS, PARAGRAPH_SEPARATOR, MIN_SPACE


def ready_to_process_text(parser, file_title) :

    # parsing the input
    parsed_file = parser.from_file(file_title)['content']
    
    # get required chapters of document  (the default is 3 and 4)
    needed_text = get_required_part(parsed_file)

    # organize the text per paragraph
    paragraphs = organize_per_paragraph(needed_text)

    # # removing content part if there is still any
    # paragraphs = remove_content_part(paragraphs)

    return paragraphs

def get_text_data(parser, file_title) :

    # parsing the input
    parsed_file = parser.from_file(file_title)['content']

    # organize the text per sentence
    sentences = organize_per_sentence(parsed_file)

    return sentences


def get_required_part(text, opening_border=OPENING_BORDER, chapters=CHAPTERS):
    """
    Get only required part of document
    Input :
        - text (str) : collection of strings in the input document
        - opening_border (str) : marker of opening part and main part of document
        - chapters (str) : marker of needed chapters in the document
    Output :
        - needed_text (str) : result of required parts of document
    """


    text = repr(text)

    # deleting opening part (JUDUL, DAFTAR ISI, DAFTAR TABEL, etc.)
    for opening in opening_border : 
        try : 
            needed_text = text.split(opening)[1]
            break
        except :
            pass
    
    if "needed_text" not in locals() :
        needed_text = text

    # get required chapters
    # checking whether the text is regular research or not
    # if no, get all the chapters
    number_of_chapters = len(needed_text.split("BAB"))
    if number_of_chapters <= 6 : 
        start, end = chapters[0], chapters[1]
        opening_pattern = [f"BAB {start}(.*)BAB {end}",f"BAB {start}(.*)BAB  {end}",f"BAB  {start}(.*)BAB {end}",f"BAB  {start}(.*)BAB  {end}"]
        for pattern in opening_pattern : 
            try : 
                needed_text = re.search(pattern, needed_text).group(0)
                break
            except :
                pass
    return needed_text

def remove_non_alphabet(text) :
    """
    Remove non alphabet character from a string
    Input :
        text (str) : text to be edited
    Output :
        text (str) : result of editing
    """

    # remove new line tag
    text = text.replace('\\n','')
    text = text.replace('\n','')
    text = text.replace(repr("\\\n"),"")
    text = text.replace(repr("\\n"),"")
    text = text.replace(repr("\n"),"")

    # remove non alphabetic character
    pattern = re.compile('[^.a-zA-Z ]+')
    text = pattern.sub('',text)
    
    # standardizing end of sentence format
    text = text.replace(' .', '.')

    return text


def join_multiple_whitespace(text) :
    """
    Join more than one whitespace become one
    Input :
        text (str) : text to be counted
    Output :
        text (int) : result of editing
    """

    text = ' '.join(text.split())

    return text

def count_space(text) :
    """
    Counting space in a sentece
    Input :
        text (str) : text to be counted
    Output :
        space (int) : number of space
    """

    space = text.count(" ")
    
    return space

def remove_content_part(paragraphs) :

    paragraphs = [x for x in paragraphs if x.count('.') < 200]

    return paragraphs

def filter_important_sentence(list_of_text, min_space=MIN_SPACE) :
    """
    Counting space in a sentece
    Input :
        - list_of_text (str) : list of sentence to be filtered
        - min_space (int) : minimum count of space on which a sentence perceived as an important sentence 
    Output :
        - list_of_text (str) : list of resulting sentences
    """
    # counting space
    list_count_space = np.fromiter(map(count_space, list_of_text), dtype=int)

    # filter sentence which doesnt reach minimum space
    index_important_sentence = np.where(list_count_space > min_space)[0].tolist()
    list_of_text = [val for i, val in enumerate(list_of_text) if i in index_important_sentence]

    return list_of_text
    

def organize_per_paragraph(text, paragraph_separator=PARAGRAPH_SEPARATOR) :
    """
    Parse the text par paragraph so that the text fed to the model isn't too long
    Input :
        - text (str) : the text to be separated
        - paragraph_separator (str) : marker of separator of paragraphs
    Output :
        - paragraphs (list) : the paragraph from the input text
    """
    # separate text by paragraph separator
    for separator in paragraph_separator:
        paragraphs = text.split(separator)
        if len(paragraphs) > 20 :
            break
    
    # remove all non alphabet characters
    paragraphs = list(map(remove_non_alphabet, paragraphs))
   
    # join multiple whitespace
    paragraphs = list(map(join_multiple_whitespace, paragraphs))

    # remove part which contains only small number of characters
    paragraphs = filter_important_sentence(paragraphs)

    return paragraphs

def organize_per_sentence(text) :
    sentences = text.lower().split('.')
    sentences = list(map(remove_non_alphabet, sentences))
    sentences = list(map(join_multiple_whitespace, sentences))
    sentences = filter_important_sentence(sentences)
    return sentences
