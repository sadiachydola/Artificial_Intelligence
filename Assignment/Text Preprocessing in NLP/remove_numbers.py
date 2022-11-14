!pip install matplotlib-venn

!apt-get -qq install -y libfluidsynth1

!pip install contractions

!pip install nltk

# import the necessary libraries 
import string 
import re, unicodedata
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import contractions
import inflect
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
nltk.download('punkt')

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/Ailab/

def read_files(file_loc):
  '''
  This function reads tsv data from a file in the drive

  args - a string containing the files location
  returns - a list containing the text data
  '''

  dataset = []

  with open(file_loc, 'r', encoding='cp1258') as file:
    for line in file:
      dataset.append(line)

  return dataset

def separate_labels(dataset):
  '''This function will separate the labels/class and examples/documents from the dataset'''
  labels = []
  documents = []

  for line in dataset:
    splitted_line = line.strip().split('\t', 2)
    labels.append(splitted_line[1])
    documents.append(splitted_line[2])

  return labels, documents

def list_to_string(document):
    #Initialize an Empty String
    str1 = ""

    #Traversing the String
    for element in document: #element[0],element[1]....
        for x in element: 
           str1 += x

        #Return String
    return str1

def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

import json

def main():
  print('Reading The Dataset...')
  dataset = read_files('datasets/corona_data/test_small.tsv')
  
  labels, documents = separate_labels(dataset)

  doclist = list_to_string(documents)
  
  #Remove numbers
  remove_number = remove_numbers(doclist)
  with open('datasets/corona_data/output/remove_numbers_out.txt', 'w') as filehandle:
    json.dump(remove_number, filehandle)
    print(remove_number)
if __name__ == '__main__':
  main()