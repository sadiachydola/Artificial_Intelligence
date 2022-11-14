import nltk
nltk.download('words')
import re

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

#tokenixation
def tokenization(documents):
  tokenization_data = []

  for line in documents:
    words = line.split()
    tokenization_data.append(words)

  return tokenization_data

import json

def main():
  print('Reading The Dataset...')
  dataset = read_files('datasets/corona_data/test_small.tsv')
  
  labels, documents = separate_labels(dataset)

  documents = tokenization(documents)
  with open('datasets/corona_data/output/tokenization_out.txt', 'w') as filehandle:
    json.dump(documents, filehandle)
    print(documents)

if __name__ == '__main__':
  main()