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

def remove_url(documents):
  '''This function removes URL's from Texts'''
  url_removed = []

  # Your code here
  for line in documents:
    url_removed.append(re.sub('http[s]?://\S+', '', line))

  return url_removed

def remove_hashtag(documents):
  '''This function will remove all occurences of # from the texts'''
  hashtag_removed = []

  # map hashtag to space
  translator = str.maketrans('#', ' '*len('#'), '')

  for line in documents:
    hashtag_removed.append(line.translate(translator))

  return hashtag_removed

def remove_whitespaces(documents):
  '''This function removes multiple whitespaces and replace them with a single whitespace'''
  whitespace_removed = []

  for line in documents:
    whitespace_removed.append(' '.join(line.split()))

  return whitespace_removed

import json

def pre_processing(documents):

  documents = remove_url(documents)
  with open('datasets/corona_data/output/remove_url_out.txt', 'w') as filehandle:
    json.dump(documents, filehandle)

  documents = remove_hashtag(documents)
  with open('datasets/corona_data/output/remove_hashtag_out.txt', 'w') as filehandle:
    json.dump(documents, filehandle)

  documents = remove_whitespaces(documents)
  with open('datasets/corona_data/output/remove_whitespaces_out.txt', 'w') as filehandle:
    json.dump(documents, filehandle)

  return documents

def main():
  print('Reading The Dataset...')
  dataset = read_files('datasets/corona_data/test_small.tsv')
  
  labels, documents = separate_labels(dataset)

  # calling the pre processing dunction
  documents = pre_processing(documents)
  # print(documents)

  for lbl, doc in zip(labels[:5], documents[:5]):
    print(lbl)
    print(doc)
    print()

if __name__ == '__main__':
  main()