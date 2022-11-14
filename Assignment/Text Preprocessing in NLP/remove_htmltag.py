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

def list_to_string(documents):
    #Initialize an Empty String
    str1 = ""

    #Traversing the String
    for element in documents: #element[0],element[1]....
        for x in element: 
           str1 += x

        #Return String
    return str1

def remove_html_tags(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

import json

def main():
  print('Reading The Dataset...')
  dataset = read_files('datasets/corona_data/test_small.tsv')
  
  labels, documents = separate_labels(dataset)

  doclist = list_to_string(documents)
  
  #Remove HTML tags
  remove_html = remove_html_tags(doclist)
  with open('datasets/corona_data/output/remove_html_tags_out.txt', 'w') as filehandle:
    json.dump(remove_html, filehandle)
    print(remove_html)
if __name__ == '__main__':
  main()