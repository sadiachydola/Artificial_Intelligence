!pip install word2number

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
import string
from word2number import w2n


from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/Ailab/

def read_files(file_loc):
  dataset = []
  
  with open(file_loc, 'r', encoding='cp1258') as test_file:
    for line in test_file:
      dataset.append(line)
    
  return dataset

def separate_labels(dataset):
  documents = []
  labels = []

  for line in dataset:
    splitted_line = line.strip().split('\t', 2)
    labels.append(splitted_line[1])
    documents.append(splitted_line[2])

  return documents, labels

def list_to_string(document):
    #Initialize an Empty String
    str1 = ""

    #Traversing the String
    for element in document: #element[0],element[1]....
        for x in element: 
           str1 += x

        #Return String
    return str1

def word_to_numeric(doc):
    s=""
    for token in doc.split():
        a=""
        try:
            a=w2n.word_to_num(token)
        except:
            a=token
        s+=(str(a)+" ")
    return s

import json

def main():
  print("Reading The Dataset ...")
  dataset = read_files('datasets/corona_data/test_small.tsv')
  documents, labels = separate_labels(dataset)
  
  doclist = list_to_string(documents)

  #Convert number words to numeric form
  word_to = word_to_numeric(doclist)
  print(word_to)
  with open('datasets/corona_data/output/convert_numwords_to_numeric_out.txt', 'w') as filehandle:
    json.dump(word_to, filehandle)
    #print(word_to)

if __name__ == "__main__":
  main()