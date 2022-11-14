!pip install contractions
!pip install nltk
!pip install Unidecode
!pip install word2number

# import the necessary libraries 
import string 
import re, unicodedata
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
import contractions
import unidecode
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem import LancasterStemmer
from nltk import pos_tag, ne_chunk
from word2number import w2n

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
           str1 += x #merge

        #Return String
    return str1

#(1)Tokenization
def tokenization(documents):
  tokenization_data = []

  for line in documents:
    words = line.split()
    tokenization_data.append(words)

  return tokenization_data

#(2)Text Lowercase
def text_lowercase(text):
    return text.lower()

#(3)Remove HTML tags
def remove_html_tags(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

#(4)Convert number words to numeric form
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

#(5)Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

#(6)Remove punctuation
def remove_puntuation(documents):
    return re.sub(r'[^\w\s]', '', documents)

#(7)Remove extra whitespaces
def remove_extra_whitespace(text):
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()

#(8)Convert accented characters to ASCII characters
def convert_accented_to_ascii(text):
    text = unidecode.unidecode(text)
    return text

#(9)Expand contractions
import contractions
def expand_contractions(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#(10)Remove special characters
def remove_special_characters(a):
    for k in a.split("\n"):
        return  re.sub(r"[^a-zA-Z0-9]+", ' ', k)

#(11)Remove default stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

#(12)Stemming
def stemming_words(text): 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    return stems 

#(13)Lemmatization
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return lemmas

#(14)Part of Speech (POS) Tagging
def pos_tagging(text):
    word_tokens = word_tokenize(text)
    return pos_tag(word_tokens)

#(15)Named Entity Recognition
def named_entity_recognition(text):
    # tokenize the text
    word_tokens = word_tokenize(text)

    # part of speech tagging of words
    word_pos = pos_tag(word_tokens)

    # tree of word entities
    return ne_chunk(word_pos)

import json

def main():
  print('Reading The Dataset...')
  dataset = read_files('datasets/corona_data/test_small.tsv')
  
  labels, documents = separate_labels(dataset)

  #(1)Tokenization
  documents = tokenization(documents)
  with open('datasets/corona_data/output1/tokenization_out.txt', 'w') as filehandle:
    json.dump(documents, filehandle)
    #print(documents)
   
  doclist = list_to_string(documents)
  
  #(2)Text Lowercase
  text_lower = text_lowercase(doclist)
  with open('datasets/corona_data/output1/text_lowercase_out.txt', 'w') as filehandle:
    json.dump(text_lower, filehandle)
    #print(text_lower)

  #(3)Remove HTML tags
  remove_html = remove_html_tags(doclist)
  with open('datasets/corona_data/output1/remove_html_tags_out.txt', 'w') as filehandle:
    json.dump(remove_html, filehandle)
    #print(remove_html)

  #(4)Convert number words to numeric form
  word_to = word_to_numeric(doclist)
  with open('datasets/corona_data/output1/convert_numwords_to_numeric_out.txt', 'w') as filehandle:
    json.dump(word_to, filehandle)
    #print(word_to)

  #(5)Remove numbers
  remove_number = remove_numbers(doclist)
  with open('datasets/corona_data/output1/remove_numbers_out.txt', 'w') as filehandle:
    json.dump(remove_number, filehandle)
    #print(remove_number)

  #(6)Remove punctuation
  remove_pun = remove_puntuation(doclist)
  with open('datasets/corona_data/output1/remove_puntuation_out.txt', 'w') as filehandle:
    json.dump(remove_pun, filehandle)
    #print(remove_pun)

  #(7)Remove extra whitespaces
  remove_extra = remove_extra_whitespace(doclist)
  with open('datasets/corona_data/output1remove_extra_whitespace_out.txt', 'w') as filehandle:
    json.dump(remove_extra, filehandle)
    #print(remove_extra)
    
  #(8)Convert accented characters to ASCII characters
  convert_accented_to_asc = convert_accented_to_ascii(doclist)
  with open('datasets/corona_data/output1/convert_accented_to_ascii_out.txt', 'w') as filehandle:
    json.dump(convert_accented_to_asc, filehandle)
    #print(convert_accented_to_asc)

  #(9)Expand_contractions
  expand_contract = expand_contractions(doclist)
  with open('datasets/corona_data/output1/expand_contractions_out.txt', 'w') as filehandle:
    json.dump(expand_contract, filehandle)
    #print(expand_contract)

  #(10)Remove_special_characters
  remove_special_charac =remove_special_characters(doclist)
  with open('datasets/corona_data/output1/remove_special_characters_out.txt', 'w') as filehandle:
    json.dump(remove_special_charac, filehandle)
    #print(remove_special_charac)

  #(11)Remove default stopwords
  remove_stop = remove_stopwords(doclist)
  with open('datasets/corona_data/output1/remove_stopwords_out.txt', 'w') as filehandle:
    json.dump(remove_stop, filehandle)
    #print(remove_stop)

  #(12)Stemming
  stemming_word = stemming_words(doclist)
  with open('datasets/corona_data/output1/stemming_words_out.txt', 'w') as filehandle:
    json.dump(stemming_word, filehandle)
    #print(stemming_word)

  #(13)Lemmatization
  lemmatization = lemmatize_word(doclist)
  with open('datasets/corona_data/output1/lemmatization_out.txt', 'w') as filehandle:
    json.dump(lemmatization, filehandle)
    #print(lemmatization)

  #(14)Part of Speech (POS) Tagging
  post_tag = pos_tagging(doclist)
  with open('datasets/corona_data/output1/pos_tagging_out.txt', 'w') as filehandle:
    json.dump(post_tag, filehandle)
    #print(post_tag)

  #(15)Named Entity Recognition
  named_entity = named_entity_recognition(doclist)
  with open('datasets/corona_data/output1/named_entity_recognition_out.txt', 'w') as filehandle:
    json.dump(named_entity, filehandle)
    #print(named_entity)

if __name__ == '__main__':
  main()