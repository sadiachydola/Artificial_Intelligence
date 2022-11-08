import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Colab Notebooks/Ailab

def read_files(file_loc):
  '''
  This function reads text data from a file in the drive

  args - a string containing the files location
  returns - a list containing the text data
  '''

  dataset = []

  with open(file_loc, 'r', encoding='utf-8') as file:
    for line in file:
      dataset.append(line)
  '''from here the dataset will be returned to main function'''
  return dataset

def separate_labels(dataset):
  '''This function will separate the labels/class and examples/documents from the dataset'''
  
  labels = []
  documents = []

  for line in dataset:
    '''This will split the whole document where it gets a tab'''

    splitted_line = line.strip().split('\t', 1)
    labels.append(splitted_line[0])
    documents.append(splitted_line[1])

  return labels, documents

# Remove URL
def remove_url(documents):
  '''This function removes URL's from Texts'''
  
  url_removed = []

  for line in documents:
    url_removed.append(re.sub('http[s]?://\S+', '', line))

  return url_removed

# Remove Hashtag
def remove_hashtag(documents):
  '''This function will remove all occurences of # from the texts'''
  
  hashtag_removed = []

  # map hashtag to space
  translator = str.maketrans('#', ' '*len('#'), '')

  for line in documents:
    hashtag_removed.append(line.translate(translator))

  return hashtag_removed

# Remove Whitespaces
def remove_whitespaces(documents):
  '''This function removes multiple whitespaces and replace them with a single whitespace'''
  
  whitespace_removed = []

  for line in documents:
    whitespace_removed.append(' '.join(line.split()))

  return whitespace_removed

# Tokenizer
def tokenize_sentence(documents):
  '''This function takes a line and provides tokens/words by splitting them using NLTK'''
  
  tokenized_docs = []
  
  for line in documents:
    tokenized_docs.append(word_tokenize(line))

  return tokenized_docs

def char_n_gram_ready(documents):
  '''This function takes space and join the string'''
  
  joined_docs = []

  for line in documents:
    joined_docs.append(' '.join(line))

  return joined_docs

# Remove Punctuation
def remove_punctuation(documents):
  '''This function will remove all punctuation(!,?,[,{ etc)'''

  punct_removed = []

  for doc in documents:
    temp = []
    for word in doc:
      if word not in string.punctuation:
        temp.append(word)
    
    punct_removed.append(temp)

  return punct_removed

# Remove Stopwords
def remove_stopwords(documents):
  '''This function will remove all stopwords(is,are,and,we,from etc)'''

  stopword_removed = []

  stop_words = set(stopwords.words('english'))

  for doc in documents:
    temp = []
    for word in doc:
      if word not in stop_words:
        temp.append(word)
    
    stopword_removed.append(temp)

  return stopword_removed

# Stemming
def apply_stemmer(documents):
  '''This function will used to extract the base form of the words by removing affixes'''

  stemmed_docs = []
  
  stemmer = PorterStemmer()

  for doc in documents:
    stemmed_docs.append([stemmer.stem(plural) for plural in doc])

  return stemmed_docs

# Lemmatization
def lemmatize_word(documents):
  '''This function will emove inflectional endings only and to return the base or dictionary form of a word'''

  lemmatize_docs = []

  lemmatizer = WordNetLemmatizer()

  for doc in documents:
    lemmatize_docs.append([lemmatizer.lemmatize(word, pos ='v') for word in doc])

  return lemmatize_docs

''' We use a dummy function as tokenizer and preprocessor '''
def identity(X):
  return X

# Vectorizer Function
def vec_tfidf(tfidf):
  if tfidf:
    '''TfidfVectorizer will tokenize documents, learn the vocabulary and 
        inverse document frequency weightings, and allow you to encode new documents,
        also convert a collection of raw documents to a matrix of TF-IDF features.'''

    vec = TfidfVectorizer(preprocessor = identity, lowercase=True, analyzer='char',
                          tokenizer = identity, ngram_range = (9,9))
  else:
    '''CountVectorizer tokenizes(tokenization means dividing the sentences in words) the text along 
      with performing basic preprocessing.It removes the punctuation marks and converts all
      the words to lowercase. The vocabulary of known words is formed which is also used 
      for encoding unseen text later.'''

    vec = CountVectorizer(preprocessor = identity, lowercase=True, analyzer='char',
                         tokenizer = identity, ngram_range = (9,9))
  return vec

# Produce the output label in the format (<predicted_label>) as a .txt file.
def write_label(prediction):
  file = open("datasets/News_Classification/output/predicted_test_label.txt", mode="w")
  file.close()
  
  for label in prediction:
    file = open("datasets/News_Classification/output/predicted_test_label.txt", "a", encoding='utf-8')
    file.write(str(label))
    file.write("\n")
    file.close()

# Naive Bayes
def Naive_Bayes(train_docs, train_lbls, test_docs, test_lbls):
  ''' The vec_tfidf vectorizer will be called and the selected vecotorizer will be passed parameter.'''

  vec = vec_tfidf(tfidf = False)
    
  # Combines the vectorizer with the Naive Bayes classifier and send it method pipeline
  classifier = Pipeline([('vec', vec),
                         ('cls', MultinomialNB())])
  
  '''In spite of the fact that we took a variable classifier where vectorizer as vec and
      multinomialNB as cls were alloted, still classifier is unfilled as there is no information
      to be prepared.So, using fit method to fit all the data in classifier'''

  classifier.fit(train_docs, train_lbls)

  # Prediction is predicting test data by using predict method
  prediction = classifier.predict(test_docs)

  print("Naive Bayes Accuracy = ", round(accuracy_score(test_lbls, prediction)*100, 2), end=" %")
  print()

  # Report on the precision, recall, f1-score
  print(classification_report(test_lbls, prediction, labels=classifier.classes_, digits=3))

  # Confusion matrix is another better way to evaluate the performance
  print("Confusion Matrix :\n\n", confusion_matrix(test_lbls, prediction))
  print()

  # Calling the write_label Function
  write_label(prediction)

# Pre-processing
def pre_processing(documents):

  documents = remove_url(documents)

  documents = remove_hashtag(documents)

  documents = remove_whitespaces(documents)

  documents = tokenize_sentence(documents)

  documents = remove_punctuation(documents)

  documents = remove_stopwords(documents)

  documents = apply_stemmer(documents)

  documents = lemmatize_word(documents)

  ''' If we use character n_gram you have to enable it | else comment the below line '''
  documents = char_n_gram_ready(documents)

  return documents

def main():
  print('Reading The Dataset ... ')
  
  # Reading the training data
  training_dataset = read_files('datasets/News_Classification/train.txt')
  train_labels, train_docs = separate_labels(training_dataset)

  # Reading the test data
  test_dataset = read_files('datasets/News_Classification/dev.txt')
  test_labels, test_docs = separate_labels(test_dataset)

  
  # calling the pre processing function
  train_docs = pre_processing(train_docs)
  test_docs = pre_processing(test_docs)


  # Calling the Naive_Bayes Function
  print('\nTraining the Naive_Bayes Classifier ... \n')
  Naive_Bayes(train_docs, train_labels, test_docs, test_labels)

  for lbl, doc in zip(train_labels[:4], train_docs[:4]):
    print(lbl)
    print(doc)
    print()

if __name__ == '__main__':
  main()

Output:

Reading The Dataset ... 

Training the Naive_Bayes Classifier ... 

Naive Bayes Accuracy =  91.11 %
              precision    recall  f1-score   support

    Business      0.879     0.869     0.874      1132
    Sci_Tech      0.888     0.892     0.890      1158
      Sports      0.959     0.973     0.966      1182
       World      0.916     0.908     0.912      1128

    accuracy                          0.911      4600
   macro avg      0.910     0.911     0.910      4600
weighted avg      0.911     0.911     0.911      4600

Confusion Matrix :

 [[ 984  100    8   40]
 [  79 1033    8   38]
 [  11    5 1150   16]
 [  46   25   33 1024]]

Business
wall st. bear claw back into black reuter reuter short-sel wall street 's dwindling\band ultra-cyn see green ''

Business
carlyl look toward commerci aerospac reuter reuter privat invest firm carlyl group \which reput make well-tim occasionally\controversi play defens industri quietli placed\it bet anoth part market ''

Business
oil economi cloud stock outlook reuter reuter soar crude price plu worries\about economi outlook earn expect to\hang stock market next week depth the\summ doldrum ''

Business
iraq halt oil export main southern pipelin reuter reuter author halt oil export\flow main pipelin southern iraq after\intellig show rebel militia could strike\infrastructur oil offici say saturday ''