## Time for Topic Modelling
## importing gensim

from itertools import chain
from gensim import corpora,models, similarities
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import codecs
import string

Data_sciene_keywords = ['']

## Library to help define stop words

from stop_words import get_stop_words


#Defining some variables

stop_words = get_stop_words('english')
stop_words_dk = get_stop_words('danish')
stoplist = set(stop_words)
stoplist_dk = set(stop_words_dk)
document_path = "C:/Users/kevin/Data_2/"
frequency = defaultdict(int)
clean_docs = defaultdict(str)


def extract_txt_from_doc():
    
    List_of_docs = []
    
    for x in range (0,800):
        document = codecs.open(document_path+"Document_"+str(x)+".txt","r",encoding="utf-8")
        List_of_docs.append(document.readlines())
        
    
    return List_of_docs

def find_word_with_index(list_of_index,word_identifier_list):

    #looks at the list of indexes and prints out the word that is connected to the Identifier

    for doc in word_identifier_list:
        for index in list_of_index:
            if word_identifier_list[doc] == index:
                print(doc)


def extract_tfidf_index(corpus_tfidf):

    #extracts all indexes/words with a TFIDF of above 0.5

    indexes = []

    for doc in corpus_tfidf:
        for a in doc:
            #0.5 is the TFIDF THRESHOLD AND IS HARDCODED ATM, SHOULD BE A VARIABLE THAT IS DECLARED ON TOP of code
            if (a[1] > 0.5):
                indexes.append(a[0])

    return indexes

List_of_documents = extract_txt_from_doc()
List_of_documents_clean = []
document_nr = 0



for document in List_of_documents:


    words = [[word for word in sentence.lower().split() if word not in stoplist and word not in stoplist_dk]for sentence in document]

    clean_words = []

    for word in words:
        for token in word:
            #remove punctutation
            token = token.translate(str.maketrans('', '', string.punctuation))
            remove_punc = token.translate(str.maketrans('', '', string.punctuation))
            #removes digits and white spaces
            remove_digits = ''.join(c for c in remove_punc if not c.isdigit()).strip()
            #check to see if string is empty before adding it to a new list
            if remove_digits:
                clean_words.append(remove_digits)

            frequency[remove_digits] += 1



    #clean_words  = [x for word in words for x in word]

    List_of_documents_clean.append(clean_words)

#Remove any empty list in my list

List_of_documents_clean_1 = filter(None,List_of_documents_clean)

dictionary = corpora.Dictionary(List_of_documents_clean)

print(dictionary)

dictionary.save('temporary.dict')

word_identifier_list = dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in List_of_documents_clean]

corpora.MmCorpus.serialize('temporary.mm',corpus)

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

#Have to swap between keys and values in dict or else LDAmodel Will not recognize the format
#new_dict = {y:x for x,y in word_identifier_list.items()}

lsi = models.LdaModel(corpus = corpus, num_topics= 20 ,id2word=dictionary,passes=10)


print(lsi.print_topics(20))

## EVERYTHING UNDER THIS IS PROBABLY NOT WORKING - NOT ALOT OF INFORMATION WAS ABLE TO BE EXTRACTED USING TOPIC MODELLING : SO I DECIDED TO TAKE ANOTHER APPROACH. 

sorted_frequency = OrderedDict(sorted(frequency.items(),key=lambda t:t[1]))

import pandas as pd
import fnmatch

data_science_dict = pd.read_csv(r'C:\Users\kevin\Documents\data science skills.csv')


for x,y in sorted_frequency.items():
    #x is the word and y is the count
    for skill in data_science_dict:
        if fnmatch.fnmatchcase(x,'*'+skill+'*') :
            print(x,y)


count = 0



