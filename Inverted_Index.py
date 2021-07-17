import xml.etree.ElementTree as ET
import os
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import numpy as np
import re

# from yaml import tokens

INDEX_FILE = "vsm_inverted_index.json"

class InvertedIndex:
    def __init__(self,path=''):
        self.corpus_path = path
        self.dictionary = {} # map words to a list of tuples of (document,score)
        self.docs_dictionary = {} # map docs to a list of tuples of (word,score)
        self.documents_count = 0
        self.documents_lenght = {} # map each doc to it's lenght

    def compute_doc_lenght(self):
        for doc in self.docs_dictionary.keys():
            doc_lenght = 0.0
            for element in self.docs_dictionary[doc]:
                doc_lenght += (element[1])**2
            self.documents_lenght[doc] = math.sqrt(doc_lenght)
    
    def build_dictionary(self):
        """ build an inverted index (as a python dict) with words mapped to all the docs that have them (+ their score) """
        stop_words = set(stopwords.words("english"))
        self.documents_count = 0
        for file in os.listdir(self.corpus_path):
            xml_path = os.path.join(self.corpus_path,file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            self.documents_count += len(root.findall("./RECORD"))
            for record in root.findall("./RECORD"):
                file_id = record.find("./RECORDNUM").text.replace(" ",'')
                
                # get all words
                text = ""
                if not record.find("./TITLE") is None:
                    text += record.find("./TITLE").text.replace("\n"," ")
                if not record.find("./ABSTRACT") is None:
                    text += record.find("./ABSTRACT").text.replace("\n"," ")
                
                # convert to tokens
                tokens =[]
                ps = PorterStemmer()
                words = [re.sub("[^a-z]+","",word.lower()) for word in text.split(" ") if re.sub("[^a-z]+","",word) != '']
                sentence = " ".join(words)
                token_words = word_tokenize(sentence)
                for t in token_words:
                    tokens.append(ps.stem(t))

                tf_dict = {}
                
                # remove stop words and duplications
                filtered_tokens = list(set([token for token in tokens if not token in stop_words]))
                
                # calculate tf scores and then normalize
                for token in filtered_tokens:
                    tf_dict[token] = tokens.count(token)
                max_freq = max([tf_dict[token] for token in filtered_tokens])
                for token in filtered_tokens:
                    tf_dict[token] = tf_dict[token]/max_freq
                
                # add to dictionary
                for token in filtered_tokens:
                    if token in self.dictionary.keys():
                        self.dictionary[token].append( [file_id,tf_dict[token]] )
                    else:
                        self.dictionary[token] = [ [file_id,tf_dict[token]] ]

    def update_tfidf_scores(self):
        """ after creating the initial inverted index with tf scores, compute idf scores and update the scores in the index to be tf-idf """
        for word in self.dictionary.keys():
            df = len(self.dictionary[word])
            idf_score = math.log2(self.documents_count/df)
            for doc in self.dictionary[word]:
                doc[1] = doc[1] * idf_score # update from tf score to tf-idf score for each document

    def create_docs_dictionary(self):
        """ create the 'inverse' dictionary, mapping docs to all the words that they have (+ their score) """
        for word in self.dictionary.keys():
            for lst in self.dictionary[word]:
                doc = lst[0]
                score = lst[1]
                if not doc in self.docs_dictionary.keys():
                    self.docs_dictionary[doc] = [(word,score)]
                else:
                    self.docs_dictionary[doc].append((word,score))

    def save(self):
        """ save all of the data to a json file """
        data = {
            "doc count": self.documents_count,
            "original path": self.corpus_path,
            "dictionary": self.dictionary
            # "doc_lenght": self.documents_lenght
        }
        json_object = json.dumps(data, indent=4)
        with open(INDEX_FILE,'w+') as json_file:
            json_file.write(json_object)

    def get_docs_for_word(self, word):
        """ given a word, return all of the docs that have it (with the relative score) """
        if word in self.dictionary.keys():
            return self.dictionary[word]
        return []


def load_index(path=INDEX_FILE):
    """ load the data back from the saved json file """
    with open(path) as json_file:
        data = json.load(json_file)
    index = InvertedIndex(data["original path"])
    index.documents_count = data["doc count"]
    index.dictionary = data["dictionary"]
    index.create_docs_dictionary()
    index.compute_doc_lenght()
    # index.documents_lenght = data["doc_lenght"]
    return index

def create_index(path):
    """ create the index for a given corpus path """
    print(f"Creating an inverted index on path {path}")
    index = InvertedIndex(path)
    print("Creating directory")
    index.build_dictionary()
    print("Updating tf-idf scores")
    index.update_tfidf_scores()
    print("Computing Documents lenght")
    index.compute_doc_lenght()
    print(f"Saving index under {INDEX_FILE}")
    index.save()
