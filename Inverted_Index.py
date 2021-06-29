import xml.etree.ElementTree as ET
import os
import math
from nltk.corpus import stopwords


class InvertedIndex:
    def __init__(self,path):
        self.corpus_path = path
        self.dictionary = {} # map words to a list of tuples of (document,score)
        self.documents_count = 1

    def build_dictionary(self):
        stop_words = set(stopwords.words("english"))
        
        self.documents_count = len(os.listdir(self.corpus_path))
        for file in os.listdir(self.corpus_path):
            xml_path = os.path.join(self.corpus_path,file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for record in root.findall("./RECORD"):
                file_id = record.find("./RECORDNUM").text.replace(" ",'')
                
                # get all words
                text = ""
                if not record.find("./TITLE") is None:
                    text += record.find("./TITLE").text.replace("\n"," ")
                if not record.find("./ABSTRACT") is None:
                    text += record.find("./ABSTRACT").text.replace("\n"," ")
                
                # convert to tokens
                tokens = [word.lower() for word in text.split(" ")]
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
        for word in self.dictionary.keys():
            df = len(self.dictionary[word])
            idf_score = math.log2(self.documents_count/df)
            for doc in self.dictionary[word]:
                doc[1] = doc[1] * idf_score # update from tf score to tf-idf score for each document

def create_index(path):
    print(f"Creating an inverted index on path {path}")
    index = InvertedIndex(path)
    print("Creating directory")
    index.build_dictionary()
    print("Updating tf-idf scores")
    index.update_tfidf_scores()
