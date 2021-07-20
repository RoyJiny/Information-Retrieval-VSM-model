from Inverted_Index import load_index
from nltk.corpus import stopwords
import nltk
import math
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer


def retrieval(index_path,query):
    """Implamantion of the Retrieval Algorithm"""
    index = load_index(index_path)
    query_tokens = extract_query_tokens(query) # Q maps token in the query to its tf-idf
    query_lenght = 0.0
    retrieved_documents = {} # R: store retrieved documents with scores
    for token in query_tokens.keys():
        idf_score = get_idf_score(index,token) # I
        query_tokens[token] = query_tokens[token]*idf_score # update token weight. 
        token_occurence = []
        try:
            token_occurence = index.dictionary[token] # List of [file_id,tf_idf score]
        except:
            token_occurence = []
        for element in token_occurence:
            if not element[0] in retrieved_documents.keys():
                retrieved_documents[element[0]] = 0.0
            retrieved_documents[element[0]] += (query_tokens[token]*element[1])
        query_lenght += query_tokens[token]**2 # L 
    
    for doc in retrieved_documents.keys():
        retrieved_documents[doc] = retrieved_documents[doc]/(math.sqrt(query_lenght)*index.documents_lenght[doc])
    
    sorted_retrived_doc = {k: v for k, v in reversed(sorted(retrieved_documents.items(), key=lambda item: item[1]))}
    return sorted_retrived_doc


def extract_query_tokens(query):
    stop_words = set(stopwords.words("english"))
    tokens =[]
    token_words2 =[]
    ps = PorterStemmer()
    lem = LancasterStemmer()

    # convert to tokens
    words = [re.sub("[^a-z]+","",word.lower()) for word in query.split(" ") if re.sub("[^a-z]+","",word) != '' if not word in stop_words]
    sentence = " ".join(words)
    sentence = " ".join(nltk.RegexpTokenizer(r"\w+").tokenize(sentence))
    token_words = word_tokenize(sentence)
    for t in token_words:
        token_words2.append(ps.stem(t))
    for t in token_words2:
        tokens.append(lem.stem(t))

    tf_dict = {}
    # remove stop words and duplications
    filtered_tokens = list(set([token for token in tokens if not token in stop_words]))
    # calculate tf scores and then normalize
    for token in filtered_tokens:
        tf_dict[token] = tokens.count(token)
    max_freq = max([tf_dict[token] for token in filtered_tokens])
    for token in filtered_tokens:
        tf_dict[token] = tf_dict[token]/max_freq
    return tf_dict

def get_idf_score(index,token):
    idf_score = 0
    try:
        df = len(index.dictionary[token])
        idf_score = math.log2(index.documents_count/df) 
    except:
        pass
    return idf_score
