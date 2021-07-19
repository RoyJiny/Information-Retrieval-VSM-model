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
            pass
        for element in token_occurence:
            if not element[0] in retrieved_documents.keys():
                retrieved_documents[element[0]] = 0.0
            retrieved_documents[element[0]] += (query_tokens[token]*element[1])
        query_lenght += query_tokens[token]**2 # L 
    
    for doc in retrieved_documents.keys():
        retrieved_documents[doc] = retrieved_documents[doc]/(math.sqrt(query_lenght)*index.documents_lenght[doc])
    
    sorted_retrived_doc = sorted(retrieved_documents.keys(), key=lambda x:x[1], reverse=True)
    # print(sorted_retrived_doc)
    return sorted_retrived_doc


def extract_query_tokens(query):
    # query_dict = {}
    stop_words = set(stopwords.words("english"))
    # convert to tokens
    # convert to tokens
    tokens =[]
    token_words2 =[]
    ps = PorterStemmer()
    lem = LancasterStemmer()
    words = [re.sub("[^a-z]+","",word.lower()) for word in query.split(" ") if re.sub("[^a-z]+","",word) != '' if not word in stop_words]
    sentence = " ".join(words)
    sentence = " ".join(nltk.RegexpTokenizer(r"\w+").tokenize(sentence))
    token_words = word_tokenize(sentence)
    for t in token_words:
        token_words2.append(ps.stem(t))
    for t in token_words2:
        tokens.append(lem.stem(t))

    #             tf_dict = {}
    # ps = PorterStemmer()
    # lem = LancasterStemmer()
    # text = query.lower()
    # # Tokenizing the words in the paper.
    # punctuation_tokenizer = nltk.RegexpTokenizer(r"\w+")
    # tokens_without_punctuation = punctuation_tokenizer.tokenize(text)
    # paper_str = " ".join(tokens_without_punctuation)
    # token_arr = word_tokenize(paper_str)

    # #  Removing stop words.
    # token_arr_without_sw = [token for token in token_arr if len(token) > 1 if not token in stop_words if not (token.isdigit() or token[0] == '-' and token[1:].isdigit())]
    # # Lancaster.
    # lem_token_arr = [lem.stem(token) for token in token_arr_without_sw]

    # # Stemming.
    # tokens = [ps.stem(token) for token in lem_token_arr]

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

    # tokens =[]
    # ps = PorterStemmer()
    # words = [re.sub("[^a-z]+","",word.lower()) for word in query.split(" ") if re.sub("[^a-z]+","",word) != '']
    # sentence = " ".join(words)
    # token_words = word_tokenize(sentence)
    # for t in token_words:
    #     tokens.append(ps.stem(t))

    # # tokens = [re.sub("[^a-z]+","",word.lower()) for word in query.split(" ") if re.sub("[^a-z]+","",word) != '']        
    # # remove stop words and duplications
    # filtered_tokens = list(set([token for token in tokens if not token in stop_words]))
    # # calculate tf scores and then normalize
    # for token in filtered_tokens:
    #     query_dict[token] = tokens.count(token)
    # max_freq = max([query_dict[token] for token in filtered_tokens])
    # for token in filtered_tokens:
    #     query_dict[token] = query_dict[token]/max_freq # K
    # return query_dict

def get_idf_score(index,token):
    idf_score = 0
    try:
        df = len(index.dictionary[token])
        idf_score = math.log2(index.documents_count/df) 
    except:
        pass
    return idf_score
