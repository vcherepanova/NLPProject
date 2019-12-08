import os
import nltk
import re
import numpy as np
import copy
import numpy as np
from nltk.tokenize import  word_tokenize

# finction opening the data
def open_data(directory):
    documents = {}
    labels = {}
    for file in os.listdir(directory):
        if file.endswith(".abstr"):
            content = open(("%s/%s" % (directory, file)), "r").read()
            content = re.sub('\s+',' ',content).lower()
            documents[file.split('.')[0]] = content
               
    for file in os.listdir(directory):
        if file.endswith(".uncontr"):
            content = open(("%s/%s" % (directory, file)), "r").read()
            content = re.sub('\s+',' ',content).lower()
            labels[file.split('.')[0]] = content.split("; ")
            
    return documents, labels

# tokenization function 
def tokenize(doc, labels):
    tokenized_doc = {}
    tokenized_labels = {}
    for key in doc.keys():
        tokenized_labels[key] = []
        text = doc[key]
        lab = labels[key]
        words_doc = nltk.word_tokenize(text)
        tokenized_doc[key] = words_doc
        for phrase in lab:
            words_phrase = nltk.word_tokenize(phrase)
            tokenized_labels[key].append(words_phrase)
    return tokenized_doc, tokenized_labels



def tag_document(tokenized_documents, tokenized_labels):
    # function creates a sequence of tags for the document: 
    # 0 for words not in keyphrase, 1 for first word in keyphrase.
    # 2 for not first word in keyphrase
    document_tagged = copy.deepcopy(tokenized_documents)
    for key in tokenized_documents.keys():
        doc = tokenized_documents[key]
        kps = tokenized_labels[key]
        document_tagged[key] = [0]*len(document_tagged[key])
        # cycle over keyphrases
        for kp in kps:
            # find indices of keyphrases in text
            idx = [(i, i+len(kp)) for i in range(len(doc)-len(kp)+1) if doc[i:i+len(kp)] == kp]
            # if keyphrase is not in abstract remove it 
            if len(idx)==0:
                tokenized_labels[key].remove(kp)

            # replace labels with 1,2 
            for j in range(len(idx)):
                document_tagged[key][idx[j][0]] = 1
                kp_len = idx[j][1]-idx[j][0]
                document_tagged [key][idx[j][0]+1:idx[j][1]] = [2]*(kp_len-1)
    return document_tagged, tokenized_labels


def data_to_seq(documents_eng, vocab_ind_dict):
    # replace words in our val data with their indices
    X = copy.deepcopy(documents_eng)
    for i, doc in enumerate(documents_eng):
        for j, token in enumerate(doc):
            X[i][j] = vocab_ind_dict [token]
    return X

def glove_emb_matrix(vocab_ind_dict, glove_dict, glove_size):
    # create embedding matrix using glove and our vocab
    embed_matrix = np.zeros((len(vocab_ind_dict), glove_size))
    for token in vocab_ind_dict.keys():
        idx = vocab_ind_dict[token]
        try: 
            embed_matrix[idx] = glove_dict[token]
        except KeyError:
            embed_matrix[idx] = np.zeros(shape=(glove_size))
    return embed_matrix 
            
def tags_to_3D(tags):
    # converting labels 0,1,2 to basis vectors (for keras)
    num_docs, len_doc = tags.shape
    tags_3d = np.zeros(shape = (num_docs, len_doc, 3))
    for i in range(num_docs):
        for j in range(len_doc):
            lab = tags[i][j]
            tags_3d[i][j][lab] = 1 
    return tags_3d
   