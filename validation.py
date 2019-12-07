
import numpy as np
import copy


def recall(kp_true, kp_predicted):
    tp = 0
    fn = 0
    for i in kp_true:
        if (i in kp_predicted):
            tp = tp + 1
        else:
            fn = fn + 1
    if (float(tp)+float(fn))!=0:
        return float(tp)/(float(tp)+float(fn)) 


def precision(kp_true, kp_predicted):
    tp = 0
    fp = 0
    for i in kp_predicted:
        if(i in kp_true):
            tp = tp + 1
        else:
            fp  = fp + 1
    
    return float(tp)/(float(tp)+float(fp))  if (float(tp)+float(fp)) > 0 else 0

def f1(kp_true, kp_predicted):
    precision_ = precision(kp_true, kp_predicted)
    recall_ = recall(kp_true, kp_predicted)
    return (2 * (precision_ * recall_)) / (precision_ + recall_) if precision_ + recall_ > 0 else 0


def retrive_phrase(tags_predicted, document_eng):
    #from tag 0,1,2 to keyphrase arrays (2d-sentences, words)
    kp = []
    sentence= []
    for i in range(len(tags_predicted)):
        if (tags_predicted[i] == 0):
            if len(sentence) != 0:
                kp.append(copy.deepcopy(sentence))
                
            sentence.clear()#we know a sentence ends when we encounter 0, so push
        elif((tags_predicted[i] == 1)):
            if len(sentence) != 0:
                kp.append(copy.deepcopy(sentence))
            sentence.clear()#we know a sentence ends when we encounter 1, so push
 
            sentence.append(document_eng[i])
            if(i== len(tags_predicted)-1):#if it is the last element, we push
                kp.append(copy.deepcopy(document_eng))
        else:
            sentence.append(document_eng[i])
            if(i== len(tags_predicted)-1):
                kp.append(copy.deepcopy(sentence))
    return kp
