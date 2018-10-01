import numpy as np
from tqdm import tqdm
import difflib

def exist_in_ques(context_word,context_length,question_word,question_length):
    feat=np.zeros(context_word.shape)
    for i in tqdm(range(context_word.shape[0])):
        word_dict={}
        for j in range(question_length[i,0]):
            if question_word[i,j] not in word_dict:
                word_dict[question_word[i,j]]=1
        for j in range(context_length[i,0]):
            if context_word[i,j] in word_dict:
                feat[i,j]=1
    return feat

def jaccard_similarity(context_word,context_length,question_word,question_length):
    feat=np.zeros(context_word.shape)
    for i in tqdm(range(context_word.shape[0])):
        length=question_length[i,0]
        sub_q=question_word[i,0:length].astype(np.int32)
        for j in range(context_length[i,0]):
            sub_c=context_word[i,max(0,j-length//2):min(context_length[i,0],j+length//2)].astype(np.int32)
            feat[i,j]=len(np.intersect1d(sub_q,sub_c))/len(np.union1d(sub_q,sub_c))
    return feat

def levenshtein_dis(context_word,context_length,question_word,question_length):
    feat=np.zeros(context_word.shape)
    for i in tqdm(range(context_word.shape[0])):
        length=question_length[i,0]
        sub_q=question_word[i,0:length].astype(np.int32)
        for j in range(context_length[i,0]):
            sub_c=context_word[i,max(0,j-length//2):min(context_length[i,0],j+length//2)].astype(np.int32)
            leven_cost = 0
            s = difflib.SequenceMatcher(None, sub_c, sub_q)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                    if tag == 'replace':
                        leven_cost += ((i2 - i1)+ (j2 - j1))
                    elif tag == 'insert':
                        leven_cost += (j2 - j1)
                    elif tag == 'delete':
                        leven_cost += (i2 - i1)
            feat[i,j]=leven_cost/(len(sub_c)+len(sub_q))
    return feat

def max_similarity(context_word,context_length,question_word,question_length,embedding_matrix):
    feat=np.zeros(context_word.shape)
    for i in tqdm(range(context_word.shape[0])):
        a_vec=embedding_matrix[question_word[i,0:question_length[i,0]].astype(np.int32),:]
        c_vec=embedding_matrix[context_word[i,0:context_length[i,0]].astype(np.int32),:]
        mat1=np.dot(np.mat(c_vec),np.mat(a_vec).transpose())
        a_vec_norm=np.linalg.norm(a_vec,axis=1)
        c_vec_norm=np.linalg.norm(c_vec,axis=1)
        mat2=np.dot(np.mat(c_vec_norm).transpose(),np.mat(a_vec_norm))
        cos_dis=np.max(mat1/mat2,axis=1).reshape((1,-1))
        feat[i,0:context_length[i,0]]=cos_dis
    return feat

def feat_extract(data,embedding_matrix):
    context_word,question_word,context_char,question_char,context_length,question_length = data

    feat_max_similarity=np.expand_dims(max_similarity(context_word,context_length,question_word,question_length,embedding_matrix),axis=-1)

    handcraft_feat=feat_max_similarity

    return handcraft_feat


if __name__ == "__main__":


    context_word=np.load('dataset2/train_contw_input.npy')
    question_word=np.load('dataset2/train_quesw_input.npy')
    context_char=np.load('dataset2/train_contc_input.npy')
    question_char=np.load('dataset2/train_quesc_input.npy')
    context_length=(np.load('dataset2/train_cont_len.npy')).astype(np.int32)
    question_length=(np.load('dataset2/train_ques_len.npy')).astype(np.int32)
    train_data=[context_word,question_word,context_char,question_char,context_length,question_length]
    train_hand_feat=feat_extract(train_data,embedding_matrix)
    old_feat=np.load('dataset2/train_hand_feat.npy')
    train_hand_feat=np.concatenate((old_feat,train_hand_feat),axis=-1)
    print(train_hand_feat.shape)
    np.save('dataset2/train_hand_feat.npy',train_hand_feat)



    context_word=np.load('dataset2/dev_contw_input.npy')
    question_word=np.load('dataset2/dev_quesw_input.npy')
    context_char=np.load('dataset2/dev_contc_input.npy')
    question_char=np.load('dataset2/dev_quesc_input.npy')
    context_length=(np.load('dataset2/dev_cont_len.npy')).astype(np.int32)
    question_length=(np.load('dataset2/dev_ques_len.npy')).astype(np.int32)
    dev_data=[context_word,question_word,context_char,question_char,context_length,question_length]
    dev_hand_feat=feat_extract(dev_data,embedding_matrix)
    old_feat=np.load('dataset2/dev_hand_feat.npy')
    dev_hand_feat=np.concatenate((old_feat,dev_hand_feat),axis=-1)
    np.save('dataset2/dev_hand_feat.npy',dev_hand_feat)


    context_word=np.load('dataset2/test_contw_input.npy')
    question_word=np.load('dataset2/test_quesw_input.npy')
    context_char=np.load('dataset2/test_contc_input.npy')
    question_char=np.load('dataset2/test_quesc_input.npy')
    context_length=(np.load('dataset2/test_cont_len.npy')).astype(np.int32)
    question_length=(np.load('dataset2/test_ques_len.npy')).astype(np.int32)
    test_data=[context_word,question_word,context_char,question_char,context_length,question_length]
    test_hand_feat=feat_extract(test_data,embedding_matrix)
    old_feat=np.load('dataset2/test_hand_feat.npy')
    test_hand_feat=np.concatenate((old_feat,test_hand_feat),axis=-1)
    print(test_hand_feat.shape)
    np.save('dataset2/test_hand_feat.npy',test_hand_feat)
