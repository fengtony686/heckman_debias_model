import pandas as pd
from sklearn import preprocessing
from scipy.sparse import coo_matrix
import numpy as np


def quora_leaky_extracting(concat):
    tid1 = concat['q1_id'].values
    tid2 = concat['q2_id'].values
    doc_number = np.max((tid1.max(), tid2.max())) + 1
    adj = coo_matrix((np.ones(len(tid1) * 2), (np.concatenate(
        [tid1, tid2]), np.concatenate([tid2, tid1]))), (doc_number,  doc_number))
    degree = adj.sum(axis=0)
    concat['q1_id_degree'] = concat['q1_id'].apply(lambda x: degree[0, x])
    concat['q2_id_degree'] = concat['q2_id'].apply(lambda x: degree[0, x])
    tmp = adj * adj
    concat['path'] = concat.apply(
        lambda row: tmp[int(row['q1_id']), int(row['q2_id'])], axis=1)
    return concat


def load_quora(path='./quora'):
    print('---------- Loading QuoraQP ----------')
    tr = pd.read_csv(path + '/train.tsv', delimiter='\t', header=None)
    tr.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
    val = pd.read_csv(path + '/dev.tsv', delimiter='\t', header=None)
    val.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
    te = pd.read_csv(path + '/test.tsv', delimiter='\t', header=None)
    te.columns = ['is_duplicate', 'question1', 'question2', 'pair_id']
    data = pd.concat([tr, val, te]).fillna('')
    questions = list(data['question1'].values) + list(data['question2'].values)
    le = preprocessing.LabelEncoder()
    le.fit(questions)
    data['q1_id'] = le.transform(data['question1'].values)
    data['q2_id'] = le.transform(data['question2'].values)
    data = quora_leaky_extracting(data)
    label = data["is_duplicate"].to_numpy()
    s1_freq = data["q1_id_degree"].to_numpy()
    s2_freq = data["q2_id_degree"].to_numpy()
    s1s2_inter = data["path"].to_numpy()
    X = pd.DataFrame({
        "s1_freq": s1_freq,
        "s2_freq": s2_freq,
        "s1s2_inter": s1s2_inter
    })
    Y = label
    print('Success!')
    return X, Y


def load_artificial_dataset(path='./artificial_dataset'):
    print('---------- Loading artificial dataset ----------')
    tr = pd.read_csv(path + '/train.tsv', delimiter='\t', header=None)
    tr.columns = ['is_duplicate', 'question1', 'question2']
    val = pd.read_csv(path + '/dev.tsv', delimiter='\t', header=None)
    val.columns = ['is_duplicate', 'question1', 'question2']
    te = pd.read_csv(path + '/test.tsv', delimiter='\t', header=None)
    te.columns = ['is_duplicate', 'question1', 'question2']
    data = pd.concat([tr, val, te]).fillna('')
    questions = list(data['question1'].values) + list(data['question2'].values)
    le = preprocessing.LabelEncoder()
    le.fit(questions)
    data['q1_id'] = le.transform(data['question1'].values)
    data['q2_id'] = le.transform(data['question2'].values)
    data = quora_leaky_extracting(data)
    label = data["is_duplicate"].to_numpy()
    s1_freq = data["q1_id_degree"].to_numpy()
    s2_freq = data["q2_id_degree"].to_numpy()
    s1s2_inter = data["path"].to_numpy()
    X = s2_freq
    Y = label
    print('Success!')
    return X, Y
