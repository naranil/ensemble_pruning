"""
Functions to load the datasets used for our experiments
"""

# Author: Anil Narassiguin <narassiguin.anil@gmail.com>, Haytham Elghazel

from __future__ import division

"""
Use those functions to call databases.
This code is given only to show the reader how we construct our databases and store them in a HDFS file. You can
download all the datasets on : https://s3-eu-west-1.amazonaws.com/ensemble-comparison-data/datasets.h5
----------
UCI machine learning repository: http://archive.ics.uci.edu/ml/
"""

import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# Dataset dictionary. Add a data directory with the following datasets. Otherwise use the url paramater: direct link
# to a dataset.
datasets = {
            'basehock': {'path' : ['basehock/data.csv', 'basehock/target.csv']},

            'cleve' : { 'path' : 'cleve/processed.cleveland.data',
                        'url' : 'heart-disease/processed.cleveland.data'},

            'cnae9' : {'path' : 'cnae9/CNAE-9.data',
                       'url' : '00233/CNAE-9.data'},

            'madelon' : { 'path' : ['madelon/madelon_train.data', 'madelon/madelon_train.labels'],
                          'url' : ['madelon/MADELON/madelon_train.data', 'madelon/MADELON/madelon_train.labels']},

            'multiple' : {'path' : ['multiple/mfeat-fac.txt', 'multiple/mfeat-fou.txt', 'multiple/mfeat-kar.txt', \
                                    'multiple/mfeat-mor.txt', 'multiple/mfeat-pix.txt', 'multiple/mfeat-zer.txt'],
                          'url' : ['mfeat/mfeat-fac', 'mfeat/mfeat-fou', 'mfeat/mfeat-kar', 'mfeat-mor.txt', \
                                   'mfeat-pix.txt', 'mfeat-zer.txt']},

            'ovarian' : {'path' : 'ovarian/ovarian.txt'},

            'pancreatic' : {'path' : 'pancreatic/pancreatic.txt'},

            'parkinson' : { 'path' : 'parkinson/parkinsons.data',
                            'url' : 'parkinsons/parkinsons.data'},

            'pima' : { 'path' : 'pima/pima-indians-diabetes.data',
                       'url' : 'pima-indians-diabetes/pima-indians-diabetes.data'},

            'pcmac' : {'path' : ['pcmac/data.csv', 'pcmac/target.csv']},

            'promoters' : { 'path' : 'promoters/promoters.data',
                            'url' : 'molecular-biology/promoter-gene-sequences/promoters.data'},

            'robot' : {'path' : 'robot/robot.txt'},

            'spect' : { 'path' : ['spect/SPECT.train', 'spect/SPECT.test'],
                        'url' : ['spect/SPECT.train', 'spect/SPECT.test']},

            'soybean' : {'path' : 'soybean/soybean-large.data',
                         'url' : 'soybean/soybean-large.data'},

            'waveform' : {'path' : 'waveform/waveform.data'}
            }

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

## Replace by your data path
absolute_path = " "

def load_datasets_names():
    """
    :return: the names of all datasets used.
    """
    return datasets.keys()

def load_datasets(dataset_name):
    """
    :param dataset_name: dataset name (string)
    :return: (data: np array, target: np array)
    """
    name = re.sub('-', '_', dataset_name)
    return globals()['load_' + name]()

def load_basehock():
    path = datasets['basehock']['path']
    X = pd.read_csv(absolute_path + path[0], header=None)
    y = pd.read_csv(absolute_path + path[1], header=None)
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y[0]

def load_cnae9(url=False):
    path = UCI_URL + datasets['cnae9']['url'] if url else absolute_path + datasets['cnae9']['path']
    data = pd.read_csv(path, header=None)
    X = data.ix[:, 1:]
    y = data[0]
    return X, y

def load_cleve(url=False):
    path = UCI_URL + datasets['cleve']['url'] if url else absolute_path + datasets['cleve']['path']
    data = pd.read_csv(path, header=None)
    data = data[data[11] != '?']
    data = data[data[12] != '?']
    X = data.iloc[:, :13]
    y = data[13]
    y = y.replace([1, 2, 3, 4], 1)
    return X.astype(np.float), y

def load_madelon(url=False):
    path = [UCI_URL + p for p in datasets['madelon']['url']] if url else [absolute_path + p for p in \
                                                                          datasets['madelon']['path']]
    X = pd.read_csv(path[0], sep=" ", header=None)
    y = pd.read_csv(path[1], header=None)
    y = y.replace(-1, 0)
    return X.iloc[:, :500].astype(np.float), y[0]

def load_multiple(url=False):
    path = [UCI_URL + p for p in datasets['multiple']['url']] if url else [absolute_path + p for p in \
                                                                           datasets['multiple']['path']]
    fac = pd.read_csv(path[0], sep=" *", header=None, engine='python')
    fou = pd.read_csv(path[1], sep=" *", header=None, engine='python')
    kar = pd.read_csv(path[2], sep=" *", header=None, engine='python')
    mor = pd.read_csv(path[3], sep=" *", header=None, engine='python')
    pix = pd.read_csv(path[4], sep=" *", header=None, engine='python')
    zer = pd.read_csv(path[5], sep=" *", header=None, engine='python')
    X = pd.concat([fac, fou, kar, mor, pix, zer], axis=1)
    y = np.array([[i for _ in range(200)] for i in range(10)]).reshape(1, 2000)[0]
    y = pd.Series(y)
    return X, y

def load_ovarian():
    path = absolute_path + datasets['ovarian']['path']
    data = pd.read_csv(path, sep="\t", header=None)
    X = data.iloc[:, :1536]
    y = data[1536]
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y

def load_pancreatic():
    path = absolute_path + datasets['pancreatic']['path']
    data = pd.read_csv(path, sep="\t", header=None)
    X = data.ix[:, :6770]
    y = data[6771]
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X, y

def load_parkinson(url=False):
    path = UCI_URL + datasets['parkinson']['url'] if url else absolute_path + datasets['parkinson']['path']
    data = pd.read_csv(path)
    X = data.drop(['status', 'name'], axis=1)
    y = data['status']
    return X.astype(np.float), y

def load_pcmac():
    path = [absolute_path + p for p in datasets['pcmac']['path']]
    X = pd.read_csv(path[0], header=None)
    y = pd.read_csv(path[1], header=None)
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    return X.astype(np.float), y[0]

def load_promoters(url=False):
    path = UCI_URL + datasets['promoters']['url'] if url else absolute_path + datasets['promoters']['path']
    data = pd.read_csv(path, header=None)
    features = data[2].str.replace('\t', '').values
    features = np.array([list(x) for x in features])
    lb = LabelBinarizer()
    lb.fit(['a', 't', 'c', 'g'])
    features = tuple([lb.transform(features[:, i]) for i in range(features.shape[1])])
    X = np.concatenate(features, axis=1)
    y = data[0]
    y = y.replace('+', 1)
    y = y.replace('-', 0)
    y = np.ravel(y)
    return pd.DataFrame(X.astype(np.float)), pd.Series(y)

def load_robot():
    path = absolute_path + datasets['robot']['path']
    data = pd.read_csv(path, sep=" *", engine='python', header=None)
    y = data[90]
    X = data.ix[:, :89]
    return X, y

def load_soybean(url=False):
    path = UCI_URL + datasets['soybean']['url'] if url else absolute_path + datasets['soybean']['path']
    data = pd.read_csv(path, header=None)
    for col in range(data.shape[1]-1):
        data = data[data[col].astype(object) != '?']
    y = data[0]
    X = data.ix[:, 1:]
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    return X, pd.Series(y)

def load_spect(url=False):
    path = [UCI_URL + p for p in datasets['spect']['url']] if url else [absolute_path + p for p in \
                                                                        datasets['spect']['path']]
    train = pd.read_csv(path[0], header=None)
    test = pd.read_csv(path[1], header=None)
    data = pd.concat([train, test], axis=0)
    X = data.iloc[:, :22]
    y = data[22]
    return X.astype(np.float), y

def load_waveform():
    path = absolute_path + datasets['waveform']['path']
    data = pd.read_csv(path, header=None)
    y = data[21]
    X = data.ix[:, :20]
    return X, y

if __name__ == "__main__":
    """
    Tests
    """

    #X, y = load_basehock()
    #X, y = load_cleve()
    #X, y = load_madelon()
    #X, y = load_multiple()
    #X, y = load_ovarian()
    #X, y = load_cnae9()
    #X, y = load_ovarian()
    #X, y = load_pancreatic()
    #X, y = load_parkinson()
    #X, y = load_pcmac()
    #X, y = load_promoters()
    #X, y = load_robot()
    #X, y = load_soybean()
    X, y = load_waveform()

    print X
    print ""
    print y




