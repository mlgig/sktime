from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.linear_model import LogisticRegression

from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.classification.shapelet_based.mrseql.mrseql import PySAX
from sktime.classification.shapelet_based.mrseql.mrseql import AdaptedSFA
#from sktime.transformers.dictionary_based.SFA import SAX
from sktime.classification.base import BaseClassifier



#########################SEQL wrapper#########################


cdef extern from "sqminer.h":
    cdef cppclass SQMiner:
        SQMiner(double)
        # void learn(vector[string] &, vector[double] &)
        # double brute_classify(string , double)
        # void print_model(int)
        vector[string] mine(vector[string] &, vector[int] &)
        # vector[double] get_coefficients(bool)




cdef class PySQM:
    cdef SQMiner *thisptr

    def __cinit__(self, double selection):
        self.thisptr = new SQMiner(selection)
    def __dealloc__(self):
        del self.thisptr

    def mine(self, vector[string] sequences, vector[int] labels):
        return self.thisptr.mine(sequences, labels)

    # def classify(self, string sequence):
    #     scr = self.thisptr.brute_classify(sequence, 0.0)
    #     return np.array([-scr,scr]) # keep consistent with multiclass case

    # def print_model(self):
    #     self.thisptr.print_model(100)

    # def get_sequence_features(self, bool only_positive = False):
    #     return self.thisptr.get_sequence_features(only_positive)

    # def get_coefficients(self, bool only_positive = False):
    #     return self.thisptr.get_coefficients(only_positive)


class MrSQMClassifier(BaseClassifier):

    # selection <= 0: brute force selection
    # selection > 0 and selection < 1 : chisquared test with p value threshold = selection
    # selection >= 1: top k selection with k = int(selection)    

    def __init__(self, selection = 100, max_selection = 300, symrep=['sax'], symrepconfig=None):

        self.symrep = symrep

        if symrepconfig is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = symrepconfig

        

        #self.label_dict = {} # for translating labels since seql only accept [1.-1] as labels

        # all the unique labels in the data
        # in case of binary data the first one is always the negative class
        self.classes_ = []



        self.clf = None # scikit-learn model

        # store fitted sfa for later transformation
        self.sfas = {}

        self.selection = selection
        self.max_selection = max_selection




    def transform_time_series(self, ts_x):
        multi_tssr = []   


        # generate configuration if not predefined
        if not self.config:
            self.config = []
            min_ws = 16
            min_len = max_len = len(ts_x.iloc[0, 0])
            for a in ts_x.iloc[:, 0]:
                min_len = min(min_len, len(a)) 
                max_len = max(max_len, len(a))
            max_ws = (min_len + max_len)//2

            if min_ws < max_ws: 
                pars = [[w, 16, 4] for w in range(min_ws, max_ws, int(np.sqrt(max_ws)))]
            else:
                pars = [[max_ws, 16, 4]]

            if 'sax' in self.symrep:
                for p in pars:
                    self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2]})

            if 'sfa' in self.symrep:
                for p in pars:
                    self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': 8, 'alphabet': p[2]})

        
        for cfg in self.config:
            for dim in ts_x:
                tssr = []

                if cfg['method'] == 'sax':  # convert time series to SAX
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                    for ts in ts_x[dim]:
                        sr = ps.timeseries2SAXseq(ts)
                        tssr.append(sr)

                if cfg['method'] == 'sfa':  # convert time series to SFA
                    if (cfg['window'], cfg['word'], cfg['alphabet']) not in self.sfas:
                        sfa = AdaptedSFA(
                            cfg['window'], cfg['word'], cfg['alphabet'])
                        sfa.fit(pd.DataFrame(ts_x[dim]))
                        self.sfas[(cfg['window'], cfg['word'],
                                cfg['alphabet'])] = sfa
                    for ts in ts_x[dim]:
                        sr = self.sfas[(cfg['window'], cfg['word'],
                                        cfg['alphabet'])].timeseries2SFAseq(ts)
                        tssr.append(sr)

                multi_tssr.append(tssr)

        return multi_tssr


 


    # represent data (in multiple reps form) in feature space
    def __to_feature_space(self, mr_seqs):
        full_fm = []

        for rep, seq_features in zip(mr_seqs, self.sequences):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            for i,s in enumerate(rep):
                for j,f in enumerate(seq_features):
                    if f in s:
                        fm[i,j] = 1
            full_fm.append(fm)

  
        full_fm = np.hstack(full_fm)
        return full_fm

    '''
    Check if X input is correct.
    From dictionary_based/boss.py
    '''
    def __X_check(self,X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("Mr-SEQL cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0, 0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects.")
        return X

    def fit(self, X, y, input_checks=True):
        

        # X = self.__X_check(X)

        # transform time series to multiple symbolic representations
        mr_seqs = self.transform_time_series(X)

        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        for rep in mr_seqs:
            miner = PySQM(self.selection)
            mined = miner.mine(rep, int_y)
            self.sequences.append(mined)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        # train_x = self.__to_feature_space(mr_seqs)
        # self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
        # self.classes_ = self.clf.classes_ # shouldn't matter
    
    def fit_random_selection(self, X, y, input_checks=True):
    # '''
    # random selection after mining
    # '''
        # max_selected = self.selection * 3

        X = self.__X_check(X)

        # transform time series to multiple symbolic representations
        mr_seqs = self.transform_time_series(X)

        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        for rep in mr_seqs:
            miner = PySQM(self.max_selection)
            mined = miner.mine(rep, int_y)       
            # print(len(mined))     
            random_selected = np.random.permutation(mined)[:self.selection].tolist()
            # print(len(random_selected))
            self.sequences.append(random_selected)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        train_x = self.__to_feature_space(mr_seqs)
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
        self.classes_ = self.clf.classes_ # shouldn't matter

    def fit_ova(self, X, y, input_checks=True):
        X = self.__X_check(X)

        # transform time series to multiple symbolic representations
        mr_seqs = self.transform_time_series(X)

        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        #int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        for rep in mr_seqs:
            c_sequences = []
            for c in self.classes_:
                tmp_y = [1 if l == c else 0 for l in y]
                miner = PySQM(self.selection)
                c_sequences.extend(miner.mine(rep, tmp_y))
            self.sequences.append(np.unique(c_sequences))


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        train_x = self.__to_feature_space(mr_seqs)
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
        self.classes_ = self.clf.classes_ # shouldn't matter

    

    def predict_proba(self, X, input_checks=True):
        
        mr_seqs = self.transform_time_series(X)
        test_x = self.__to_feature_space(mr_seqs)
        return self.clf.predict_proba(test_x) 

    def predict(self, X, input_checks=True):
        
        proba = self.predict_proba(X, False)
        return np.array([self.classes_[np.argmax(prob)] for prob in proba])

    def get_all_sequences(self):
        return self.sequences

    def to_csv(self, X,y, target_file):
        mr_seqs = self.transform_time_series(X)
        vs_x = self.__to_feature_space(mr_seqs)
        np.savetxt(target_file, np.hstack((np.reshape(y,(len(y),1)),vs_x)), fmt='%i', delimiter=",")

    def get_coverage(self, X):
        

        # transform time series to multiple symbolic representations
        mr_seqs = self.transform_time_series(X)

        n_ts = X.shape[0]

        coverage = np.zeros(n_ts)

        for fts, rep in zip(self.sequences, mr_seqs):
            for i in range(0,n_ts):
                for f in fts:
                    if f in rep[i]:
                        coverage[i] += 1


        return coverage

 






