from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sktime.transformers.dictionary_based.SFA import SFA
#from sktime.transformers.dictionary_based.SFA import SAX
from sktime.classifiers.base import BaseClassifier
from sklearn.utils import resample

######################### SAX and SFA #########################

cdef extern from "sqm_cpp/sax_converter.h":
    cdef cppclass SAX:
        SAX(int, int, int)
        #string timeseries2SAX(string, string)
        vector[string] timeseries2SAX(vector[double])
        vector[double] map_weighted_patterns(vector[double], vector[string], vector[double])

cdef class PySAX:
    cdef SAX *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int N, int w, int a):
        self.thisptr = new SAX(N, w, a)
    def __dealloc__(self):
        del self.thisptr
    def timeseries2SAX(self, ts):
        return self.thisptr.timeseries2SAX(ts)
        #if isinstance(obj, basestring):
        #    return self.thisptr.timeseries2SAX(ts, delimiter)
    def timeseries2SAXseq(self, ts):
        words = self.thisptr.timeseries2SAX(ts)
        seq = b''
        #print(words)
        for w in words:
            seq = seq + b' ' + w
        if seq: # remove extra space
            seq = seq[1:]
        return seq
    def map_weighted_patterns(self, ts, sequences, weights):
        return self.thisptr.map_weighted_patterns(ts, sequences, weights)

class AdaptedSFA:
    def __init__(self, int N, int w, int a):
        self.sfa = SFA(w,a,N,norm=True,remove_repeat_words=True)

    def fit(self, train_x):
        self.sfa.fit(train_x)

    def timeseries2SFAseq(self, ts):
        dfts = self.sfa.MFT(ts)
        sfa_str = b''
        for window in range(dfts.shape[0]):
            if sfa_str:
                sfa_str += b' '
            dft = dfts[window]
            first_char = ord(b'A')
            for i in range(self.sfa.word_length):
                for bp in range(self.sfa.alphabet_size):
                    if dft[i] <= self.sfa.breakpoints[i][bp]:
                        sfa_str += bytes([first_char + bp])
                        #print(chr(first_char + bp))
                        break
                first_char += self.sfa.alphabet_size
        return sfa_str

#########################SEQL wrapper#########################


cdef extern from "sqm_cpp/sqminer.h":
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

    def __init__(self, selection = 100, symrep=['sax'], symrepconfig=None, n_samples = 5, sample_rate = 0.5):

        self.symbolic_methods = symrep

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

        self.n_samples = 5
        self.sample_rate = 0.5


    def __transform_time_series(self, ts_x):
        multi_tssr = []

        # generate configuration if not predefined
        if not self.config:
            min_ws = 16
            max_ws = ts_x.shape[1]
            pars = [[w, 16, 4] for w in range(min_ws, max_ws, int(np.sqrt(max_ws)))]

            if 'sax' in self.symbolic_methods:
                for p in pars:
                    self.config.append({'method':'sax','window':p[0],'word':p[1],'alphabet':p[2]})

            if 'sfa' in self.symbolic_methods:
                for p in pars:
                    self.config.append({'method':'sfa','window':p[0],'word':8,'alphabet':p[2]})


        for cfg in self.config:

            tssr = []

            if cfg['method'] == 'sax': # convert time series to SAX
                ps = PySAX(cfg['window'],cfg['word'],cfg['alphabet'])
                for ts in ts_x:
                    sr = ps.timeseries2SAXseq(ts)
                    tssr.append(sr)

            if cfg['method'] == 'sfa':  # convert time series to SFA
                if (cfg['window'],cfg['word'],cfg['alphabet']) not in self.sfas:
                    sfa = AdaptedSFA(cfg['window'],cfg['word'],cfg['alphabet'])
                    sfa.fit(ts_x)
                    self.sfas[(cfg['window'],cfg['word'],cfg['alphabet'])] = sfa
                for ts in ts_x:
                    sr = self.sfas[(cfg['window'],cfg['word'],cfg['alphabet'])].timeseries2SFAseq(ts)
                    tssr.append(sr)

            multi_tssr.append(tssr)


        return multi_tssr


 


    # represent data (in multiple reps form) in feature space
    def __to_feature_space(self, mr_seqs):
        full_fm = []

        for rep, seq_features in zip(mr_seqs, self.sequences):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.bool)
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

    def _resample(self, y):
        train_x = [i for i in range(0,len(y))]

        sample_size = self.sample_rate * len(y)

        resample_indices = []

        for i in range(0,self.n_samples):
            selected, _ = resample(train_x, y,n_samples=sample_size, stratify=y)
            resample_indices.append(selected)
        
        return resample_indices



    def fit(self, X, y, input_checks=True):

        X = self.__X_check(X)

        # transform time series to multiple symbolic representations
        mr_seqs = self.__transform_time_series(X)

        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        resample_indices = self._resample(int_y)

        self.sequences = []

        for rep in mr_seqs:
            seqs = []
            for sample in resample_indices:
                new_rep = [rep[i] for i in sample]
                new_y = [int_y[i] for i in sample]
                miner = PySQM(self.selection)
                seqs.extend(miner.mine(new_rep, new_y))
            seqs = np.unique(seqs)
            self.sequences.append(seqs)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        train_x = self.__to_feature_space(mr_seqs)
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
        self.classes_ = self.clf.classes_ # shouldn't matter

    def fit_ova(self, X, y, input_checks=True):
        X = self.__X_check(X)

        # transform time series to multiple symbolic representations
        mr_seqs = self.__transform_time_series(X)

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
        if input_checks:
            X = self.__X_check(X)
        mr_seqs = self.__transform_time_series(X)
        test_x = self.__to_feature_space(mr_seqs)
        return self.clf.predict_proba(test_x) 

    def predict(self, X, input_checks=True):
        if input_checks:
            X = self.__X_check(X)
        proba = self.predict_proba(X, False)
        return np.array([self.classes_[np.argmax(prob)] for prob in proba])

    def get_all_sequences(self):
        return self.sequences

 






