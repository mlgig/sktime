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
# from sklearn.naive_bayes import BernoulliNB
import catch22
from scipy import stats
from scipy.signal import find_peaks
from sklearn import preprocessing

import logging
from datetime import datetime
#########################SEQL wrapper#########################


cdef extern from "mytrie.cpp":
    cdef cppclass SearchTrie:
        SearchTrie(vector[string])
        vector[int] search(string)


cdef extern from "sqminer.h":
    cdef cppclass SQMiner:
        SQMiner(double, double)        
        # void learn(vector[string] &, vector[double] &)
        # double brute_classify(string , double)
        # void print_model(int)
        vector[string] mine(vector[string] &, vector[int] &)
        # vector[double] get_coefficients(bool)


cdef class PyFeatureTrie:
    cdef SearchTrie *thisptr

    def __cinit__(self, vector[string] sequences):
        self.thisptr = new SearchTrie(sequences)
    def __dealloc__(self):
        del self.thisptr

    def search(self, string sequence):
        return self.thisptr.search(sequence)


cdef class PySQM:
    cdef SQMiner *thisptr

    def __cinit__(self, double selection, double threshold):
        self.thisptr = new SQMiner(selection,threshold)
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

# class SEQLLogger():
#     def __init__(self, not_fake = True, logging_level = logging.INFO):
#         self.not_fake = not_fake
#         if self.not_fake:
#             logging.basicConfig(filename=datetime.now().strftime("%y%m%d%H%M") + ".log",
#                                 format='%(asctime)s %(levelname)-8s %(message)s',
#                                 level=logging_level,
#                                 datefmt='%Y-%m-%d %H:%M:%S')
#     def info(self, message):
#         if self.not_fake:
#             logging.info(message)
#     def warning(self, message):
#         if self.not_fake: 
#             logging.warning(message)
#     def debug(self, message):
#         if self.not_fake:
#             logging.debug(message)


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

        # self.logger = SEQLLogger()

        
     

    def create_pars(self, min_ws, max_ws, random_sampling=False):
        
        if random_sampling:
            pars = []
            ws_choices = [i for i in range(10,max_ws+1)]
            wl_choices = [6,8,10,12,14,16]
            alphabet_choices = [3,4,5,6] 
            if max_ws > min_ws:                
                for w in range(min_ws, max_ws, np.max((1,int(np.sqrt(max_ws))))): # to make sure it has the same number of reps
                    pars.append([np.random.choice(ws_choices) , np.random.choice(wl_choices), np.random.choice(alphabet_choices)])
            else:
                pars.append([np.random.choice(ws_choices) , np.random.choice(wl_choices), np.random.choice(alphabet_choices)])
        else:           
            if max_ws > min_ws:
                pars = [[w, 16, 4] for w in range(min_ws, max_ws, int(np.sqrt(max_ws)))]                
            else:
                pars = [[max_ws, 16, 4]]
        
        return pars

    def compute_stats(self, X):
        logging.info("Computing stats ...")
        df = pd.DataFrame()
        
        # min
        df['min'] = X.applymap(np.min)['dim_0']

        # max
        df['max'] = X.applymap(np.max)['dim_0']

        # avg
        df['avg'] = X.applymap(np.mean)['dim_0']

        # median
        df['median'] = X.applymap(np.median)['dim_0']

        # kurtosis
        df['kurtosis'] = X.applymap(stats.kurtosis)['dim_0']

        # peaks: number of peaks (instead of first peak)
        df['peaks'] = X.applymap(lambda x: len(find_peaks(x)))['dim_0']

        # skew
        df['skew'] = X.applymap(stats.skew)['dim_0']

        # standard deviation
        df['std'] = X.applymap(np.std)['dim_0']

        # first quartile
        df['p25'] = X.applymap(lambda x: stats.scoreatpercentile(x, 25))['dim_0']
        
        # third quartile
        df['p75'] = X.applymap(lambda x: stats.scoreatpercentile(x, 75))['dim_0']

        # catch22
        # features = [
        # 'DN_HistogramMode_5',
        # 'DN_HistogramMode_10',
        # 'CO_f1ecac',
        # 'CO_FirstMin_ac',
        # 'CO_HistogramAMI_even_2_5',
        # 'CO_trev_1_num',
        # 'MD_hrv_classic_pnn40',
        # 'SB_BinaryStats_mean_longstretch1',
        # 'SB_TransitionMatrix_3ac_sumdiagcov',
        # 'PD_PeriodicityWang_th0_01',
        # 'CO_Embed2_Dist_tau_d_expfit_meandiff',
        # 'IN_AutoMutualInfoStats_40_gaussian_fmmi',
        # 'FC_LocalSimple_mean1_tauresrat',
        # 'DN_OutlierInclude_p_001_mdrmd',
        # 'DN_OutlierInclude_n_001_mdrmd',
        # 'SP_Summaries_welch_rect_area_5_1',
        # 'SB_BinaryStats_diff_longstretch0',
        # 'SB_MotifThree_quantile_hh',
        # 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
        # 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
        # 'SP_Summaries_welch_rect_centroid',
        # 'FC_LocalSimple_mean3_stderr'
        # ]

        # for f in features:
        #     featureFun = getattr(catch22, f)
        #     df[f] = X.applymap(lambda x: featureFun(x.tolist()))['dim_0']
        #     if df[f].isnull().values.any():
        #         df[f].fillna(df[f].mean())
      

        return df




    def compute_optimal_chi2(self, df, y):
        logging.info("Computing split ...")
        # 	int feature_count = 0;
	# for (int c : observed){
	# 	feature_count += c;
	# }


	# double chisq_score = 0.0;
	# for (int i = 0; i < observed.size(); i++){
	# 	double expected = y_prob[i] * feature_count;

	# 	chisq_score += pow(observed[i] - expected,2.0) / expected;
	# }


	# return chisq_score;
        labels = np.unique(y)
        y_prob = {}
        for l in labels:
            y_prob[l] = np.sum(y == l)/len(y)
        
        rt = []

        # for i in range(df.shape[1]):
        #     vec = df.iloc[:,i]
        #     max_chi2 = 0.0
        #     selected_t = np.min(vec)
        #     for t in np.sort(vec): # each value in the data as candidate threshold
        #         chi2 = 0.0
        #         feature_count = np.sum(vec > t)
        #         if feature_count > 0:
        #             for l in labels:
        #                 expected = y_prob[l] * feature_count
        #                 observed = np.sum(vec[y == l] > t)
        #                 chi2 += ((observed - expected)**2)/expected
        #             if chi2 >= max_chi2:
        #                 max_chi2 = chi2
        #                 selected_t = t
        #     rt.append([selected_t, max_chi2])
        # return rt

        # speed up
        for i in range(df.shape[1]):
           
            vec = df.iloc[:,i]
            max_chi2 = 0.0
            si = np.argsort(vec) # may reverse to sort descendingly
            last_label = None
            feature_count = 0
            y_count_so_far = {}
            selected_t = vec[si[0]]
            for l in labels:
                y_count_so_far[l] = 0
            for j in range(0,len(si)-1):
                t = si[j]
                next_t = si[j+1]
                y_count_so_far[y[t]] += 1
                feature_count = feature_count + 1               
                if y[t] != y[next_t]: #only compute chi2 when the label changes
                    chi2 = 0.0                         
                    for l in labels:
                        expected = y_prob[l] * feature_count
                        observed = y_count_so_far[l]
                        chi2 += ((observed - expected)**2)/expected
                    if chi2 >= max_chi2:
                        max_chi2 = chi2
                        selected_t = vec[t]
                    last_label = y[t]
            rt.append([selected_t, max_chi2])

        return rt



            
                


            
            
            

    def transform_time_series(self, ts_x):
        logging.info("Transforming time series ...")
        
        multi_tssr = []   
     
        if not self.config:
            self.config = []
            
            min_ws = 16
            min_len = max_len = len(ts_x.iloc[0, 0])
            for a in ts_x.iloc[:, 0]:
                min_len = min(min_len, len(a)) 
                max_len = max(max_len, len(a))
            max_ws = (min_len + max_len)//2

            pars = self.create_pars(min_ws, max_ws, False)
            
            if 'sax' in self.symrep:
                for p in pars:
                    self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2], 
                        # 'dilation': np.int32(2 ** np.random.uniform(0, np.log2((min_len - 1) / (p[0] - 1))))})
                        'dilation': 1})

            if 'sfa' in self.symrep:
                for p in pars:
                    self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': p[1], 'alphabet': p[2]})       

        
        for cfg in self.config:
            for i in range(ts_x.shape[1]):
                tssr = []

                if cfg['method'] == 'sax':  # convert time series to SAX                    
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'], cfg['dilation'])
                    for ts in ts_x.iloc[:,i]:
                        sr = ps.timeseries2SAXseq(ts)
                        tssr.append(sr)
                    

                if cfg['method'] == 'sfa':  # convert time series to SFA
                    if (cfg['window'], cfg['word'], cfg['alphabet']) not in self.sfas:
                        sfa = AdaptedSFA(
                            cfg['window'], cfg['word'], cfg['alphabet'])
                        sfa.fit(ts_x.iloc[:,[i]])
                        self.sfas[(cfg['window'], cfg['word'],
                                cfg['alphabet'])] = sfa
                    for ts in ts_x.iloc[:,i]:
                        sr = self.sfas[(cfg['window'], cfg['word'],
                                        cfg['alphabet'])].timeseries2SFAseq(ts.values)
                        tssr.append(sr)

                multi_tssr.append(tssr)

        

        return multi_tssr


 


    # represent data (in multiple reps form) in feature space
    # def __to_feature_space(self, mr_seqs):
    #     logging.info("Computing feature vectors ...")
    #     full_fm = []

    #     for rep, seq_features in zip(mr_seqs, self.sequences):            
    #         fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
    #         for i,s in enumerate(rep):
    #             for j,f in enumerate(seq_features):
    #                 if f in s:
    #                     fm[i,j] = 1
    #         full_fm.append(fm)

  
    #     full_fm = np.hstack(full_fm)
    #     return full_fm

    def __to_feature_space(self, mr_seqs):
        logging.info("Computing feature vectors ...")
        full_fm = []

        for rep, seq_features in zip(mr_seqs, self.sequences):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            ft = PyFeatureTrie(seq_features)
            for i,s in enumerate(rep):
                fm[i,:] = ft.search(s)
            full_fm.append(fm)

  
        full_fm = np.hstack(full_fm)
        return full_fm > 0


    def fit(self, X, y, input_checks=True):
        

        # X = self.__X_check(X)

        # transform time series to multiple symbolic representations

        mr_seqs = self.transform_time_series(X)
        
        # print(mr_seqs)
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        for rep in mr_seqs:
            miner = PySQM(self.selection,0.0)
            mined = miner.mine(rep, int_y)
            self.sequences.append(mined)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        train_x = self.__to_feature_space(mr_seqs)
        # print(train_x)
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
        # self.clf = BernoulliNB().fit(train_x,y)
        self.classes_ = self.clf.classes_ # shouldn't matter
    
    def fit_random_selection(self, X, y, ext_reps = None, input_checks=True):
    # '''
    # random selection after mining
    # '''
        # max_selected = self.selection * 3

        # transform time series to multiple symbolic representations
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []
        mr_seqs = []

        if X is not None:
            mr_seqs = self.transform_time_series(X)
        if ext_reps is not None:
            mr_seqs.extend(ext_reps)

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

    def convert_stat_features(self, stat_x):
        # apply split to convert stats to binary features
        # for i in range(stat_x.shape[1]):
        #     stat_x.iloc[:,i] = stat_x.iloc[:,i] > self.opchi2[i,0]
        
        # minmax-normalize
        # min_max_scaler = preprocessing.MinMaxScaler()
        # np_scaled = min_max_scaler.fit_transform(stat_x)
        # stat_x = pd.DataFrame(np_scaled, columns = stat_x.columns)


        return stat_x

    def fit_stats(self, X, y, input_checks = False):

        logging.info("---Start training---")

        

        stat_x = self.compute_stats(X)
        
        self.opchi2 = np.array(self.compute_optimal_chi2(stat_x,y))
        # print(stat_x)
        # print(self.opchi2)
        chi2_threshold = np.percentile(self.opchi2[:,1],75)
        stat_x = self.convert_stat_features(stat_x)
        
            
        



        
        

        mr_seqs = self.transform_time_series(X)
        
        # print(mr_seqs)
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        logging.info("Searching for sequences ...")
        for rep in mr_seqs:
            miner = PySQM(self.selection,chi2_threshold)
            mined = miner.mine(rep, int_y)
            self.sequences.append(mined)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        

        train_x = np.hstack((stat_x, self.__to_feature_space(mr_seqs)))
        # train_x = stat_x

        logging.info("Training logreg ...")
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)
        # self.clf = BernoulliNB().fit(train_x,y)
        self.classes_ = self.clf.classes_ # shouldn't matter

    def predict_stats(self, X):
        logging.info("---Start testing---")
        stat_x = self.compute_stats(X)        
        stat_x = self.convert_stat_features(stat_x)

        mr_seqs = self.transform_time_series(X)
        test_x = np.hstack((stat_x, self.__to_feature_space(mr_seqs)))
        # test_x = stat_x
        return self.clf.predict(test_x)

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
        # self.clf = 
        self.classes_ = self.clf.classes_ # shouldn't matter

    
    

    def predict_proba(self, X, input_checks=True):
        
        mr_seqs = self.transform_time_series(X)
        test_x = self.__to_feature_space(mr_seqs)
        return self.clf.predict_proba(test_x) 

    def predict(self, X, ext_reps = None, input_checks=True):        
        
        mr_seqs = []
        if X is not None:
            mr_seqs = self.transform_time_series(X)
        if ext_reps is not None:
            mr_seqs.extend(ext_reps)

        test_x = self.__to_feature_space(mr_seqs)
        return self.clf.predict(test_x)

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

 






