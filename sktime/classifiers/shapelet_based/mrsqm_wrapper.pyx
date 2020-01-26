from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sktime.transformers.dictionary_based.SFA import SFA
from sktime.transformers.dictionary_based.SFA import SAX
from sktime.classifiers.base import BaseClassifier

######################### SAX and SFA #########################

cdef extern from "mrseql_cpp/sax_converter.h":
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