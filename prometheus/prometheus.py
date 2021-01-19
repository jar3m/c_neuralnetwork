#!/usr/bin/python3
import math, random
import sys, getopt
import json
from ctypes import *
import os
import csv
import time


# TODO Code to be deprecated
"""
class t_neuron(Structure):
   _fields_ = [("bypass", c_bool), ("weight",c_float), ("actv_fn",CFUNCTYPE(c_void_p, c_void_p)), ("dactv_fn",CFUNCTYPE(c_void_p, c_void_p))]

class t_layer(Structure):
   _fields_ = [("lyrtype",c_int),("actvfntype",c_int),("n_out",c_int),("n_in",c_int),("ierror",c_float),("oerror",c_float),("input",c_float),("output",c_float),("error",c_float),("neuron",POINTER(t_neuron))]

class t_neural_nw(Structure):
   _fields_ = [("nntype",c_int),("nhdn",c_int),("ilyr",POINTER(t_layer)),("olyr",POINTER(t_layer)),("hlyr",POINTER(t_layer)),("eta", c_float),("train_fn",CFUNCTYPE(c_void_p, c_void_p)),("predict_fn",CFUNCTYPE(c_void_p, c_void_p))]
"""


class t_lyrinfo(Structure):
    _fields_ = [("size", c_int), ("lyrtype", c_int), ("actv", c_int)]


class t_nn_cfg(Structure):
    _fields_ = [
        ("type", c_int),
        ("eta", c_float),
        ("n_in", c_int),
        ("n_out", c_int),
        ("oactv", c_int),
        ("n_hdn", c_int),
        ("hinfo", POINTER(t_lyrinfo)),
    ]

    

class neural_network(object):
    def __init__(self):
        self.cfg = t_nn_cfg()
        self.stdtype = std_typedefs()

    def init_lib(self, path):
        self.lib = CDLL(path)
        self.create = self.lib.create_neural_network
        self.train = self.lib.train_network
        self.predict = self.lib.predict_network
        self.destroy = self.lib.destroy_neural_network
        self.create.restype = c_void_p
        self.train.restype = c_void_p
        self.predict.restype = c_void_p
        self.destroy.restype = c_void_p

    def update_config(self, nn_cfg_dict):
        self.cfg.type = self.stdtype.nn[nn_cfg_dict["neural_nw_type"]]
        self.cfg.eta = nn_cfg_dict["learning_rate"]
        self.cfg.n_in = nn_cfg_dict["num_input"]
        self.cfg.n_out = nn_cfg_dict["num_output"]
        self.cfg.n_hdn = nn_cfg_dict["num_hidden_layers"]
        self.cfg.hinfo = (t_lyrinfo * self.cfg.n_hdn)()
        hl_props = list(nn_cfg_dict["hl_prop"])
        for index in range(self.cfg.n_hdn):
            self.cfg.hinfo[index].actv = self.stdtype.actv[hl_props[index]["actv_fn"]]
            self.cfg.hinfo[index].size = hl_props[index]["size"]
            self.cfg.hinfo[index].lyrtype = self.stdtype.layer["HIDDEN"]


class std_typedefs(object):
    def __init__(self):
        self.nn = {"REGRESS": 0, "CLASSIFY": 1}
        self.layer = {"INPUT": 0, "HIDDEN": 1, "OUTPUT": 2}
        self.actv = {"LINEAR": 0, "RELU": 1, "SIGMOID": 2}
        self.norm = {"L1": 0, "L2": 1}

class t_sample(Structure):
    _fields_ = [("input",POINTER(c_float)), ("output",POINTER(c_float)), ("error", POINTER(c_float))]

class test(object):
    def __init__(self):
        self.nin  = 0
        self.nout = 0
        self.nset = 0
        self.data = []
        self.maxm = []
        self.minm = []
        self.mean = []
        self.stdv = []
    

class prometheus(object):
    def __init__(self):
        print("Creating Prometheus")
        self.nn = neural_network()
        self.test = test()

    def read_data_set(self, testcfg):
        csv_file = open(testcfg["test_file"]) 
        self.set = csv.reader(csv_file, delimiter=",")

    # Parse json cfg file and create nn cfg params
    def parse_configs(self):
        with open(self.cfg_file) as conf:
            config_dict = json.load(conf)
            self.nw_lib_path = config_dict["neural_nw_lib_path"]
            self.nn.update_config(config_dict["neural_nw_config"])
            testcfg = config_dict["neural_network_test"]
            self.read_data_set(testcfg)
       
    # fetch cmd line params
    def fetch_configs(self, argv):
        print("Fething configs for prometheus")
        try:
            opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
        except getopt.GetoptError:
            print("test.py -i <inputfile> -o <outputfile>")
            sys.exit(2)

        for opt, arg in opts:
            if opt == "-h":
                print("test.py -i <inputfile> -o <outputfile>")
                sys.exit()
            elif opt in ("-i", "--ifile"):
                self.cfg_file = arg
            elif opt in ("-o", "--ofile"):
                self.out_file = arg
        print("Input file is ", self.cfg_file)
        #print ('Output file is ', self.out_file)
        self.parse_configs()

    # Create  Neural Network Structure and Bind NN C lib
    def create_brain(self):
        print("* prometheus brain alive *")
        self.nn.init_lib(self.nw_lib_path)
        self.nn.obj = self.nn.create(self.nn.cfg)

    # Destroy Neural Network
    def destroy_brain(self):
        print("* prometheus brain dead *")
        self.nn.destroy(self.nn.obj)


    def get_data_set(self,nin,nout):
        data = (t_sample * 1)()
        data.input  = (c_float * nin)()
        data.output = (c_float * nout)()
        data.error  = (c_float * nout)()
        return data

#   def teach_brain(self):
#       print(type(self.test.data[0]))
#       for idx in range(100):
#          self.nn.train(self.nn.obj,self.test.data[idx])
#
#   def sentient_brain(self):
#       for idx in range(50):
#          self.nn.predict(self.nn.obj,self.test.data[idx])

    def fetch_test_train_data(self):
        # skip first row with feature names
        self.featurename =(next(self.set,None))
        # initalialze max, min, mean and std dev 
        self.test.maxm = [float("-inf") for i in range(self.nn.cfg.n_in)]
        self.test.minm = [float("inf") for i in range(self.nn.cfg.n_in)]
        self.test.mean = [0 for i in range(self.nn.cfg.n_in)]
        self.test.stdv = [0 for i in range(self.nn.cfg.n_in)]
        # read data from the csv file and
        for row in self.set:
            data = self.get_data_set(self.nn.cfg.n_in,self.nn.cfg.n_out)
            for idx in range(self.nn.cfg.n_in):
                data.input[idx] = float(row[idx])
                self.test.mean[idx] = self.test.mean[idx] + data.input[idx]
                self.test.maxm[idx] = max(self.test.maxm[idx],data.input[idx])
                self.test.minm[idx] = min(self.test.minm[idx],data.input[idx])
            temp = idx
            for idx in range(self.nn.cfg.n_out):
                data.output[idx] = float(row[idx+temp])
            self.test.nset += 1
            self.test.data.append(data)
        
        #shuffle the data
        random.shuffle(self.test.data) 
       
        # calc mean
        for idx in range(self.nn.cfg.n_in):
            self.test.mean[idx] = self.test.mean[idx]/self.test.nset
        
        # calc std deviation
        for elm in range(self.test.nset):
            for idx in range(self.nn.cfg.n_in):
                self.test.stdv[idx] = self.test.stdv[idx] + math.pow((self.test.data[elm].input[idx] - self.test.mean[idx]),2)
        for idx in range(self.nn.cfg.n_in):
            self.test.stdv[idx] = math.sqrt(self.test.stdv[idx]/self.test.nset)

#    start_time = time.time()
#    line_count = 0
#    for row in csv_reader:
#        if line_count == 0:
#            print(f'Column names are {", ".join(row)}')
#            line_count += 1
#        else:
#            print(f"{row[:]}\n")
#            line_count += 1
#
#
#           end_time = time.time()
#           print(end_time - start_time)


p1 = prometheus()
p1.fetch_configs(sys.argv[1:])
p1.create_brain()
p1.fetch_test_train_data()
#p1.teach_brain()
#p1.sentient_brain()
p1.destroy_brain()
