#!/usr/bin/python3

import sys, getopt
import json
from ctypes import *
import os
import pandas
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


class prometheus(object):
    def __init__(self):
        print("Creating Prometheus")
        self.nn = neural_network()

    # Parse json cfg file and create nn cfg params
    def parse_configs(self):
        with open(self.cfg_file) as conf:
            config_dict = json.load(conf)
            self.nw_lib_path = config_dict["neural_nw_lib_path"]
            self.nn.update_config(config_dict["neural_nw_config"])
            testcfg = config_dict["neural_network_test"]
            print(testcfg["test_file"])
            start_time = time.time()
            df = pandas.read_csv(testcfg["test_file"])
            print(df)
            end_time = time.time()
            print(end_time - start_time)

            start_time = time.time()
            with open(testcfg["test_file"]) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print(f'Column names are {", ".join(row)}')
                        line_count += 1
                    else:
                        print(f"{row[:]}\n")
                        line_count += 1
            end_time = time.time()
            print(end_time - start_time)

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
        # 	   print ('Output file is ', self.out_file)
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


p1 = prometheus()
p1.fetch_configs(sys.argv[1:])
p1.create_brain()
p1.destroy_brain()
