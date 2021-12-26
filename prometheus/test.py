#!/usr/bin/python3
## @file test.py
## Driver code for testing the prometheus module

import sys, getopt
from prometheus import prometheus


p1 = prometheus()
p1.fetch_configs(sys.argv[1:])
p1.create_brain()
p1.teach_brain()
p1.sentient_brain()
p1.destroy_brain()
