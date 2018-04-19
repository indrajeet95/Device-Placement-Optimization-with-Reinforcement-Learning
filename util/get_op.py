
import argparse
import os.path
import re
import sys
import json
import numpy as np

from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

def get_op():
    gdef = gpb.GraphDef()
    with open("mnist.txt", 'r') as fh:
        graph_str = fh.read()

    pbtf.Parse(graph_str, gdef)

    f = open("mnist_get_op.txt",'w+')
    for node in gdef.node:
        f.write("%s " %(node.op))
    f.close()

get_op()
