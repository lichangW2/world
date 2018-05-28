from numpy as np
import os
import shutil

import caffe2.python.predictor_exporter as pe
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew

core.GlobalInit(["caffe2", "--caffe2_log_level=1"])
print("Necessities imported!")


def AddInput(model, batch_size, db, db_type):
    data_uint8, label = model.TensorProtosDBInput()
