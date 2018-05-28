#!/usr/bin/python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import logging
import gzip
import cPickle
import lmdb
import traceback
import numpy as np
import argparse
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper, helpers

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

mnist_data = "/workspace/data/MNIST/mnist.pkl.gz"
mnist_lmdb_train = "/workspace/data/MNIST/train.lmdb"
mnist_lmdb_test = "/workspace/data/MNIST/test.lmdb"


def load_data(filename=None):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return (training_data, validation_data, test_data)


def make_lmdb(data, labels, output_file, data_shape=None):
    print(">>> write database...")

    LMDB_MAP_SIZE = 3 * (1 << 30)
    env = lmdb.open(output_file, LMDB_MAP_SIZE)
    dt_shape = None

    if type(data) is np.array:
        dt_shape = data.shape
    elif data_shape is None:
        logging.error("data_shape is required,exit!")
        return
    else:
        dt_shape = data_shape
    with env.begin(write=True) as txn:

        if len(data) != len(labels):
            logging.error("data and labels do not match")
            raise Exception("mismatched data and labels")

        try:
            for index in xrange(len(data)):
                image_protos = caffe2_pb2.TensorProtos()

                image_proto = image_protos.protos.add()
                image_proto.data_type = caffe2_pb2.TensorProto.FLOAT
                image_proto.dims.extend(dt_shape)
                dt = None
                if type(data) is np.array:
                    dt = data[index].reshape(np.prod(dt_shape))
                else:
                    dt = data[index]
                image_proto.float_data.extend(dt)

                label_proto = image_protos.protos.add()
                label_proto.data_type = caffe2_pb2.TensorProto.INT32
                label_proto.int32_data.append(labels[index])

                txn.put('{}'.format(index).encode("ascii"), image_protos.SerializeToString())
                if index % 16 == 0:
                    logging.info('Inserted {} rows'.format(index))
        except Exception as _e:
            logging.error("excetion traceback: %s", traceback.format_exc())
            return


def read_lmdb_with_caff2(db_file):
    print(">>>  read database...")
    model = model_helper.ModelHelper(name="lmdbtest")
    model.TensorProtosDBInput([], ["x", "y"], batch_size=32, db=db_file, db_type="lmdb")
    print(">>>> net proto:")
    print(model.net.Proto())
    print(">>> init_net proto:")
    print(model.param_init_net.Proto())

    workspace.ResetWorkspace()
    workspace.RunNetOnce(model.param_init_net)
    print("Blobs in the workspace: {}".format(workspace.Blobs()))
    workspace.CreateNet(model.net)
    workspace.RunNet(model.net.Proto().name)

    print("the first batch of data and label is:")
    data = workspace.FetchBlob("x")
    print("blob data: type: {}, shape: {}, sample data[1,10,10]:{}".format(type(data), data.shape,
                                                                           data[0][0][:5, :5]))
    # print(workspace.FetchBlob("x"))
    print("blob label is:")
    print(workspace.FetchBlob("y"))

    workspace.RunNet(model.net.Proto().name)
    print("label 2 {}".format(workspace.FetchBlob("y")))
    #    workspace.RunNet(model.net.Proto().name)
    #    print("The second batch of feature is:")
    #    print(workspace.FetchBlob("x"))
    #    print("The second batch of label is:")
    #    print(workspace.FetchBlob("y"))

if __name__ == '__main__':
    train_data, valid_data, test_data = load_data(mnist_data)
    make_lmdb(train_data[0],train_data[1],mnist_lmdb_train,data_shape=(1,28,28))
    #make_lmdb(test_data[0],test_data[1],mnist_lmdb_test,data_shape=(1,28,28))
    #read_lmdb_with_caff2(mnist_lmdb_train)