#!/usr/bin/python
# -*- coding: UTF-8 -*-

###用spark做密集型运算

import math
import random

from pyspark.sql import Row
from pyspark.sql import Column
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions
from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression

## 测试ML库中的逻辑回归

if __name__=="__main__":

    ss=SparkSession.builder \
        .master("local") \
        .appName("logistic regression") \
        .getOrCreate()

    rdf=ss.createDataFrame([
    ( 1.0,Vectors.dense([144.5])),
    ( 1.0,Vectors.dense([167.2])),
    ( 0.0,Vectors.dense([124.1])),
    ( 1.0,Vectors.dense([144.5])),
    ( 0.0,Vectors.dense([133.2])),
    ( 0.0,Vectors.dense([124.1])),
    ( 1.0,Vectors.dense([129.2])),
    (0.0, Vectors.dense([124.1])),
    (1.0, Vectors.dense([144.5])),
    (0.0, Vectors.dense([133.2])),
    (0.0, Vectors.dense([124.1])),
    (1.0, Vectors.dense([129.2])),
   ],["label", "features"])

    lr=LogisticRegression(maxIter=10);
    model=lr.fit(rdf)

    ## test df, model只会用,features列
    tdf = ss.createDataFrame([
        (0.0, Vectors.dense([166.5])),
        (0.0, Vectors.dense([177.2])),
        (0.0, Vectors.dense([124.1]))
    ],["label", "features"])

    model.transform(tdf).show()




