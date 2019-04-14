#!/usr/bin/python
# -*- coding: UTF-8 -*-

###用spark做密集型运算

import math
import random

from pyspark import SparkContext

from pyspark.sql import Row
from pyspark.sql import Column
from pyspark.sql import DataFrame
from pyspark.sql import functions
from pyspark import sql
from pyspark.sql import SparkSession

##测试dataframe

if __name__=="__main__":

    sc=SparkContext("local","Text Search")
    ss=SparkSession(sc)
    data = "file:///codes_data/test_3.txt"
    lineRdd=sc.textFile(data)

    df=lineRdd.map(lambda r: Row(r)).toDF(["line"])
    err=df.filter(functions.col("line").like("%ERROR%"))
    print(err.count())
    mysql_err=err.filter(functions.col("line").like("%MySQL%"))
    print("mysql error count:", mysql_err.count())
    print("mysql error collection:", mysql_err.collect())


