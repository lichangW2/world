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

## 测试dataframe

if __name__=="__main__":

    url="jdbc:mysql://localhost:3306/sparktestdb?user=root&password=wlc123"
    ss=SparkSession.builder \
        .master("local") \
        .appName("build dataframe from mysql") \
        .getOrCreate()

    df=ss \
    .read \
    .format("jdbc")\
    .option("url",url)\
    .option("dbtable", "people") \
    .load()
    #.option("driver", 'com.mysql.jdbc.Driver')\


    print("database schema",df.printSchema())
    countsByAge = df.groupBy("age").count()
    countsByAge.show()

    countsByAge.write.format("json").save("file:///codes_data/test_result_4.json")




