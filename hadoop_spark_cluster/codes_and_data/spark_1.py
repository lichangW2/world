import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf


#os.environ['SPARK_HOME'] = "/opt/spark/spark-2.3.0-bin-hadoop2.7"
#os.environ['PYSPARK_SUBMIT_ARGS'] = "--master yarn pyspark-shell"
#sys.path.append(os.path.join(os.environ['SPARK_HOME'], "python"))
#sys.path.append("/opt/spark/spark-2.3.0-bin-hadoop2.7/python/lib/py4j-0.10.6-src.zip")

def wordCount():
    conf=SparkConf().setMaster("local") \
             .setAppName("wordcount") \
             .set("spark.executor.memoryOverhead","512m") \
             .set("spark.driver.memoryOverhead","512m") \
             .set("spark.driver.memory","512m") \
             .set("spark.executor.memory","512")
    #spark://master:7077
    sc=SparkContext(conf=conf)
    dir="hdfs://master:11230/spark/test/"
    #dir="file:///workspace/"
    path=dir+"hello.txt"

    textRDD=sc.textFile(path)
    flatRDD=textRDD.flatMap(lambda line:line.split(" "))
    wdCt=flatRDD.map(lambda word:(word,1)).reduceByKey(lambda a,b:a+b)
    print("here")
    print(wdCt.collect())

if __name__=='__main__':
    wordCount()