#!/usr/bin/python
# -*- coding: UTF-8 -*-

###用spark做密集型运算

import math
import random
from pyspark import SparkContext

##一般函数代替lambda函数，接收一个参数
def inside(p):
    x,y= random.random(),random.random()
    return math.sqrt((x-0.5)**2+(y-0.5)**2) <0.5

if __name__=="__main__":
    RNUM=10000
    sc=SparkContext("local","Pi Estimation")
    count=sc.parallelize(xrange(0,RNUM)).filter(inside).count()
    print "Pi is roughly %f" % (4.0 * count / RNUM)