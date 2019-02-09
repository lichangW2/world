FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN apt-get update -y --allow-unauthenticated && \
    apt-get -y --allow-unauthenticated install software-properties-common python-software-properties \
    python-setuptools python-pip libpython2.7-dev python-software-properties openssh-server git wget && \
    add-apt-repository ppa:webupd8team/java 

RUN pip install -q --trusted-host http://mirrors.aliyun.com/pypi/simple/  kafka-python

# environments settings
ENV SOURCE /cluster/source

## JDK update the url each time to build 
RUN  apt-get update && mkdir -p ${SOURCE} && cd ${SOURCE} && \
     wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" \
     https://download.oracle.com/otn-pub/java/jdk/8u191-b12/2787e4a523244c269598db4e85c51e0c/jdk-8u191-linux-x64.tar.gz && \
     mkdir -p /opt/java && tar -zxvf jdk-8u191-linux-x64.tar.gz -C /opt/java 
## Zookeeper
RUN wget http://mirror.bit.edu.cn/apache/zookeeper/zookeeper-3.4.10/zookeeper-3.4.10.tar.gz && \
    mkdir -p /opt/zookeeper && tar -zxvf zookeeper-3.4.10.tar.gz -C /opt/zookeeper 
## Hadoop
RUN  wget  http://mirror.bit.edu.cn/apache/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz && \
     mkdir -p /opt/hadoop && tar -zxvf hadoop-2.7.7.tar.gz -C /opt/hadoop 
## Spark compile depending on scala, maven
RUN wget http://apache.communilink.net/maven/maven-3/3.6.0/binaries/apache-maven-3.6.0-bin.tar.gz && \
    mkdir -p /opt/maven && tar -zxvf apache-maven-3.6.0-bin.tar.gz -C /opt/maven && \
    wget https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.tgz && \
    mkdir -p /opt/scala && tar -zxvf scala-2.11.12.tgz -C /opt/scala && \
    wget https://archive.apache.org/dist/spark/spark-2.3.0/spark-2.3.0-bin-hadoop2.7.tgz && \
    mkdir -p /opt/spark && tar -zxvf spark-2.3.0-bin-hadoop2.7.tgz -C /opt/spark
## HBase 
RUN wget http://mirror.bit.edu.cn/apache/hbase/1.3.3/hbase-1.3.3-bin.tar.gz && \
    mkdir -p /opt/hbase && tar -zxvf hbase-1.3.3-bin.tar.gz -C /opt/hbase 
## Kafka
RUN wget http://mirrors.hust.edu.cn/apache/kafka/2.1.0/kafka_2.11-2.1.0.tgz && \
    mkdir -p /opt/kafka && tar -zxvf kafka_2.11-2.1.0.tgz -C /opt/kafka 

ENV JAVA_HOME=/opt/java/jdk1.8.0_191
ENV JRE_HOME=/opt/java/jdk1.8.0_191/jre
ENV CLASSPATH=.:${JAVA_HOME}/lib/dt.jar:${JAVA_HOME}/lib/tools.jar:${JRE_HOME}/lib

ENV HADOOP_HOME=/opt/hadoop/hadoop-2.7.7 
ENV HADOOP_COMMON_LIB_NATIVE_DIR=${HADOOP_HOME}/lib/native
ENV HADOOP_OPTS=-Djava.library.path=${HADOOP_HOME}/lib
ENV HBASE_HOME=/opt/hbase/hbase-1.3.3

ENV ZK_HOME=/opt/zookeeper/zookeeper-3.4.10
ENV SCALA_HOME=/opt/scala/scala-2.11.12
ENV MAVEN_HOME=/opt/maven/apache-maven-3.6.0
ENV SPARK_HOME=/opt/spark/spark-2.3.0-bin-hadoop2.7
ENV KAFKA_HOME=/opt/kafka/kafka_2.11-2.1.0

ENV PATH=.:${JAVA_HOME}/bin:${SCALA_HOME}/bin:${SPARK_HOME}/bin:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${ZK_HOME}/bin:${HBASE_HOME}/bin:${HIVE_HOME}/bin:$PATH

# environments settings
WORKDIR /workspace
CMD /bin/bash 
