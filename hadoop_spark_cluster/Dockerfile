FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN add-apt-repository ppa:webupd8team/java && \
    apt-get update -y --allow-unauthenticated && \
    apt-get -y --allow-unauthenticated install python-setuptools \
    python-pip libpython2.7-dev python-software-properties openssh-server git \

# environments settings
ENV SOURCE /cluster/source

## JDK
RUN  apt-get update && mkdir -p ${SOURCE} && cd -O ${SOURCE} &&
     wget https://download.oracle.com/otn-pub/java/jdk/8u191-b12/2787e4a523244c269598db4e85c51e0c/jdk-8u191-linux-x64.tar.gz && \
     tar -zxvf jdk-8u191-linux-x64.tar.gz -C /opt/java && \
## Zookeeper
    wget http://mirror.bit.edu.cn/apache/zookeeper/zookeeper-3.4.10/zookeeper-3.4.10.tar.gz && \
    tar -zxvf zookeeper-3.4.10.tar.gz -C /opt/zookeeper 
## Hadoop
     wget  http://mirror.bit.edu.cn/apache/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz && \
     tar -zxvf hadoop-2.7.7.tar.gz -C /opt/hadoop && \
## Hive
     wget http://mirror.bit.edu.cn/apache/hive/hive-2.3.4/apache-hive-2.3.4-bin.tar.gz && \
     tar  -zxvf apache-hive-2.3.4-bin.tar.gz -C /opt/hive && \
## Spark compile depending on scala, maven
    wget http://apache.communilink.net/maven/maven-3/3.6.0/binaries/apache-maven-3.6.0-bin.tar.gz && \
     tar -zxvf apache-maven-3.6.0-bin.tar.gz -C /opt/maven && \
     wget https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.tgz && \
     tar -zxvf scala-2.11.12.tgz -C /opt/scala && \
## HBase 
    wget http://mirror.bit.edu.cn/apache/hbase/1.3.3/hbase-1.3.3-bin.tar.gz && \
    tar -zxvf hbase-1.3.3-bin.tar.gz -C /opt/hbase && \
## Kafka
    wget https://www.apache.org/dyn/closer.cgi?path=/kafka/2.1.0/kafka_2.11-2.1.0.tgz && \
    tar -zxvf kafka_2.11-2.1.0.tgz -C /opt/kafka

RUN  wget https://github.com/apache/spark/archive/branch-2.0.zip && \
     unzip branch-2.0.zip && \
     ./dev/make-distribution.sh --name "hadoop2-without-hive" --tgz "-Pyarn,hadoop-provided,hadoop-2.7,parquet-provided" && \
     tar -zxvf spark-2.1.1-bin-hadoop2-without-hive.tgz -C /opt/spark

## Hive
RUN wget 
## HBase

## Kafka



ENV JAVA_HOME=/opt/java/jdk1.8.0_191
ENV JRE_HOME=/opt/java/jdk1.8.0_191/jre
ENV CLASSPATH=.:${JAVA_HOME}/lib/dt.jar:${JAVA_HOME}/lib/tools.jar:${JRE_HOME}/lib

ENV HADOOP_HOME=/opt/hadoop/hadoop2.8 
ENV HADOOP_COMMON_LIB_NATIVE_DIR=${HADOOP_HOME}/lib/native
ENV HADOOP_OPTS=-Djava.library.path=${HADOOP_HOME}/lib
ENV HBASE_HOME=/opt/hbase/hbase-1.3.3

ENV ZK_HOME=/opt/zookeeper/zookeeper-3.4.10

ENV HIVE_HOME=/opt/hive/apache-hive-2.3.4
ENV HIVE_CONF_DIR=${HIVE_HOME}/conf

ENV SCALA_HOME=/opt/scala/scala-2.12.8
ENV MAVEN_HOME=/opt/maven/apache-maven-3.6.0
ENV SPARK_HOME=/opt/spark/spark-2.4.0-bin


ENV PATH=.:${JAVA_HOME}/bin:${SCALA_HOME}/bin:${SPARK_HOME}/bin:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${ZK_HOME}/bin:${HBASE_HOME}/bin:${HIVE_HOME}/bin:$PATH


# environments settings
WORKDIR /workspace
CMD sh 