
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib:$HADOOP_COMMON_LIB_NATIVE_DIR"

hdfs namenode -format #初始化namenode, 只能运行一次，不然造成node不一致
start-dfs.sh
start-yarn.sh
jps
