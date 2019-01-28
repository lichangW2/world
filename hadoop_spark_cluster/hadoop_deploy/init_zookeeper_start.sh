mkdir   /opt/zookeeper/data  
mkdir   /opt/zookeeper/dataLog

#running on each node, then run zkServer.sh status will show follower and leader
zkServer.sh start
jps

#必须从myid 小的节点开始启动(后启动master)，配置.zfg中当前机器的server.x=0.0.0.0:2888:3888
