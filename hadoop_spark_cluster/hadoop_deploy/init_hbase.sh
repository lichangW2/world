#not use the hbase integrated zookeeper, use the one we installed
# the export clauses must be put in hbase-env.sh
export HBASE_MANAGES_ZK=false
export HBASE_PID_DIR=/root/hbase/pids
export HBASE_LOG_DIR=/opt/hbase/logs

