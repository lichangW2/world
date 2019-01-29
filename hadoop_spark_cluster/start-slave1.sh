na1="slave1"
docker rm -f $na1
docker  run -d -it --name $na1 -h "slave1"  -p 11261:11261 -v /Users/cj/workspace/world/hadoop_spark_cluster/codes_and_data:/codes_and_data bigdata_test_lichang_slave:201901282105 /bin/bash
