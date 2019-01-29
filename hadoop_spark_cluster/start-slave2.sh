na2="slave2"
docker rm -f $na2
docker  run -d -it --name $na2 -h "slave2" -v /Users/cj/workspace/world/hadoop_spark_cluster/codes_and_data:/codes_and_data bigdata_test_lichang_slave:201901282105 /bin/bash
