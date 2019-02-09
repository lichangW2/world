#每个节点上分别启动和停止
nohup /opt/kafka/kafka_2.11-2.1.0/bin/kafka-server-start.sh /opt/kafka/kafka_2.11-2.1.0/config/server.properties &
/opt/kafka/kafka_2.11-2.1.0/bin/kafka-server-stop.sh

##创建topic, 发送消息，接收消息
/opt/kafka/kafka_2.11-2.1.0/bin/kafka-topics.sh --create --zookeeper localhost:2181 --topic test --replication-factor 1 --partitions 1
/opt/kafka/kafka_2.11-2.1.0/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic test
/opt/kafka/kafka_2.11-2.1.0/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning
