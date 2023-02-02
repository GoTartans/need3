# Requirements
### Install java
    sudo apt update
    sudo apt install default-jdk
    java -version

### Install kafka
    wget https://dlcdn.apache.org/kafka/3.3.1/kafka_2.13-3.3.1.tgz
    tar -xzf kafka_2.13-3.3.1.tgz

### Install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
    sha256sum Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
    bash Miniconda3-py38_22.11.1-1-Linux-x86_64.sh




# Kafka Commands
### Start the ZooKeeper service
    bin/zookeeper-server-start.sh config/zookeeper.properties

### Start the Kafka broker service
    bin/kafka-server-start.sh config/server.properties

### Create Topic
    bin/kafka-topics.sh --create --topic test --bootstrap-server localhost:9092

### List Topic
    bin/kafka-topics.sh --list --bootstrap-server localhost:9092

### Init Producer
    bin/kafka-console-producer.sh --topic test --bootstrap-server localhost:9092

### Init Consumer
    bin/kafka-console-consumer.sh --topic test --bootstrap-server localhost:9092 --from-beginning

### Remove logs
    rm -rf /tmp/kafka-logs /tmp/zookeeper /tmp/kraft-combined-logs




# Connection Setting
### Socket Server Settings (/config/server.properties)
    listeners=PLAINTEXT://0.0.0.0:9092
    advertised.listeners=PLAINTEXT://34.132.166.200:9092

### Firewall Rule for a GCP instance
    Network interfaces -> Network(default) -> FIREWALLS -> ADD FIREWALL RULE -> Source IPv4 ranges: 0.0.0.0/0 (allow all the IPs)
