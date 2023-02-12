from kafka import KafkaConsumer
import json

# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'35.184.175.239',
    'instance-team3':'34.145.236.124',
    'Chanwoo':'34.132.166.200',
}
EXTERNAL_IP = 'localhost'
PORT = '9092'

TOPIC_NAME = 'senti_test'
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = [EXTERNAL_IP+':'+PORT],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

for msg in consumer:
    print(msg.value)