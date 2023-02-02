from kafka import KafkaConsumer
from lightbulb import Bulb
import asyncio
import json


# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'35.184.175.239',
    'instance-team3':'34.145.236.124',
    'Chanwoo':'34.132.166.200',
}
EXTERNAL_IP = EXTERNAL_IPs['team3-gpu']
PORT = '9092'
TOPIC_NAME = 'senti_test'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = [EXTERNAL_IP+':'+PORT],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

file_path = '/root/need3/Kafka_Connection/sentiment.json'
bulb = Bulb()
for message in consumer:
    with open(file_path, mode='a') as f:
        json.dump(message.value, f)
    probs = message.value['logits']
    print(probs)
    print(type(probs))
    asyncio.run(bulb.change_color(probs))
