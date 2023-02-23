from kafka import KafkaConsumer
from lightbulb import Bulb
import asyncio
import json


# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'34.122.190.200',
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
    
    wakeup = message.value['wakeup']
    if wakeup == '':
        emo = message.value['first_emotion']
        probs = message.value['emotion_softmax_logit']
        print(emo, probs)
        asyncio.run(bulb.change_color(probs))
    elif wakeup == 'sleep':
        asyncio.run(bulb.turn_off())
    elif wakeup == 'wake':
        asyncio.run(bulb.turn_on())
    elif wakeup == 'bright':
        asyncio.run(bulb.dim())
    elif wakeup == 'dark':
        asyncio.run(bulb.brighten())
