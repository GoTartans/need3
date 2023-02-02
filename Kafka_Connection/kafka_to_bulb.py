from kafka import KafkaConsumer
from lightbulb import Bulb
import asyncio

# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'35.184.175.239',
    'instance-team3':'34.145.236.124',
    'Chanwoo':'34.132.166.200',
}
EXTERNAL_IP = EXTERNAL_IPs['team3-gpu']
PORT = '9092'
TOPIC_NAME = 'sentiment_test'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = [EXTERNAL_IP+':'+PORT]
    )

bulb = Bulb()
for message in consumer:
    message = message.value.decode('utf-8')
    print(message)
    asyncio.run(bulb.change_color(probs))
