from kafka import KafkaProducer
import json

# info of the instance where kafka cluster is located
EXTERNAL_IPs = {
    'team3-gpu':'35.184.175.239',
    'instance-team3':'34.145.236.124',
    'Chanwoo':'34.132.166.200',
}
EXTERNAL_IP = 'localhost'
PORT = '9092'

producer = KafkaProducer(
    bootstrap_servers=[EXTERNAL_IP+':'+PORT],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC_NAME = 'test'
producer.send(TOPIC_NAME,
            key= b'hello',
            value={'keeeey':'valueeeee','keeeeeey2':'valueeeee2'}
        )
producer.flush()