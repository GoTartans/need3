from kafka import KafkaConsumer

EXTERNAL_IP = '34.132.166.200'
PORT = '9092'

consumer = KafkaConsumer(
    'sentiment_test',
    bootstrap_servers = [EXTERNAL_IP+':'+PORT]
    )


for message in consumer:
    message = message.value.decode('utf-8')
    print(message)
