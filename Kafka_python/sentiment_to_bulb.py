from kafka import KafkaConsumer

EXTERNAL_IP = '35.236.200.251'
PORT = '9092'

consumer = KafkaConsumer(
    'sentiment_test',
    bootstrap_servers = [EXTERNAL_IP+':'+PORT],
    group_id = 'bulb-app'
    )


for message in consumer:
    message = message.value.decode('utf-8')
    print(message)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(bulbAction(message))