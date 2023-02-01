from kafka import KafkaProducer

EXTERNAL_IP = '34.132.166.200'
PORT = '9092'

producer = KafkaProducer(
    bootstrap_servers=[EXTERNAL_IP+':'+PORT]
)

run_app = True
i = 1

TOPIC_NAME = 'wav_test'
while run_app:
    print('Line - ' + str(i))
    output = input('wav example:')
    i = i + 1

    producer.send(TOPIC_NAME,
        key=b'voice', 
        value=bytes(output, 'utf-8')
    )

    if output == 'stop Voice': 
        run_app = False
    