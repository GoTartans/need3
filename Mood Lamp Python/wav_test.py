from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'wav_test',
    bootstrap_servers = ['localhost:9092'],
    group_id = 'test',
    # auto_offset_reset='earliest',
    # enable_auto_commit=False
    )


for message in consumer:
    incoming = message.value
    with open('/home/sunri/myfile.wav', mode='bw') as f:
        f.write(incoming)
    print(incoming)
    print(type(incoming))




