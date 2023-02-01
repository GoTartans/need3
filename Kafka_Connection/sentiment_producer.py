from kafka import KafkaConsumer
from kafka import KafkaProducer

EXTERNAL_IP = '35.236.200.251'
PORT = '9092'
producer = KafkaProducer(
    bootstrap_servers=[EXTERNAL_IP+':'+PORT]
)

def sentiment_sending(sentiment, logit):
    producer.send('sentiment_test',
            key=b'sentiment', 
            value=bytes(sentiment, 'utf-8')
        )
    producer.send('sentiment_test',
            key=b'logit', 
            value=bytes(logit, 'utf-8')
        )
    producer.flush()
    
    
TOPIC_NAME = 'sentiment_test'
run_app = True
i = 0

while run_app:
    print('Line - ' + str(i))
    sentiment = input()
    logit = input()
    sentiment_sending(sentiment, logit)
    i = i + 1