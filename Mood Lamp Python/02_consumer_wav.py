from kafka import KafkaConsumer

from re import X
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

consumer = KafkaConsumer(
    'first_topic',
    bootstrap_servers = ['localhost:9092'],
    group_id = 'NLP-app'
    )

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092']
)

sent_analyzer = SentimentIntensityAnalyzer()

for message in consumer:
    incoming = message.value.decode('utf-8')
    print(incoming)

    label = sent_analyzer.polarity_scores(incoming)
    # keys can be neg, neu, pos, and compound
    max_key = max(label, key=label.get)

    print(max_key)

    if max_key == 'pos':
        outgoing = 'green'
    elif max_key == 'neg':
        outgoing = 'red'
    elif max_key == 'neu':
        outgoing = 'yellow'
    else: outgoing = 'blue'

    producer.send('sentiment_topic',
        key=b'commands', 
        value=bytes(outgoing, 'utf-8')
    )
    producer.flush()



