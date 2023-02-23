import multiprocessing
from music_player import Player
import ytDownload
import os
import json
from kafka import KafkaConsumer, KafkaProducer

EXTERNAL_IP = '34.122.190.200'
PORT = '9092'
TOPIC_NAME = 'senti_test'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = [EXTERNAL_IP+':'+PORT],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    # auto_offset_reset='earliest',
    # enable_auto_commit=False
    )

def DeleteAllFiles(filePath=None):
    if filePath is None:
        DeleteAllFiles("audio/anger")
        DeleteAllFiles("audio/fear")
        DeleteAllFiles("audio/joy")
        DeleteAllFiles("audio/love")
        DeleteAllFiles("audio/sadness")
        DeleteAllFiles("audio/surprise")
        return
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        print("Remove All Music File")
    else:
        print("Directory Not Found")
        
def kafka_consumer(player):
    turned_off = False
    senti_list = []
    for message in consumer:
        wakeup = message.value['wakeup']
        if wakeup == '':
            emo = message.value['first_emotion']
            senti_list.append(emo)
            with open('emotion_list.txt', 'a') as f:
                f.write(emo)
                f.write('\n')
        # elif wakeup == 'sleep' and not turned_off:
        #     player.turn_off()
        #     turned_off = True
        # elif wakeup == 'wake' and turned_off:
        #     player.play_song()
        #     turned_off = False
            

def main():
    DeleteAllFiles()

    p = Player()

    second = multiprocessing.Process(target=ytDownload.download, args=()) 
    second.start()
    first = multiprocessing.Process(target=p.play_song, args=())
    first.start()
    third = multiprocessing.Process(target=kafka_consumer, args=(p,))
    third.start()

    # print('all code playing')

    # emo = "joy"
    # p.sentiments.append(emo)
    # print(p.sentiments)

    second.join()
    first.join()
    third.join()


if __name__ == '__main__':
    main()