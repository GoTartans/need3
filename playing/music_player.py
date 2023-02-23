from collections import defaultdict
import asyncio
import os
import vlc
import glob
import time
import json
from kafka import KafkaConsumer, KafkaProducer


EXTERNAL_IP = '34.122.190.200'
PORT = '9092'
TOPIC_NAME = 'senti_test'

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers = [EXTERNAL_IP+':'+PORT],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=False
    )

base_folder = '/home/pi/playing/audio'

class Player:
    def __init__(self):
        self.sentiments = []
        self.play_length =25
        self.play_count = defaultdict(int)
        self.media_player = None
        self.stopped_from_outside = False

    def send_sentiment(self, sentiment):
        self.sentiments.append(sentiment)

    def DeleteAllFiles(self, filePath=None):
        if filePath is None:
            self.DeleteAllFiles("audio/anger")
            self.DeleteAllFiles("audio/fear")
            self.DeleteAllFiles("audio/joy")
            self.DeleteAllFiles("audio/love")
            self.DeleteAllFiles("audio/sadness")
            self.DeleteAllFiles("audio/surprise")
            return
        if os.path.exists(filePath):
            for file in os.scandir(filePath):
                os.remove(file.path)
            print("Remove All Music File")
        else:
            print("Directory Not Found")

    def calculate_total_sentiment(self):
        if not self.sentiments:
            return 'love'
        scores = defaultdict(float)
        gamma = 0.95
        for i, sentiment in enumerate(self.sentiments[::-1]):
            scores[sentiment] += gamma ** i

        scores = list(scores.items())
        scores.sort(key=lambda x:x[1], reverse=True)
        print(scores)
        # remove sentiment history except the last five
        if len(self.sentiments) > 3:
            self.sentiments = self.sentiments[-3:]
        return scores[0][0]

    def turn_off(self):
        if self.media_player is not None:
            print('turn off?')
            self.stopped_from_outside = True
            self.media_player.stop()

    def play_song(self, emotion='neutral'):
        start_time = time.time()
        print(start_time)

        base_folder = '/home/pi/playing/audio'
        # vlc State 0: Nowt, 1 Opening, 2 Buffering, 3 Playing, 4 Paused, 5 Stopped, 6 Ended, 7 Error
        playing = set([1,2,3,4])

        def add_media(inst, media_list, playlist):
            for song in playlist:
                print('Loading: - {0}'.format(song))
                media = inst.media_new(song)    
                media_list.add_media(media)
        
        playlist = glob.glob(base_folder + "/" + "*.mp3")
        playlist = sorted(playlist)
        
        media_player = vlc.MediaListPlayer()
        inst = vlc.Instance('--no-xlib --quiet ')
        media_list = vlc.MediaList()
        add_media(inst, media_list, playlist)

        media_player.set_media_list(media_list)
        media_player.play()
        self.media_player = media_player
        time.sleep(0.1)

        current = ""
        idx = 1
        player = media_player.get_media_player()

        init_volume = player.audio_get_volume() 
        fadeout_threshold = self.play_length - 5

        try:
            while True:
            # for message in consumer:
                
                current_time = time.time()
                player = media_player.get_media_player()
                state = player.get_state()
                

                if current_time - start_time > fadeout_threshold: # 꺼져야 할 때
                    player.audio_set_volume(int(player.audio_get_volume() * 0.8))
                    fadeout_threshold += 0.5
                    
                if state.value not in playing and self.stopped_from_outside:
                    return

                if state.value not in playing or current_time - start_time > self.play_length: # 꺼졌을 때 
                    
                    start_time = time.time()
                    media_player.stop() # TODO: usage right?

                    with open('emotion_list.txt', 'r') as f:
                        self.sentiments = [line.strip() for line in f.readlines()]
                    os.remove('emotion_list.txt')
                    
                    print(self.sentiments)
                    playlist = glob.glob(base_folder + "/" + self.calculate_total_sentiment() + "/" + "*.mp3")
                    self.sentiments = [] # sentiment 초기화

                    idx = min(self.play_count[emotion], len(playlist)-1)
                    
                    playlist = [sorted(playlist)[idx]] # play only one song
                    self.play_count[emotion] += 1
                    
                    media_player = vlc.MediaListPlayer()
                    inst = vlc.Instance('--no-xlib --quiet ')
                    media_list = vlc.MediaList()
                    add_media(inst, media_list, playlist)

                    media_player.set_media_list(media_list)
                    media_player.play()
                    self.media_player = media_player
                    player = media_player.get_media_player()
                    player.audio_set_volume(init_volume) 
                    time.sleep(0.1)
                title = player.get_media().get_mrl()
                
                if title != current: # 새로운 노래 틀어줄 때
                    print("\nPlaying - {0}\t{1}".format(str(title), idx))
                    current = title
                    idx += 1
                    

                time.sleep(1.0)
        except KeyboardInterrupt:
            self.DeleteAllFiles()

        print("\nPlaylist Finished")

if __name__ == '__main__':
    p = Player()
    p.play_song()
    start_time = time.time()

    # for message in consumer:
    #     emo = message.value['first_emotion']
    #     print(emo)
    #     p.sentiments.append(emo)
    #     print(p.sentiments)
    #     res = p.calculate_total_sentiment()
    #     if time.time() - start_time < 10:
    #         p.play_song(emotion=res)

    
    # print(res)