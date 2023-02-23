#from moviepy.editor import * 
import pytube
import os.path
import random
import json
import re
import time

emotions = ['joy', 'sadness', 'surprise', 'love', 'anger', 'fear']
link_dic = {}

with open('result.json', 'r') as f:

    json_data = json.load(f)


for i in emotions:
    json_link = json_data[i]
    random.shuffle(json_link)
    link_dic[i] = json_link[:3]

# 다운로드 받을 파일 경로
path = r'/home/pi/playing/audio'
def download_mp3(link,sentiment,idx):
    global path
    yt = pytube.YouTube(link)
    file_name = str(idx)
    sentiment_path = os.path.join(path, sentiment)
    full_path = os.path.join(sentiment_path, file_name)
    # full_path = str(path+"/"+sentiment+"/"+str(file_name)) # 유튜브 영상을 소리만 있는 mp4파일로 다운로드
    yt.streams.filter(
        adaptive=True,
        file_extension='mp4',
        only_audio=True).order_by('abr').desc().first().download(sentiment_path, str(file_name)+'.mp4') # 파일 변환, mp4 -> mp3
    try: 
        os.rename(full_path + '.mp4', full_path + '.mp3') 
        print(f"Succeed downloading {str(file_name)}") 
    except: 
        os.remove(full_path + '.mp4')
        print("파일 이미 mp3파일이 있거나, 다른 오류가 발생함") # 비디오 링크 
        video_link = 'https://music.youtube.com/watch?v=TMT9MNM-NHg&feature=share'

def download():
    for k in range(3):
        start_time = time.time()
        for i in link_dic.keys():
            download_mp3(link_dic[i][k],i,k)
        end_time = time.time()
        print(f'elapsed time: {end_time - start_time:.2f}s')

# video_link = input("입력하세요: ")
# download_mp3(video_link)