import vlc
import glob
import time

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
playlist[0], playlist[1] = playlist[1], playlist[0]
#playlist = ['./2005.mp3', './vp1.mp3']
media_player = vlc.MediaListPlayer()
inst = vlc.Instance('--no-xlib --quiet ') 
media_list = vlc.MediaList()
add_media(inst, media_list, playlist)

media_player.set_media_list(media_list)
media_player.play()
time.sleep(0.1)
current = ""
idx = 1
player = media_player.get_media_player()
while True:
    state = player.get_state()
    if state.value not in playing:
        break
    title = player.get_media().get_mrl()
    if title != current:
        print("\nPlaying - {0}\t{1} of {2}".format(str(title), idx, len(playlist)))
        current = title
        idx += 1
    time.sleep(1.0)
print("\nPlaylist Finished")