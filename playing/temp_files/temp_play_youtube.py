import vlc
import pafy
import urllib.request
import keyboard

url = "https://open.spotify.com/track/64V565ZQvU8LUlBd8yFJpt"
video = pafy.new(url)
best = video.getbest()
playurl = best.url
ins = vlc.Instance()
player = ins.media_player_new()

code = urllib.request.urlopen(url).getcode()
if str(code).startswith('2') or str(code).startswith('3'):
    print('Stream is working')
else:
    print('Stream is dead')

Media = ins.media_new(playurl)
Media.get_mrl()
player.set_media(Media)
print('Song played')
player.play()
print('Song stopped')

good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
while str(player.get_state()) in good_states:
    if keyboard.read_key() == 's':
        player.stop()
        break
    print('Stream is working. Current state = {}\n'.format(player.get_state()))

print('Stream is not working. Current state = {}\n'.format(player.get_state()))
player.stop()