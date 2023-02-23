from ytmusicapi import YTMusic
import json
from tqdm import tqdm

# YTMusic.setup(filepath="headers_auth.json")
line = '-' * 50

yt = YTMusic()

result = {}

emotions = ['joy', 'sadness', 'surprise', 'love', 'anger', 'fear']
    
for emotion in tqdm(emotions):
    print(line)
    print(f'Processing {emotion}')
    result[emotion] = []
    browse_ids = []

    search_results = yt.search(f'instrumental {emotion}')
    for elem in search_results:
        if elem['category'] == "Community playlists":
            browse_ids.append(elem['browseId'])

    # trim down playlist to 3
    if len(browse_ids) > 3:
        browse_ids = browse_ids[:3]

    for browse_id in browse_ids:
        pl = yt.get_playlist(browse_id)

        tracks = pl['tracks']

        for i, item in enumerate(tracks):
            # trim down number of songs per playlist to 3
            if i == 3:
                break

            vid_id = item['videoId']
            if vid_id is None:
                continue
            # print(item)
            song = yt.get_song(vid_id)
            result[emotion].append(song['microformat']['microformatDataRenderer']['urlCanonical'])
            

with open('result.json', 'w') as f:
    json.dump(result, f, indent = 4)
            

# search_results = yt.search('instrumental joy')
# for elem in search_results:
#     if elem['category'] == "Community playlists":
#         print(elem)


# cat = yt.get_mood_categories()
# mood_moment = cat['Moods & moments'] # list of dictionaries
# print('List of titles in Moods & moments')
# for dct in mood_moment:
#     print(dct['title'])

# print(line)

# genres = cat['Genres'] # list of dictionaries
# print('List of titles in Genres')
# for dct in genres:
#     print(dct['title'])

# print(line)

# for i, item in enumerate(cat):
#     for dct in cat[item]:
#         try:
#             print(f'Title: {dct["title"]}')
#             print(f'Params: {dct["params"]}')
#             param = dct['params']
#             mood_playlists = yt.get_mood_playlists(param)
#             playlist = mood_playlists[0]
#             # print(playlist)
#             playlistId = playlist['playlistId']
#             # print(playlistId)
#             playlist = yt.get_playlist(playlistId)
#             # print(playlist)
#             print(playlist.keys())
#             tracks = playlist['tracks']
#             for i, track in enumerate(tracks):
#                 print(track['videoId'])
#                 song = yt.get_song(track['videoId'])
#                 print(song['microformat']['microformatDataRenderer']['urlCanonical'])
#                 if i == 10:
#                     break
#                 # break
#             break
#         except KeyError:
#             print(f'KeyError occured at {dct["title"]}')
#     # if i==0:
#     #     break