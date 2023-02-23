import requests
from bs4 import BeautifulSoup

webpage = requests.get("https://music.youtube.com/playlist?list=PLqk7PmhxKRBd4Qxslc9P07J_t_9Ty9tZQ")
soup = BeautifulSoup(webpage.content, "html.parser")

b = soup.find_all()
print(b)
#a = soup.find_all(attrs={'class':'style-scope ytmusic-section-list-renderer', "id":})
a = soup.find_all("ytmusic-responsive-list-item-renderer", {"class": "style-scope ytmusic-playlist-shelf-renderer"})
print(a)

"""
for i in a:
    b = i.find_all("a")[0]
    c = str(b)
    initial_name = c.split("/people/")[1].split(">")[0][:-1]
    final_name = initial_name.replace("-"," ")
    print(final_name)
"""

