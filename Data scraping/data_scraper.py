import pylyrics3
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

def getArtists():
    with open('artists.txt') as f:
        return f.read().splitlines()

'''
def getIds(gen):
    ids_tosearch = []
    for artist in getArtists():
        if artist == 'Andre 3000' or artist == 'Nas':
            id = gen.search(artist)['response']['hits'][1]['result']['primary_artist']['id']
            name = gen.search(artist)['response']['hits'][1]['result']['primary_artist']['name']
        else:
            id = gen.search(artist)['response']['hits'][0]['result']['primary_artist']['id']
            name = gen.search(artist)['response']['hits'][0]['result']['primary_artist']['name']
        print(name,id)
        ids_tosearch.append(id)
        return ids_tosearch

base_url = "http://api.genius.com"
headers = {'Authorization': 'Bearer thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ'}
def lyrics_from_song_api_path(song_api_path):
  song_url = base_url + song_api_path
  response = requests.get(song_url, headers=headers)
  json = response.json()
  path = json['response']['song']['path']
  page_url = 'http://genius.com' + path
  page = requests.get(page_url)
  html = BeautifulSoup(page.text, 'html.parser')
  [h.extract() for h in html('script')]
  lyrics = html.find('div', class_='lyrics').get_text()
  return lyrics
'''

def scrape():
    with open('songs.csv','w',encoding='utf-8') as file:
        writer = csv.writer(file, delimiter = '|')
        artists = getArtists()
        for art in artists:
            lyrics = pylyrics3.get_artist_lyrics(art)
            for key, val in lyrics:
                newRow = [art, key, val]
                print(newRow)
                writer.writerow(newRow)


data = pd.read_csv('songs.csv',delimiter='|')
print(data.head())
