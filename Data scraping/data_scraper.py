from py_genius import Genius
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

def getArtists():
    with open('artists.txt') as f:
        return f.read().splitlines()


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

def scrape():
    with open('songs.csv','w',encoding='utf-8') as file:
        writer = csv.writer(file, delimiter = '|')
        gen = Genius('thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ')
        ids = list(map(int, getArtists()))
        for id in ids:
            response = gen.get_artist(id)
            name = response['response']['artist']['name']
            title = ''
            k = 0
            num_songs = 0
            lst = []
            response = gen.get_artist_songs(id)
            print(response)
            while True:
                page_num = 1
                print(k)
                print(name)
                song = response['response']['songs'][k]
                outcome = song['primary_artist']['name']
                title = song['title']
                if name == outcome and song['title'] == song['title_with_featured']:
                    newRow = [name, title, lyrics_from_song_api_path(song['api_path']).replace('\n', ' ')]
                    print(newRow)
                    writer.writerow(newRow)
                    if num_songs >= 30:
                        break
                    num_songs+=1
                k+=1
                if k == 18:
                    response = gen.get_artist_songs(id, page=page_num+1)
                    k = 0


data = pd.read_csv('songs.csv',delimiter='|')
print(data.head())