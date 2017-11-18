import pylyrics3
from py_genius import Genius
import csv
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from time import sleep
import random


def getIDs():
    with open('ids.txt') as f:
        return f.read().splitlines()

def getArtists():
    with open('artists.txt') as f:
        return f.read().splitlines()




def writeIDs(gen):
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
    with open('ids.txt','w') as file:
        file.writelines(map(str,ids_tosearch))

base_url = "http://api.genius.com"
headers = {'Authorization': 'Bearer thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ'}
def lyrics_from_song_api_path(song_api_path):
    print(song_api_path)
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
    with open('test4.csv','w',encoding='utf-8') as file:
        writer = csv.writer(file, delimiter = '|')
        gen = Genius('thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ')
        ids = list(map(int, getIDs()))[4:]
        artists = getArtists()
        curr = 4
        seen = set()
        for id in ids:
            print('============New Artist===========')
            artist = artists[curr]
            title = ''
            k = 0
            lst = []
            response = gen.get_artist_songs(id)
            num_songs = 0
            page_num = 1
            while True:
                print(k)
                song = response['response']['songs'][k]
                main_artist = song['primary_artist']['name']
                outcome = song['primary_artist']['name']
                title = song['title']
                if song['primary_artist']['id'] == id and (not title[:5] in seen or artist == 'Nas'):
                    print(title)
                    print(seen)
                    newRow = [artist, title, lyrics_from_song_api_path(song['api_path']).replace('\n', ' ')]
                    print(newRow)
                    writer.writerow(newRow)
                    seen.add(title[:5])
                    num_songs += 1
                k += 1
                if k == len(response['response']['songs']) - 1:
                    k = 0
                    page_num += 1
                    response = gen.get_artist_songs(id,page=page_num)
                if num_songs == 29:
                    print(num_songs)
                    break
            curr += 1

scrape()
data = pd.read_csv('test2.csv',delimiter='|')
print(data.head())

