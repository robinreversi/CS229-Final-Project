
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
    with open('test.csv','w',encoding='utf-8') as file:
        writer = csv.writer(file, delimiter = '|')
        gen = Genius('thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ')
        ids = list(map(int, getIDs()))
        artists = getArtists()
        for artist in artists:
            name = artist
            title = ''
            k = 9
            num_songs = 0
            lst = []
            response = gen.search(artist)
            page_num = 1
            while True:
                print(k)
                print(name)
                song = response['response']['hits'][k]['result']
                print(len(response['response']['hits']))
                outcome = song['primary_artist']['name']
                title = song['title']
                newRow = [name, title, lyrics_from_song_api_path(song['api_path']).replace('\n', ' ')]
                print(newRow)
                writer.writerow(newRow)
                k += 1
                #if k == len(response['response']['hits']) - 1:
                break
scrape()
data = pd.read_csv('songs.csv',delimiter='|')
print(data.head())

