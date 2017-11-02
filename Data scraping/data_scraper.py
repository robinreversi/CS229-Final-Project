import pylyrics3
import csv
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from time import sleep
import random


def getUrls():
    with open('urls.txt') as f:
        return f.read().splitlines()

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
        urls = getUrls()
        ind = 0
        for url in urls:
            sleep(random.randint(0, 7))
            response = requests.get(url, headers={'User-Agent': random.choice(user_agents)},
                                    proxies=random.choice(proxies))
            soup = BeautifulSoup(response.content, 'lxml')
            song_urls = []
            song_names = []
            for song in soup.findAll(target='_blank'):
                song_urls.append(song['href'])
                song_names.append(str(song.text))
            for i in range(len(song_urls)):
                sleep(random.randint(0, 7))
                song_url = song_urls[i]
                song_url = 'https://www.azlyrics.com/'+ song_url[3:]
                resp = requests.get(song_url, headers={'User-Agent': random.choice(user_agents)},
                                    proxies=random.choice(proxies))
                song_soup = BeautifulSoup(resp.content, 'lxml')
                ring = song_soup.find('div',{'class' : None, 'id' : None})
                page_lyric = ring.text
                lyrics = re.sub('[(<.!,;?>/\-)]', " ", str(page_lyric)).split()
                lyrics = [word for word in lyrics if word != 'br']
                newRow = [artists[ind], song_names[i], ' '.join(lyrics)]
                print(newRow)
                writer.writerow(newRow)
            ind += 1

user_agents = [
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
        'Opera/9.25 (Windows NT 5.1; U; en)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
        'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
        'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.142 Safari/535.19',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:8.0.1) Gecko/20100101 Firefox/8.0.1',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.151 Safari/535.19']
proxies = [{"http": "http://107.170.13.140:3128"}, {"http": "http://198.23.67.90:3128"}]
scrape()
data = pd.read_csv('songs.csv',delimiter='|')
print(data.head())
