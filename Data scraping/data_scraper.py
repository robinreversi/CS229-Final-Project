import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

def getArtists():
    with open('artists.txt') as f:
        return f.read().splitlines()


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
