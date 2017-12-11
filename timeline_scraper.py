import pandas as pd
import kanye_periods
from py_genius import Genius
#from vocabulary_builder import buildVocabulary

artist = "Kanye West"

raw = pd.read_csv('data_scraping/finaldata.csv', delimiter="|")
artist_data = raw[raw['Artist'] == artist]
gen = Genius('thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ')

# Given a song 'title' returns the period it belongs to
def set_period(title):
    print("Song: ", title)
    result = gen.search(title)['response']['hits']

    for i in range(len(result)):
        if(result[i]['result']['primary_artist']['name'] == artist):
            album = gen.get_song(result[i]['result']['id'])['response']['song']['album']
            album = album if not album else album['name']
            if(album in kanye_periods.PERIOD_1):
                return 0
            elif(album in kanye_periods.PERIOD_2):
                return 1
            elif(album in kanye_periods.PERIOD_3):
                return 2
            else:
                return None

artist_data['Artist'] = artist_data['Title'].apply(set_period)
artist_data = artist_data.dropna(subset = ['Artist'])
artist_data.to_csv(artist + '_period_data.csv')


