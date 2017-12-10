import pandas as pd
import kanye_periods
from py_genius import Genius

artist = "Kanye West"

raw = pd.read_csv('data_scraping/finaldata.csv', delimiter="|")
artist_data = raw[raw['Artist'] == artist]
gen = Genius('thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ')

def set_period(title):
    print("Song: ", title)
    result = gen.search(title)['response']['hits']

    for i in range(len(result)):
        if(result[i]['result']['primary_artist']['name'] == artist):
            album = gen.get_song(result[i]['result']['id'])['response']['album']
            print(album)


for song in artist_data['Title']:
    set_period(song)
'''
def set_period(title):
    if(title in kanye_periods.PERIOD_1):
        return 0
    elif(title in kanye_periods.PERIOD_2):
        return 1
    elif(title in kanye_periods.PERIOD_3):
        return 2
    else:
        return None

kanye['Period'] = kanye['Title'].apply(set_period)
kanye = kanye.dropna(subset = ['Period'])

kanye.to_csv('kanye_period_data.csv')

'''