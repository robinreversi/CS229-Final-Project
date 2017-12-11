import pandas as pd

all_songs = pd.read_csv('nobrack.csv',sep='|')

def remove_brackets(lyrics):
    ret = ''
    add = True
    for char in lyrics:
        if char == '(':
            add = False
        elif char == ')':
            add = True
        if add and char != ')':
            ret += char
    return ret

all_songs['Lyrics'] = all_songs['Lyrics'].apply(remove_brackets)
all_songs.to_csv('nobp.csv',sep='|',index = False,encoding='utf-8')
'''
def inter(title):
    if 'interview' in title or 'Interview' in title or 'Speech' in title or 'speech' in title or 'Original' in title or 'original' in title or 'Version' in title or 'version' in title or 'Radio' in title or 'Demo' in title:
        return 1
    else:
        return 0

all_songs['interview'] = all_songs['Title'].apply(inter)

no_inter = all_songs[all_songs['interview'] == 0]
print(all_songs.shape)
print(no_inter.shape)

no_inter.to_csv('finaldata.csv',sep='|',index = False, encoding='utf-8')
'''