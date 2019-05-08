# Import the required libraries

import sys
import string
import time
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factorys = StemmerFactory()
stemmer = factorys.create_stemmer()
from sklearn.externals import joblib


# Processing Tweets

def emoji(data):
    # Senyum -- :), : ), :-), (:, ( :, (-:, :')
    data = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', '', data)
    # Tertawa -- :D, : D, :-D, xD, x-D, XD, X-D
    data = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', '', data)
    # Love -- <3, :*
    data = re.sub(r'(<3|:\*)', '', data)
    # Kedip -- ;-), ;), ;-D, ;D, (;,  (-;
    data = re.sub(r'(;-?\)|;-?D|\(-?;)', '', data)
    # Sedih -- :-(, : (, :(, ):, )-:
    data = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', '', data)
    # Menangis
    data = re.sub(r'(:,\(|:\'\(|:"\()', '', data)
    return data

def cleaning(data):
    # Menghilangkan tanda baca
    data = data.rstrip()
    translator = str.maketrans(' ', ' ', string.punctuation)
    # Menghilangkan unicode
    unli = re.compile(r'\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}')
    data = unli.sub(r'', data)
    data = data.replace('\\n', '')

    # Convert more than 2 letter repetitions to 2 letter funnnnny --> funny
    data = re.sub(r'(.)\1+', r'\1\1', data)
    # Menghilangkan - & '
    data = re.sub(r'(-|\')', '', data)
    # Menghilangkan link
    data = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', data)
    # Menghilangkan mention
    data = re.sub(r'@[\S]+', '', data)
    data = re.sub(r"^\s+", "", data, flags=re.UNICODE)
    data = data.strip()
    # Mengganti #hashtag menjadi hashtag
    data = re.sub(r'#(\S+)', r'\1', data)
    data = re.sub(r'(\S+)2', r'\1'+'-'+'\1', data)
    data = re.sub(' +', ' ', data)
    # Menghilangkan RT (redata)
    data = re.sub(r'\brt\b', '', data)
    # Mengganti 2+ titik dengan spasi
    data = re.sub(r'\.{2,}', ' ', data)
    data = data.translate(translator)
    # Menghilangkan Emoji
    data = emoji(data)
    data = re.sub('^[0-9]+', '', data)
    # Mengganti huruf menjadi kecil (lowercase)
    data = data.lower()

    return data

# Stemming of Tweets

def stemm(data):
    sf = StemmerFactory()
    stemmer = sf.create_stemmer()
    dstem = stemmer.stem(data)
    return dstem

def stopw(data):
    wf = StopWordRemoverFactory()
    more_stopword = ['hehe', 'wkwk']
    stop = wf.create_stop_word_remover()
    sword = stop.remove(data)
    return sword

# Predict the sentiment

def predict(tweet, classifier):
    tweet_processed = stem(preprocessTweets(tweet))

    if (('__positive__') in (tweet_processed)):
        sentiment = 1
        return sentiment

    elif (('__negative__') in (tweet_processed)):
        sentiment = 0
        return sentiment
    else:

        X = [tweet_processed]
        sentiment = classifier.predict(X)
        return (sentiment[0])


# Main function

# def main():
#     print('Loading the Classifier, please wait....')
#     classifier = joblib.load('svmClassifier.pkl')
#     print('READY')
#     tweet = ' '
#     for tweet in sys.stdin:
#         print(predict(tweet, classifier))
#
#
# if __name__ == "__main__":
#     main()