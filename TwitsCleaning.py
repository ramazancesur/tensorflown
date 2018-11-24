import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer


class TwitsCleaning(object):
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df = pd.read_csv("/Users/ramazancesur/Desktop/data/traindata.csv", header=None,usecols=[0,5], names=['sentiment','text'])
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

    def __init__(self):
        self.df.head()


    def tweet_cleaner_updated(self,text):
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9_]+'
        pat2 = r'https?://[^ ]+'
        combined_pat = r'|'.join((pat1, pat2))
        www_pat = r'www.[^ ]+'
        negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                         "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                         "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                         "can't": "can not", "couldn't": "could not", "shouldn't": "should not",
                         "mightn't": "might not",
                         "mustn't": "must not"}
        neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
        stripped = re.sub(combined_pat, '', bom_removed)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
        return (" ".join(words)).strip()


    def cleansBatchData(self, dataFrame=df):
        print "Cleaning the tweets...\n"
        clean_tweet_texts = []
        for i in xrange(0, len(dataFrame)):
            if ((i + 1) % 100000 == 0):
                print "Tweets %d of %d has been processed" % (i + 1, len(dataFrame))
            clean_tweet_texts.append(self.tweet_cleaner_updated(dataFrame['text'][i]))
        return clean_tweet_texts

    def writeCleanData(self ,cleanData, cleanPath, dataframe=df,):
        clean_df = pd.DataFrame(cleanData, columns=['text'])
        clean_df['target'] = dataframe.sentiment
        clean_df.to_csv(cleanPath, encoding='utf-8')

def mainFunction():
    apiClient = TwitsCleaning()
    cleansTweet=apiClient.cleansBatchData()
    apiClient.writeCleanData(cleanData=cleansTweet, cleanPath="/Users/ramazancesur/Desktop/data/cleanTweets.csv")


if __name__ == "__main__":
    # calling main function
    mainFunction()
    print "process have been start"



