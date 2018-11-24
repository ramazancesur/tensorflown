import pandas as pd
import numpy as np
from wordcloud import WordCloud
from scipy.stats import hmean
import matplotlib

import seaborn as sns


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.feature_extraction.text import CountVectorizer


class DataVirtualization(object):
    def __init__(self):
        print "data virtualizetion have been start"
        mainDataFrame= self.dataCleaning(cleanFile="/Users/ramazancesur/Desktop/data/cleanTweets.csv",
                                                            trainFile="/Users/ramazancesur/Desktop/data/trainData.csv")

    def dataCleaning(self, cleanFile, trainFile):
        cleadDataFrame = pd.read_csv(cleanFile, index_col=0)
        cleadDataFrame.head()

        df = pd.read_csv(trainFile, header=None)
        df.iloc[cleadDataFrame[cleadDataFrame.isnull().any(axis=1)].index, :].head()

        cleadDataFrame.dropna(inplace=True)
        cleadDataFrame.reset_index(drop=True, inplace=True)
        cleadDataFrame.info()
        self.showNegativeKeyWord(my_df=cleadDataFrame)
        self.showPositiveKeyWord(cleadDataFrame)
        return cleadDataFrame


    def showNegativeKeyWord(self, my_df):
        neg_tweets = my_df[my_df.target == 0]
        neg_string = []
        for t in neg_tweets.text:
            neg_string.append(t)
        neg_string = pd.Series(neg_string).str.cat(sep=' ')

        wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(neg_string)
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def showNegativeAndPositiveTweetGraphic(self, neg_tf, pos_tf , cleaningDataFrame):
        cvec = CountVectorizer()
        cvec.fit(cleaningDataFrame.text)

        neg = np.sum(neg_tf, axis=0)
        pos = np.sum(pos_tf, axis=0)
        term_freq_df2 = pd.DataFrame([neg, pos], columns=cvec.get_feature_names()).transpose()
        term_freq_df2.columns = ['negative', 'positive']
        term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
        term_freq_df2.sort_values(by='total', ascending=False).iloc[:10]

        y_pos = np.arange(50)
        plt.figure(figsize=(12, 10))
        plt.bar(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50], align='center',
                alpha=0.5)
        plt.xticks(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50].index,
                   rotation='vertical')
        plt.ylabel('Frequency')
        plt.xlabel('Top 50 negative tokens')
        plt.title('Top 50 tokens in negative tweets')

        y_pos = np.arange(50)
        plt.figure(figsize=(12, 10))
        plt.bar(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50], align='center',
                alpha=0.5)
        plt.xticks(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,
                   rotation='vertical')
        plt.ylabel('Frequency')
        plt.xlabel('Top 50 positive tokens')
        plt.title('Top 50 tokens in positive tweets')

        term_freq_df2['pos_hmean'] = term_freq_df2.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])
                                                                    if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0
                                                                    else 0), axis=1)
        term_freq_df2.sort_values(by='pos_hmean', ascending=False).iloc[:10]

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(x="neg_normcdf_hmean", y="pos_normcdf_hmean", fit_reg=False, scatter_kws={'alpha': 0.5},
                         data=term_freq_df2)
        plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
        plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
        plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')



    def dataFrequency(self, cleaningDataFrame, frequenceCsvPath):
        cvec = CountVectorizer()
        cvec.fit(cleaningDataFrame.text)
        # number of twits word count
        print len(cvec.get_feature_names())
        ## negative keyword matrix

        document_matrix = cvec.transform(cleaningDataFrame.text)
        neg_batches = np.linspace(0, 798179, 100).astype(int)
        i = 0
        ## return negatif data frame range table which last 10 data row
        print cleaningDataFrame[cleaningDataFrame.target == 0].tail()

        neg_tf = []
        while i < len(neg_batches) - 1:
            batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i + 1]].toarray(), axis=0)
            neg_tf.append(batch_result)
            if (i % 10 == 0) | (i == len(neg_batches) - 2):
                print neg_batches[i + 1], "entries' term freuquency calculated"
            i += 1

        pos_batches = np.linspace(798179, 1596019, 100).astype(int)
        i = 0
        ## return positive data frame range table which last 10 data row

        print cleaningDataFrame[cleaningDataFrame.target == 1].tail()
        pos_tf = []
        while i < len(pos_batches) - 1:
            batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i + 1]].toarray(), axis=0)
            pos_tf.append(batch_result)
            if (i % 10 == 0) | (i == len(pos_batches) - 2):
                print pos_batches[i + 1], "entries' term freuquency calculated"
            i += 1

        neg = np.sum(neg_tf, axis=0)
        pos = np.sum(pos_tf, axis=0)
        term_freq_df = pd.DataFrame([neg, pos], columns=cvec.get_feature_names()).transpose()
        term_freq_df.head()

        term_freq_df.columns = ['negative', 'positive']
        term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
        term_freq_df.sort_values(by='total', ascending=False).iloc[:10]
        term_freq_df.to_csv(frequenceCsvPath+"term_freq_df.csv", encoding="utf-8")
        print "word frequence summerizing printed csv file  with missed complated"


    def showPositiveKeyWord(self, my_df):
        pos_tweets = my_df[my_df.target == 1]
        pos_string = []
        for t in pos_tweets.text:
            pos_string.append(t)
        pos_string = pd.Series(pos_string).str.cat(sep=' ')

        wordcloud = WordCloud(width=1600, height=800, max_font_size=200, colormap='magma').generate(pos_string)
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

def mainFunction():
    dataVirt= DataVirtualization()
    print "main function have be called"
if __name__ == '__main__':
    mainFunction()
    print "process is completed"
