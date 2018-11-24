import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


class TfIdfAndNgrams(object):

    def __init__(self,  csv = 'clean_tweet.csv'):
        my_df = pd.read_csv(csv, index_col=0)
        my_df.head()
        my_df.dropna(inplace=True)
        my_df.reset_index(drop=True, inplace=True)
        my_df.info()
        ### Represent of traning feature by x
        x = my_df.text
        ### Represent of traning result by yd
        y = my_df.target





def mainMethod(self):
    instance= TfIdfAndNgrams()

if __name__ == '__main__':
    mainMethod()