from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from textblob import TextBlob


class DataTraining(object):
    dataFrame = pd.DataFrame()

    def __init__(self, cleanCsvPath="/Users/ramazancesur/Desktop/data/cleanTweets.csv"):
        print "data training class have running at this time " + str(datetime.now())
        csv = cleanCsvPath
        self.dataFrame = pd.read_csv(csv, index_col=0)
        self.dataFrame.dropna(inplace=True)
        self.dataFrame.reset_index(drop=True, inplace=True)

    def trainDataModelWithTextBlob(self):
        x = self.dataFrame.text
        y = self.dataFrame.target
        SEED = 2000
        x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.04,
                                                                                          random_state=SEED)
        x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                      test_size=.5, random_state=SEED)
        #
        # tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
        #
        # tbpred = [0 if n < 0 else 1 for n in tbresult]
        #
        # conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[1, 0]))
        #
        # confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
        #                          columns=['predicted_positive', 'predicted_negative'])
        # print "Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred) * 100)
        # print confusion

        print "RESULT FOR UNIGRAM WITHOUT STOP WORDS\n"
        feature_result_wosw = self.nfeature_accuracy_checker(trainingData=x, trainingDataClass=y,stop_words='english', ngram_range=(1,3))
        print feature_result_wosw

    def train_test_and_evaluate(self,pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                                 columns=['predicted_negative', 'predicted_positive'])
        print "null accuracy: {0:.2f}%".format(null_accuracy * 100)
        print "accuracy score: {0:.2f}%".format(accuracy * 100)
        if accuracy > null_accuracy:
            print "model is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100)
        elif accuracy == null_accuracy:
            print "model has the same accuracy with the null accuracy"
        else:
            print "model is {0:.2f}% less accurate than null accuracy".format((null_accuracy - accuracy) * 100)
        print "-" * 80
        print "Confusion Matrix\n"
        print confusion
        print "-" * 80
        print "Classification Report\n"
        print classification_report(y_test, y_pred, target_names=['negative', 'positive'])


    def accuracy_summary(self,pipeline, x_train, y_train, x_test, y_test):
        if len(x_test[y_test == 0]) / (len(x_test) * 1.) > 0.5:
            null_accuracy = len(x_test[y_test == 0]) / (len(x_test) * 1.)
        else:
            null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test) * 1.))
        t0 = time()
        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        train_test_time = time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        print "null accuracy: {0:.2f}%".format(null_accuracy * 100)
        print "accuracy score: {0:.2f}%".format(accuracy * 100)
        if accuracy > null_accuracy:
            print "model is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100)
        elif accuracy == null_accuracy:
            print "model has the same accuracy with the null accuracy"
        else:
            print "model is {0:.2f}% less accurate than null accuracy".format((null_accuracy - accuracy) * 100)
        print "train and test time: {0:.2f}s".format(train_test_time)
        print "-" * 80
        return accuracy, train_test_time


    cvec = CountVectorizer()
    lr = LogisticRegression()
    n_features = np.arange(10000, 100001, 10000)

    def nfeature_accuracy_checker(self,trainingData, trainingDataClass,vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1),
                                  classifier=lr):
        result = []
        print (classifier)
        print "\n"
        SEED=2000
        x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(trainingData, trainingDataClass, test_size=.04,
                                                                                          random_state=SEED)
        x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                      test_size=.5, random_state=SEED)

        for n in n_features:
            vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print "Validation result for {} features".format(n)
            nfeature_accuracy, tt_time = self.accuracy_summary(pipeline= checker_pipeline, x_train= x_train, y_train= y_train,
                                                                        x_test=x_test, y_test= y_test)
            result.append((n, nfeature_accuracy, tt_time))
        return result





def mainMethod():
    training = DataTraining()
    training.trainDataModelWithTextBlob()


if __name__ == '__main__':
    print "main function have call by DataTraining class"
    mainMethod()
