# Email-SMS-Spam-Detection


## Email Spam Detection By Sumit
#### Email/SMS spam, are also called as junk emails/sms, are unsolicited messages sent in bulk by email (spamming).

##### In this Data Science Project I will show you how to detect email spam using Machine Learning technique called Natural Language Processing and Python.

# Import All Library
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Load the data and print the first 5 rows :
data_sms = pd.read_csv('spam.csv',encoding='latin-1')
data_sms.head()

# Now let’s explore the data and get the number of rows & columns :
data_sms.shape

# Count the columns
data_sms.columns.value_counts()

# Drop the unnecessary columns
data_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1,inplace = True)
data_sms.rename(columns={"v1":"label", "v2":"sms"})
data_sms.head()

# Count the how many of normal and spam
data_sms["v1"].value_counts()

# calculate the leanth of all text that present in "v2"
data_sms['length'] = data_sms['v2'].apply(len)
data_sms.head()

# rename the all columns in sutable or readble 
data_sms.rename( columns = {"v1" : "Label" , "v2" : "SMS" , "length" : "Length"},inplace  =True)

data_sms.head()

# see the frequescy of text length
data_sms['Length'].plot(bins=50, kind='hist')

# creating the plot to show how many are spam or normal
data_sms.hist(column = "Length" , by = "Label")
plt.show()

# conver the ham and spam in binery form label incoded
data_sms.loc[:,'Label'] = data_sms.Label.map({'ham':0, 'spam':1})
data_sms.head()

# check the null values
data_sms.isnull().sum()

# Now Download the stop words
nltk.download("stopwords")

##### Now Create a function to clean the text and return the tokens. The cleaning of the text can be done by first removing punctuation and then removing the useless words also known as stop words.


def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

# to show the tokenization
data_sms['SMS'].head().apply(process)

# Now convert the text into a matrix of token counts :

from sklearn.feature_extraction.text import CountVectorizer
message = CountVectorizer(analyzer=process).fit_transform(data_sms['SMS'])

#split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(message, data_sms['Label'], test_size=0.20, random_state=0)
# To see the shape of the data
message.shape

#### Now we need to create and train the Multinomial Naive Bayes classifier which is suitable for classification with discrete features.

# create and train the Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(xtrain, ytrain)

# To see the classifiers prediction and actual values on the data set :
classifier.predict(xtrain)
ytrain.values

#### Now let’s see how well our model performed by evaluating the Naive Bayes classifier and the report, confusion matrix & accuracy score.

# Evaluating the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(xtrain)
print(classification_report(ytrain, pred))
print()
print("Confusion Matrix: \n", confusion_matrix(ytrain, pred))
print("Accuracy: \n", accuracy_score(ytrain, pred))

#### It looks like the model used is 99.46% accurate. Let’s test the model on the test data set (xtest &  ytest) by printing the predicted value, and the actual value to see if the model can accurately classify the email text.

#print the predictions
print(classifier.predict(xtest))
#print the actual values
print(ytest.values)

### Now let’s evaluate the model on the test data set :

# Evaluating the model on the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(xtest)
print(classification_report(ytest, pred))
print()
print("Confusion Matrix: \n", confusion_matrix(ytest, pred))
print("Accuracy: \n", accuracy_score(ytest, pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: {}'.format(accuracy_score(ytest, pred)))
print('Precision score: {}'.format(precision_score(ytest, pred)))
print('Recall score: {}'.format(recall_score(ytest, pred)))
print('F1 score: {}'.format(f1_score(ytest, pred)))

## The classifier accurately identified the email messages as spam or not spam with 95.78 % accuracy on the test data.

# ------- Thanks By Sumit ---------

