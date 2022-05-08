import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import feature_extraction 
stop_words = feature_extraction.text.ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
import pickle



#scrape nbc news 

#list of nbc links to scrape

nbc_links = ["https://www.nbcnews.com/business", 'https://www.nbcnews.com/politics']

#list of topics

nbc_topics = ['business/tech', 'politics']

count = 0

#empty lists to hold headlines and their labels 

nbc_articles = []
nbc_labels = []

#get each headline in from the nbc links 

for nbc in nbc_links:
    res = requests.get(nbc)
    soup = BeautifulSoup(res.content, 'html.parser')
    
    headlines = soup.find_all('span',{'class':'tease-card__headline'})
    
    for headline in headlines:
        #append the headline to nbc_articles list
        nbc_articles.append(headline.get_text())
        #append the label (either business or tech depending on which link the loop is on)
        nbc_labels.append(nbc_topics[count])
    #update the counter variable
    count += 1 
#create a dictionary of headlines and their labels. then convert to dataframe
nbc_dict = {'Headline':nbc_articles, 'Label': nbc_labels}
nbc_data = pd.DataFrame(nbc_dict)

    

#scrape cnn 

cnbc_link = "https://www.cnbc.com/politics/"

cnbc_articles = []
cnbc_labels = []
res = requests.get(cnbc_link)
soup = BeautifulSoup(res.content, 'html.parser')
    
headlines = soup.find_all('div',{'class':'Card-titleContainer'})
        
    
for headline in headlines:
    cnbc_articles.append(headline.get_text())
    cnbc_labels.append('politics')
    
cnbc_dict = {'Headline':cnbc_articles, 'Label':cnbc_labels}
cnbc_data = pd.DataFrame(cnbc_dict)

#conatentate the data from cnn and nbc

fulldata = pd.concat([nbc_data,cnbc_data], axis = 0)

#change the format of the column names of fulldata

fulldata = fulldata.rename(columns = {'Headline':'headline', 'Lable':'Label'})

#text preprocessing and stemming


def preprocess(text):
  text = text.lower() #lowercase
  text = re.sub(r'[^\w\s]', '', text) #remove punctuations
  text = re.sub(r'\d+', '', text) #remove numbers
  text = " ".join(text.split()) #stripWhitespace
  text = text.split()
  text = [x for x in text if x not in stop_words] #remove stopwords
  text = " ".join(text)
  stemmer_ps = PorterStemmer()  
  text = [stemmer_ps.stem(word) for word in text.split()] #stemming
  text = " ".join(text)
  return(text)
  
#define stop words to remove later

stop_words = feature_extraction.text.ENGLISH_STOP_WORDS


#preprocess the headlines and store in the text column

fulldata['text'] = fulldata['headline'].apply(lambda x:preprocess(x))

alltext = fulldata['text'].tolist()

#load the pre-fitted TFIDF vectorizer

vectorizer = pickle.load(open("vectorizer (1).pickle", "rb"))

#Vectorize the TFIDF values for the current data

TFIDF = vectorizer.transform(alltext)
TFIDFdf=pd.DataFrame(TFIDF.toarray(),columns=vectorizer.get_feature_names())
TFIDFdf.columns = ['tfidf_'+x for x in TFIDFdf.columns]




#load the logistic regression model

with open('news_logistic_regression.p', 'rb') as l:
    loaded_model = pickle.load(l)
    
#Classify the news articles as business/tech (1) or politics (0)

y_pred = loaded_model.predict(TFIDFdf) 

#put predictions into the fulldata 

fulldata['prediction'] = y_pred

#Seperate into busines/tech and politics dataframes

Btech = fulldata[fulldata['prediction'] == 1]
Pol = fulldata[fulldata['prediction'] == 0] 

#Keep relevant Columns

Btech2 = Btech[['headline']].copy()
Pol2 = Pol[['headline']].copy()

#User Interface

st.title('News Aggregator')
st.header('News Alerts Powered By Machine Learning')
st.text('')
st.markdown('This app scrapes CNN and NBC for current news headlines related to business/tech and politics. These headlines are then transformed into numerical vectors using the TFIDF method from Natural Language Processing. Next, these numerical vectors are fed into a logistic regression model, which was trained on about 4,000 news articles from the Huffington Post. This Logistic Regression model then classifies the news stories into either business/tech or politics. This app can be used to get quick alerts to stay informed about current events.')

st.text('')
st.text('')

st.header('Business/Tech')
st.dataframe(Btech2)

st.text('')
st.header('Politics')
st.dataframe(Pol2)

#Allow user input to test the system
st.header('Test the Model')
st.text('')
custom_headline = [st.text_input("Enter a custom headline about business/tech or politics")]
custom_preprocessed = [preprocess(custom_headline[0])]

custom_headline_tfidf = vectorizer.transform(custom_preprocessed)
custom_predict = loaded_model.predict(custom_headline_tfidf)
custom_predict_prob = loaded_model.predict_proba(custom_headline_tfidf)

prob_bisTech = custom_predict_prob.tolist()[0][1] * 100
prob_pol = custom_predict_prob.tolist()[0][0] * 100
st.markdown('The model predicts that this headline is business/tech with a probability of {:.2f}% and politics with a probability of {:.2f}%'.format(prob_bisTech, prob_pol))
