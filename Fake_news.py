#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pickle


# In[9]:


dataset=pd.read_csv('C:\\Users\\makaria\\.conda\\news.csv')
x=dataset['text']
y=dataset['label']


# In[10]:


dataset.head()


# In[12]:


dataset.shape


# In[13]:


dataset.isnull().any()


# In[14]:


x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2)


# In[15]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[16]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[17]:


pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', MultinomialNB())])


# In[18]:


pipeline.fit(x_train, y_train)


# In[19]:


score=pipeline.score(x_test,y_test)
print('acuracy',score)


# In[20]:


pred= pipeline.predict(x_test)


# In[23]:


print(classification_report(y_test, pred))


# In[24]:


print(confusion_matrix(y_test, pred))


# In[27]:


with open('mode1.pk1','rb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:




