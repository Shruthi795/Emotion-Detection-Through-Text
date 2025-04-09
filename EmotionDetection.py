#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!pip install lime


# In[1]:


import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()

# Modelling
from sklearn.model_selection import train_test_split,KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.svm import SVC

#Lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from lime.lime_text import IndexedString,IndexedCharacters
from lime.lime_base import LimeBase
from lime.lime_text import explanation
sns.set(font_scale=1.3)
nltk.download('omw-1.4')


# In[2]:


# Read datasets
df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')


# In[3]:


df_train.head()


# In[4]:


print(df_train.shape)


# In[5]:


df_test.head()


# In[6]:


print(df_test.shape)


# In[7]:


df_val.head()


# In[8]:


print(df_val.shape)


# In[9]:


df_train.Emotion.value_counts()


# In[10]:


df_train.Emotion.value_counts() / df_train.shape[0] *100


# In[11]:


plt.figure(figsize=(8,4))
sns.countplot(x='Emotion', data=df_train);


# In[12]:


df_train.isnull().sum()


# In[13]:


df_train.duplicated().sum()


# In[14]:


#removing duplicated values
index = df_train[df_train.duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)


# In[15]:


df_train[df_train['Text'].duplicated() == True]


# In[16]:


df_train[df_train['Text'] == df_train.iloc[7623]['Text']]


# In[17]:


df_train[df_train['Text'] == df_train.iloc[14313]['Text']]


# In[18]:


df_train[df_train['Text'] == df_train.iloc[13879]['Text']]


# In[19]:


#removing duplicated text
index = df_train[df_train['Text'].duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)


# In[20]:


#Count the number of stopwords in the data
temp =df_train.copy()
stop_words = set(stopwords.words("english"))
temp['stop_words'] = temp['Text'].apply(lambda x: len(set(x.split()) & set(stop_words)))
temp.stop_words.value_counts()


# In[21]:


temp['stop_words'].plot(kind= 'hist')


# In[22]:


df_test.Emotion.value_counts()


# In[23]:


plt.figure(figsize=(8,4))
sns.countplot(x='Emotion', data=df_test);


# In[24]:


df_test.isnull().sum()


# In[25]:


df_test.duplicated().sum()


# In[26]:


df_test[df_test['Text'].duplicated() == True]


# In[27]:


#Count the number of stopwords in the data
temp =df_test.copy()
temp['stop_words'] = temp['Text'].apply(lambda x: len(set(x.split()) & set(stop_words)))
temp.stop_words.value_counts()


# In[28]:


sns.set(font_scale=1.3)
temp['stop_words'].plot(kind= 'hist')


# In[29]:


df_val.Emotion.value_counts()


# In[30]:


plt.figure(figsize=(8,4))
sns.countplot(x='Emotion', data=df_val);


# In[31]:


df_val.isnull().sum()


# In[32]:


df_val.duplicated().sum()


# In[33]:


df_val[df_val['Text'].duplicated() == True]


# In[34]:


df_val[df_val['Text'] == df_val.iloc[603]['Text']]


# In[35]:


df_val[df_val['Text'] == df_val.iloc[1993]['Text']]


# In[36]:


#removing duplicated text
index = df_val[df_val['Text'].duplicated() == True].index
df_val.drop(index, axis = 0, inplace = True)
df_val.reset_index(inplace=True, drop = True)


# In[37]:


#Count the number of stopwords in the data
temp =df_val.copy()
temp['stop_words'] = temp['Text'].apply(lambda x: len(set(x.split()) & set(stop_words)))
temp.stop_words.value_counts()[:10]


# In[38]:


sns.set(font_scale=1.3)
temp['stop_words'].plot(kind= 'hist');


# In[39]:


def dataframe_difference(df1, df2, which=None):
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


# In[40]:


dataframe_difference(df_train, df_test, which='both')


# In[41]:


dataframe_difference(df_train, df_val, which='both')


# In[42]:


dataframe_difference(df_val, df_test, which='both')


# In[43]:


def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence


# In[44]:


#nltk.download('wordnet')


# In[45]:


normalized_sentence("My Name is Mohamed. @Tweets,  plays 2022  Egypt_")


# In[46]:


df_train= normalize_text(df_train)
df_test= normalize_text(df_test)
df_val= normalize_text(df_val)


# In[47]:


X_train = df_train['Text'].values
y_train = df_train['Emotion'].values

X_test = df_test['Text'].values
y_test = df_test['Emotion'].values

X_val = df_val['Text'].values
y_val = df_val['Emotion'].values


# In[48]:


def train_model(model, data, targets):
    # Create a Pipeline object with a TfidfVectorizer and the given model
    text_clf = Pipeline([('vect',TfidfVectorizer()),
                         ('clf', model)])
    # Fit the model on the data and targets
    text_clf.fit(data, targets)
    return text_clf


# In[49]:


def get_F1(trained_model,X,y):
    # Make predictions on the input data using the trained model
    predicted=trained_model.predict(X)
    # Calculate the F1 score for the predictions
    f1=f1_score(y,predicted, average=None)
    # Return the F1 score
    return f1


# In[50]:


#Train the model with the training data
log_reg = train_model(LogisticRegression(solver='liblinear',random_state = 0), X_train, y_train)


# In[51]:


#Make a single prediction
y_pred=log_reg.predict(['Happy'])
y_pred


# In[52]:


#test the model with the test data
y_pred=log_reg.predict(X_test)

#calculate the accuracy
log_reg_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', log_reg_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(log_reg,X_test,y_test)
pd.DataFrame(f1_Score, index=df_train.Emotion.unique(), columns=['F1 score'])


# In[53]:


print(classification_report(y_test, y_pred))


# In[54]:


#Train the model with the training data
DT = train_model(DecisionTreeClassifier(random_state = 0), X_train, y_train)

#test the model with the test data
y_pred=DT.predict(X_test)

#calculate the accuracy
DT_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', DT_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(DT,X_test,y_test)
pd.DataFrame(f1_Score, index=df_train.Emotion.unique(), columns=['F1 score'])


# In[55]:


print(classification_report(y_test, y_pred))


# In[56]:


#Train the model with the training data
SVM = train_model(SVC(random_state = 0), X_train, y_train)

#test the model with the test data
y_pred=SVM.predict(X_test)

#calculate the accuracy
SVM_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', SVM_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(SVM,X_test,y_test)
pd.DataFrame(f1_Score, index=df_train.Emotion.unique(), columns=['F1 score'])


# In[57]:


print(classification_report(y_test, y_pred))


# In[58]:


#Train the model with the training data
RF = train_model(RandomForestClassifier(random_state = 0), X_train, y_train)

#test the model with the test data
y_pred=RF.predict(X_test)

#calculate the accuracy
RF_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', RF_accuracy,'\n')

#calculate the F1 score
f1_Score = get_F1(RF, X_test, y_test)
pd.DataFrame(f1_Score, index=df_train.Emotion.unique(), columns=['F1 score'])


# In[59]:


print(classification_report(y_test, y_pred))


# In[60]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree','Support Vector Machine','Random Forest'],
    'Accuracy': [log_reg_accuracy.round(2), DT_accuracy.round(2), SVM_accuracy.round(2), RF_accuracy.round(2)]})

models.sort_values(by='Accuracy', ascending=False).reset_index().drop(['index'], axis=1)


# In[61]:


explainer_LR = LimeTextExplainer(class_names=RF.classes_)
idx  = 15
print("Actual Text : ", X_test[idx])
print("Prediction : ", RF.predict(X_test)[idx])
print("Actual :     ", y_test[idx])
exp = explainer_LR.explain_instance(X_test[idx], RF.predict_proba,top_labels=5)
exp.show_in_notebook()


# In[62]:


explainer_LR = LimeTextExplainer(class_names=RF.classes_)
idx  = 5
print("Actual Text : ", X_test[idx])
print("Prediction : ", RF.predict(X_test)[idx])
print("Actual :     ", y_test[idx])
exp = explainer_LR.explain_instance(X_test[idx], RF.predict_proba,top_labels=5)
exp.show_in_notebook()


# In[63]:


#Make a single prediction
y_pred=log_reg.predict(['The loss of a loved one left me feeling empty and drained'])
y_pred


# In[64]:


#Make a single prediction
y_pred=log_reg.predict(['I couldnt stop smiling when I saw my newborn baby for the first time'])
y_pred


# In[65]:


y_pred=log_reg.predict(['i am feeling outraged it shows everywhere'])
y_pred


# In[66]:


y_pred=log_reg.predict(['I feel a little nervous i go to the gym'])
y_pred


# In[67]:


y_pred=log_reg.predict(['I want each of you to feel my gentle embrace'])
y_pred


# In[68]:


y_pred=log_reg.predict(['i love and captured an atmospheric feeling in their landscapes that really impressed me'])
y_pred


# In[69]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


# In[70]:


pipe_rf = Pipeline(steps=[('cv',CountVectorizer()),('rf', RandomForestClassifier())])
pipe_rf.fit(X_train,y_train)
pipe_rf.score(X_test,y_test)


# In[71]:


import joblib
pipeline_file = open("text_emotion.pkl","wb")
joblib.dump(pipe_rf,pipeline_file)
pipeline_file.close()


# In[ ]:




