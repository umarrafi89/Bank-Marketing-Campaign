#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np
import pandas as pd
import os,warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import plotly.express as px


# In[359]:


raw_data=pd.read_csv("bank (1).csv")


# In[163]:


raw_data.shape


# In[164]:


raw_data.head()


# In[165]:


raw_data.info()


# In[166]:


raw_data.nunique()


# In[167]:


raw_data


# In[168]:


px.histogram(raw_data, x = 'age', title = 'age vs deposit', color = 'deposit')


# In[169]:


px.histogram(raw_data, x = 'marital', title = 'marital vs deposit', color = 'deposit')


# In[170]:


px.strip(raw_data.sample(10000), x = 'age', y = 'loan', title = 'age vs loan', color = 'deposit')


# In[171]:


px.histogram(raw_data, x = 'loan', title = 'loan vs deposit', color = 'deposit')


# In[172]:


px.histogram(raw_data, x = 'balance', title = 'balance vs deposit', color = 'deposit')


# In[173]:


px.histogram(raw_data, x = 'housing', title = 'housing vs deposit', color = 'deposit')


# In[174]:


px.histogram(raw_data, x = 'duration', title = 'duration vs deposit', color = 'deposit')


# In[175]:


px.histogram(raw_data, x = 'month', title = 'month vs deposit', color = 'deposit')


# In[176]:


#it is quite 
#common to split the data into three sets, training sets, validation set and test set.


# In[177]:


from sklearn.model_selection import train_test_split


# In[178]:


train_val_df, test_df = train_test_split(raw_data, test_size = 0.2, random_state = 42)
train_df, val_df = train_test_split(train_val_df, test_size = 0.25, random_state = 42)


# In[179]:


print(train_val_df.shape)


# In[180]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# In[181]:


input_cols = list(train_df.columns)[:-1]
target_col = 'deposit'


# In[182]:


print(input_cols)


# In[183]:


print(target_col)


# In[184]:


type(target_col)


# In[185]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


# In[186]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()


# In[187]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


# In[188]:


train_targets


# In[189]:


train_inputs


# In[190]:


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
numeric_cols


# In[191]:


categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
categorical_cols


# In[192]:


train_inputs[numeric_cols].describe()


# In[193]:


train_inputs[categorical_cols].describe()


# In[194]:


train_inputs[categorical_cols].nunique()


# In[195]:


#imputation is a technique to fill out Nan values with some average value sklearn.impute


# In[196]:


from sklearn.preprocessing import MinMaxScaler


# In[197]:


scaler = MinMaxScaler()


# In[198]:


scaler.fit(train_inputs[numeric_cols])


# In[199]:


print('maximum')
list(scaler.data_max_)


# In[200]:


print('minimum')
list(scaler.data_min_)


# In[201]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# ML tenchinque only works for the numeric data, we need to convert the categorical data into numeric data as well.
# A common tenchique is to use one-hot encoding for categorical columns is also called as One hot vector
# In[202]:


train_inputs[numeric_cols].describe()


# In[203]:


val_inputs[numeric_cols].describe()


# In[204]:


test_inputs[numeric_cols].describe()


# In[205]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse = False, handle_unknown = "ignore")


# In[206]:


encoder.fit(train_inputs[categorical_cols])


# In[207]:


encoder.categories_


# In[208]:


categorical_cols


# In[209]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)


# In[210]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# In[211]:


train_inputs[encoded_cols]


# In[212]:


val_inputs[encoded_cols]


# In[213]:


test_inputs[encoded_cols]


# In[214]:


from sklearn.linear_model import LogisticRegression


# In[321]:


model = LogisticRegression(solver = "lbfgs", max_iter = 1000)


# In[322]:


get_ipython().run_cell_magic('time', '', 'model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)')


# In[323]:


#print length of each array
print(len(numeric_cols), len(encoded_cols))


# In[324]:


weight_df = pd.DataFrame({'features': (numeric_cols + encoded_cols),
              'weight': model.coef_.tolist()[0]
             })


# In[325]:


print(model.intercept_)


# In[326]:


plt.figure(figsize= (20,50))
sns.barplot(data = weight_df.sort_values('weight', ascending = False), x = 'weight', y= 'features')


# In[221]:


# Making predictions and Evaluating model


# In[327]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val   = val_inputs[numeric_cols + encoded_cols]
X_test  = test_inputs[numeric_cols + encoded_cols]


# In[328]:


train_pred = model.predict(X_train)
train_pred


# In[224]:


train_targets


# In[225]:


relation = pd.DataFrame({'predic': train_pred,
              'target': train_targets
             })
relation


# In[226]:


train_probs = model.predict_proba(X_train)
train_probs


# In[227]:


model.classes_


# In[228]:


# check accuracy


# In[329]:


from sklearn.metrics import accuracy_score


# In[330]:


accuracy_score(train_targets, train_pred)


# In[331]:


from sklearn.metrics import confusion_matrix


# In[332]:


confusion_matrix(train_targets, train_pred, normalize = 'true')


# In[233]:


#among actual no all the deposit suppose to be no it predicts 86% it predicts "no"
#among the casess when actual deposit is "yes" it predicted "no" and its 78%times its yes 


# In[333]:


def predict_and_plot(inputs, targets, name = ''):
    preds = model.predict(inputs)
    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize = "true")
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.title("{} Confusion Matrix".format(name));
    
    return preds


# In[334]:


train_preds = predict_and_plot(X_train, train_targets, 'Training')


# In[236]:


val_preds = predict_and_plot(X_val, val_targets, 'Validation')


# In[237]:


test_preds = predict_and_plot(X_test, test_targets, 'Testing')


# In[238]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    # input_df[numeric_cols] = #imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0] # for 1 row
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred,prob


# In[239]:


new_input = {'age' : "95",
             'job' : "no",
             'marital': 'single',
             'education' : 'primary',
             'default': 'no',
             'balance': '5',
             'housing': 'yes',
             'loan': "no",
             'contact': 'unknown',
             'day': '25',
             'month': 'december',
             'duration' : "2",
             'campaign': '3',
             'pdays': '-1',
            'previous': '0',
            'poutcome': 'unkown'}


# In[240]:


predict_input(new_input)


# ## Naive Bayes Classification
# 

# In[242]:


from sklearn.naive_bayes import GaussianNB
modelG = GaussianNB().fit(train_inputs[numeric_cols + encoded_cols], train_targets)
modelG


# In[243]:


X_trainN = train_inputs[numeric_cols + encoded_cols]
X_valN   = val_inputs[numeric_cols + encoded_cols]
X_testN  = test_inputs[numeric_cols + encoded_cols]


# In[244]:


train_predN = modelG.predict(X_trainN)
train_predN


# In[256]:


train_probsN = modelG.predict_proba(X_trainN)
train_probsN


# In[257]:


accuracy_score(train_targets, train_predN)


# In[258]:


confusion_matrix(train_targets, train_predN, normalize = 'true')


# In[259]:


train_predsN = predict_and_plot(X_trainN, train_targets, 'Training')


# In[260]:


val_predsN = predict_and_plot(X_valN, val_targets, 'Validation')


# In[261]:


test_predsN = predict_and_plot(X_testN, test_targets, 'Testing')


# In[ ]:


## SVM


# In[338]:


from sklearn.svm import SVC
classifier = SVC(kernel  = 'rbf')
classifier.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# In[339]:


X_trainS = train_inputs[numeric_cols + encoded_cols]
X_valS   = val_inputs[numeric_cols + encoded_cols]
X_testS  = test_inputs[numeric_cols + encoded_cols]


# In[340]:


train_predS = model.predict(X_trainS)
train_predS


# In[341]:


train_probsS = model.predict_proba(X_trainS)
train_probsS


# In[342]:


accuracy_score(train_targets, train_predS)


# In[343]:


confusion_matrix(train_targets, train_predS, normalize = 'true')


# In[344]:


train_predsS = predict_and_plot(X_trainS, train_targets, 'Training')


# In[345]:


val_predsS = predict_and_plot(X_valS, val_targets, 'Validation')


# In[346]:


test_predsS = predict_and_plot(X_testS, test_targets, 'Testing')


# ## KN Classifier

# In[282]:


from sklearn.neighbors import KNeighborsClassifier


# In[283]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# In[285]:


X_trainK = train_inputs[numeric_cols + encoded_cols]
X_valK   = val_inputs[numeric_cols + encoded_cols]
X_testK  = test_inputs[numeric_cols + encoded_cols]


# In[286]:


train_predK = model.predict(X_trainK)
train_predK


# In[287]:


train_probsK = model.predict_proba(X_trainK)
train_probsK


# In[288]:


accuracy_score(train_targets, train_predK)


# In[289]:


confusion_matrix(train_targets, train_predK, normalize = 'true')


# In[290]:


train_predsK = predict_and_plot(X_trainK, train_targets, 'Training')


# In[291]:


val_predsK = predict_and_plot(X_valK, val_targets, 'Validation')


# In[292]:


test_predsK = predict_and_plot(X_testK, test_targets, 'Testing')


# # Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[302]:


knnclassifier = KNeighborsClassifier(n_neighbors=10)
print(cross_val_score(knnclassifier, train_inputs[numeric_cols + encoded_cols], train_targets, cv=3, scoring ='accuracy').mean())


# In[335]:


logreg= LogisticRegression(solver = "lbfgs", max_iter = 1000)
print(cross_val_score(logreg, train_inputs[numeric_cols + encoded_cols], train_targets, cv=10, scoring ='accuracy').mean())


# In[336]:


GB = GaussianNB()
print(cross_val_score(GB, train_inputs[numeric_cols + encoded_cols], train_targets, cv=10, scoring ='accuracy').mean())


# In[337]:


classifier = SVC(kernel  = 'rbf')
print(cross_val_score(classifier, train_inputs[numeric_cols + encoded_cols], train_targets, cv=10, scoring ='accuracy').mean())


# In[ ]:




