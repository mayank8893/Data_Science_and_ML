#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction
# 
# In this project, I am looking at **the heart disease dataset from kaggle**. This dataset has information on a person age, sex, cholestrol and other heart health metrics from EKG etc. The dataset also has a target variable which which tells if the person has heart issues (1) or if they have a healthy heart (0). **The aim of this project is to build a classification model that tells whether a person has healthy heart or not**.
# 
# I will first load the dataset, get some statistical info on it and clean it. I will then split it into a training and a test set. Then I will train a **Logistic Regression model on the test set and use it to make prediction about the test set**. I have also trained a bunch of other model, but there was no noticeable improvement in accuracy. The model works with an **accuracy of 80% on the test set**. Finally, I have made a **predictive system** where you can input the metrics for your heart and the model will predict if you have a good heart.
# 
# I have then saved this model and deployed it to webapp using **streamlit**.

# In[145]:


# importing the dependencies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ### Data Collection and Processing

# In[146]:


# loading the data file.
heart = pd.read_csv("heart_disease_data.csv")
heart.head()


# In[147]:


# getting the shape of data.
heart.shape


# In[148]:


# getting info of the data.
heart.info()


# In[149]:


# checking for null values
heart.isnull().sum()


# In[150]:


# getting statistical measures of the data.
heart.describe()


# In[151]:


# how many have heart disease vs not.
heart['target'].value_counts()


# In[ ]:





# In[ ]:





# ### Feature Extraction

# In[152]:


X  = heart.drop(columns='target', axis = 1)
y = heart['target']


# In[153]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=1)


# In[ ]:





# In[154]:


print(X.shape, X_train.shape, X_test.shape)


# ### Model training and prediction

# ### Logistic Regression model

# In[155]:


from sklearn.linear_model import LogisticRegression

m1 = 'Logistic Regression'
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)

lr_acc_score = accuracy_score(y_test, lr_predict)
print("Accuracy of Logistic Regression:", lr_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, lr_predict))

# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, lr_predict)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### Gaussian NB model

# In[156]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train, y_train)
nbpred = nb.predict(X_test)

nb_acc_score = accuracy_score(y_test, nbpred)
print("Accuracy of Naive Bayes model:", nb_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, nbpred))

# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, nbpred)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### Random Forest Classifier.

# In[157]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

m3 = 'Random Forest Classifier'
rf = RandomForestClassifier(n_estimators=20, random_state=12, max_depth=5)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)
rf_acc_score = accuracy_score(y_test, rf_predicted)

print("Accuracy of Random Forest:", rf_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, rf_predicted))
# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, rf_predicted)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### XG Boost Classifier

# In[158]:


from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

m4 = 'Extreme Gradient Boost'
xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15, gamma=0.6, subsample=0.52,
                    colsample_bytree=0.6, seed=27, reg_lambda=2, booster='dart',
                    colsample_bylevel=0.6, colsample_bynode=0.5)

xgb.fit(X_train, y_train)
xgb_predicted = xgb.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)

print("Accuracy of Extreme Gradient Boost:", xgb_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, xgb_predicted))
# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, xgb_predicted)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### KNN Classifier.

# In[159]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

m5 = 'K-Neighbors Classifier'
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
knn_acc_score = accuracy_score(y_test, knn_predicted)

print("Accuracy of K-Neighbors Classifier:", knn_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, knn_predicted))
# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, knn_predicted)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### Decision Tree Classifier

# In[160]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
dt.fit(X_train, y_train)

dt_predicted = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)

print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))
# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, dt_predicted)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### SVC Classifier

# In[161]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)

svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)

print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))
# Plotting the confusion matrix
confusion_matrix = confusion_matrix(y_test, dt_predicted)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")


# ### Most important features learned by xgb

# In[162]:


imp_feature = pd.DataFrame({'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'], 'Importance': xgb.feature_importances_})
plt.figure(figsize=(10,4))
plt.title("barplot Represent feature importance ")
plt.xlabel("importance ")
plt.ylabel("features")
plt.barh(imp_feature['Feature'],imp_feature['Importance'])
plt.show()


# ### Evaluating model performace

# In[163]:


model_ev = pd.DataFrame({'Model': ['Logistic Regression','Naive Bayes','Random Forest','Extreme Gradient Boost',
                    'K-Nearest Neighbour','Decision Tree','Support Vector Machine'], 'Accuracy': [lr_acc_score*100,
                    nb_acc_score*100,rf_acc_score*100,xgb_acc_score*100,knn_acc_score*100,dt_acc_score*100,svc_acc_score*100]})
model_ev


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Building a predictive system

# In[164]:


input_data = (60,1,0,117,230,1,1,160,1,1.4,2,2,3,) #(56,0,1,140,294,0,0,153,0,1.3,1,0,2)

# change the input data to numpy array.
input_data = np.asarray(input_data)

# making prediction for this particular input.
input_data_reshaped = input_data.reshape(1,-1)

prediction = lr.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("You dont have a Heart Disease.")
else:
    print("You have a Heart Disease.")


# In[ ]:





# ### Saving the trained model

# In[165]:


import pickle


# In[166]:


filename = 'trained_model.sav'
pickle.dump(lr, open(filename, 'wb'))


# In[167]:


# loading the saved model.
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[168]:


input_data = (60,1,0,117,230,1,1,160,1,1.4,2,2,3,) #(56,0,1,140,294,0,0,153,0,1.3,1,0,2)

# change the input data to numpy array.
input_data = np.asarray(input_data)

# making prediction for this particular input.
input_data_reshaped = input_data.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("You dont have a Heart Disease.")
else:
    print("You have a Heart Disease.")


# In[ ]:




