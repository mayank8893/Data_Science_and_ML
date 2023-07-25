#!/usr/bin/env python
# coding: utf-8

# # Loan Default analysis
# 
# For this project, I want to look at the credit risk dataset and train a Machine learning (ML) model to predict whether the loan is going to default or not. The credit risk dataset has information about the age, income, homeownership, employment length, loan intent and 6 more variables along with actual information about whether the loan defaulted as Y/N. **I want to train a classifier that can predict whether a loan is going to default given a set of parameters.** First, I will train a Decision Tree classifier and then improve on it with a Random Forest classifier to study on a training set of data and make a prediction on a test set.
# 
# I will go through all the usual data analysis steps of **loading and cleaning the data, exploratory data analysis via outlier detection and correlation matrix, encode the labels for ML, split the data into training and test set, do a grid search to find optimal hyperparameters of the model and finally measure the performance of the ML model.**

# ### Loading and cleaning the data.

# In[118]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[119]:


df = pd.read_csv('credit_risk_dataset.csv')
df.head()


# In[120]:


df.tail()


# In[121]:


df.describe().T


# **Note: Std. deviation of income is almost the same as the mean.**

# In[122]:


df.shape


# **Checking for null and na values.**

# In[123]:


# check for na values.
df.isna().sum()


# In[124]:


# check for na values.
df.isnull().sum()


# Employment length and loan interest rate columns have null/na values. **Filling these with the mean value of the columns.**

# In[125]:


df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace = True)
df['person_emp_length'].fillna(df['person_emp_length'].mean(), inplace = True)


# In[127]:


# check that the changes indeed got propagated and now there are no null values.
df.isnull().sum()


# ### Exploring the data.
# 
# Doing exploratory data analysis, checking for outliers, correlations and visualizing the data.

# In[129]:


# listing all the columns.
columns = df.columns.tolist()
columns


# In[130]:


# Listing out all the non-numerical or object data types.
obj = df.select_dtypes(include='object').columns
obj


# In[132]:


# Plotting count of default and non default as a function of the 'object' variables.
get_ipython().run_line_magic('matplotlib', 'inline')

obj1 = ['person_home_ownership', 'loan_intent', 'loan_grade']
fig, ax = plt.subplots(1,3,figsize = (20,5))
ax = ax.flatten()
for i, var in enumerate (obj1):
    sns.countplot(data = df,x=var,hue = 'cb_person_default_on_file', ax = ax[i])
    ax[i].tick_params(axis = 'x', labelrotation=45)
plt.tight_layout()


# **Note: According to the above visual, loan grade seems to be a very import feature.** Loans with grade A and B almost never default, whereas grade D and C have equal chance to default vs not.

# In[134]:


# Plotting distribution of loan with respect to homeownership, loan intent and loan grade.

fig,ax = plt.subplots(1,3, figsize=(20,5))

for i, var in enumerate (obj1):
    if i < len(ax.flat):
        obj_cont = df[var].value_counts()
        ax.flat[i].pie(obj_cont, labels = obj_cont.index, autopct = "%1.1f%%")
        ax.flat[i].set_title(f'{var} distribution')
fig.tight_layout()
plt.show()


# This visual shows that:
#     1) Most people getting loans rent their home.
#     2) The reason for loan is very evenly distributed between Education, Medical and Venture.
#     3) Most of the loans are either grade A or grade B.

# In[55]:


# Looking for any outliers.
var = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_status',
       'loan_percent_income', 'cb_person_cred_hist_length']
fig,ax = plt.subplots(4,2,figsize = (20,10))
ax = ax.flatten()

for i, var in enumerate(var):
    sns.boxplot(data = df, x = var, ax = ax[i])
fig.tight_layout()    


# In[135]:


# Looking for any highly correlated columns.
sns.heatmap(data = df.corr(), annot = True)
fig.tight_layout()


# Some highly correlated variables are: (person_age, credit_history_length), (loan_amount, loan_percent_income) both of which make sense logically. The older a person is the more likely that they have a longer credit history and the lower the loan_percent_income is the more likely that they get approved for bigger loan amounts.

# ### Figuring out the important variables using chi-square test.
# 
# As an alternative to training ML model, here I am using Chi squared test to figure out some important features that are "good predictors". Later on, we can compare these features with what ML models tell us.

# In[149]:


#import the necessary libraries for Chi_square test
from scipy.stats import chi2_contingency
import stat


# In[137]:


#Select the categorical variables for chi_square test
obj=df.select_dtypes(include='object').columns
obj


# In[138]:


#Create the loop for categorical value for chi_square test
for i in obj:
    print(i + ":")
    plt.figure(figsize=(10,5))
    sns.countplot(data=df,hue="loan_status",x=i)
    plt.show()
    a=np.array(pd.crosstab(df.loan_status, df[i]))
    (stats,p,dof,_)=chi2_contingency(a,correction=False)
    if p >0.05:
        print("'{}' is a bad predictor.".format(i))
        print('p_val={}\n'.format(p))
    else:
        print("'{}' is a Good Predictor.".format(i))
        print('p_val={}\n'.format(p))


# **According to the p value homeownership, loan intent and loan grade are good predictors.**

# ### Preprocessing the data to train an ML model.
# 
# I will now preprocess the data to be compatible with the training an ML model. ML models cannot work on categorical data and therefore need to be encoded to some numerical values. In our case, I will assign numbers for each category of categorical data.

# In[140]:


from sklearn.preprocessing import LabelEncoder

for col in df.select_dtypes(include='object').columns:
    print(f'{col}:{df[col].unique()}')


# In[141]:


# encode with numbers.
for col in df.select_dtypes(include='object').columns:
    lable_encode=LabelEncoder()
    lable_encode.fit(df[col].unique())
    df[col]=lable_encode.transform(df[col])
    print(f'{col}: {df[col].unique()}')


# In[144]:


plt.figure(figsize = (16,9))
sns.heatmap(data=df.corr(),annot=True)


# ### Feature extraction and splitting the data.
# 
# The data once encoded needs to be split in X and y, where X is the feature matrix and y is the predictor varibale.

# In[145]:


# splitting the data into X and y.
x = df.drop(['loan_status'],axis=1)
y = df['loan_status']
print(x.shape)
print(y.shape)


# In[146]:


#split the data into training and tet set.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[150]:


#outlier removal.
from scipy import stats

num_col = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income',
     'cb_person_cred_hist_length']

z_score = np.abs(stats.zscore(x_train[num_col]))
threshold = 3
outlier_index = np.where(z_score > threshold)[0]
X_Train = x_train.drop(x_train.index[outlier_index])
Y_Train = y_train.drop(y_train.index[outlier_index])


# In[ ]:





# ### Training a Decision tree clasifier first and evaluating its performance.

# In[151]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import accuracy_score


# **Finding the optimal max_depth hyperparameter using grid search. I am not optimizing other parameters for loss of generality.**

# In[154]:


decison_tree=DecisionTreeClassifier(class_weight='balanced')
pram_grid={
    "max_depth":[3,4,5,6,7,8]
}

grid_search=GridSearchCV(decison_tree,pram_grid,cv=5)
grid_search.fit(X_Train,Y_Train)
print(grid_search.best_params_)


# In[155]:


# Training a decision tree with the optimized hyperparamter.
dtree=DecisionTreeClassifier(max_depth=6,random_state=42)
dtree.fit(X_Train,Y_Train)


# In[169]:


# predicting and checking accuracy, precision score, recall score.
from sklearn.metrics import precision_score,recall_score, confusion_matrix

y_pred=dtree.predict(x_test)

print(f'Accuracy_score: ', round((accuracy_score(y_test,y_pred)),2))
print(f'Precision_score :',round((precision_score(y_test,y_pred,average='micro')),2))
print(f'Recall_score :',round((recall_score(y_test,y_pred,average='micro')),2))


# In[ ]:





# In[168]:


# Plotting the seven most important features according to the decision tree.

im_ft=pd.DataFrame({"Feature_names":X_Train.columns,"Features_importance":dtree.feature_importances_})
fe=im_ft.sort_values(by='Features_importance',ascending=False)
fe1=fe.head(7)
plt.figure(figsize=(15,10))
sns.barplot(data=fe1,x="Features_importance",y="Feature_names")
plt.title("Features_Importance Matrix",fontsize=18)
plt.show()


# **According to the Decision Tree classifier loan_percent_income, loan_grade and homeownership are the three most important features.**

# In[170]:


#plot the confusion matrix.
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(data=cm,annot=True,cmap="Blues",linewidths=0.5,fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show();


# In[108]:


#Finally plotting the ROC curve and calculating the are under the ROC curve.

y_pred_prob = dtree.predict_proba(x_test)[:][:,1]
df_actual_predicted=pd.concat([pd.DataFrame(np.array(y_test),columns=['y_actual']),
                              pd.DataFrame(y_pred_prob,columns=['y_pred_prob'])],axis=1)


df_actual_predicted.index=y_test.index
fqr,tqr,tr=roc_curve(df_actual_predicted['y_actual'],df_actual_predicted['y_pred_prob'])
auc=roc_auc_score(df_actual_predicted['y_actual'],df_actual_predicted['y_pred_prob'])


plt.plot(fqr,tqr,label='AUC =%0.4f' %auc)
plt.plot(fqr,fqr,linestyle="--",color="k")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve",fontsize=18)
plt.legend();


# The accuracy of even a very simple decision tree classifier is 92% which is very good. I will now try to improve on thiss by training a Random Forest classifier.

# ### Random Forest classifier.

# In[171]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
}
grid_search=GridSearchCV(rfc,param_grid,cv=5)
grid_search.fit(X_Train,Y_Train)
print(grid_search.best_params_)


# In[172]:


#train the model using the optimized hyperparameters.
rfc1=RandomForestClassifier(n_estimators=200 ,max_depth=None ,max_features='sqrt')
rfc1.fit(X_Train,Y_Train)


# In[174]:


#prediction and scores.
y_prd=rfc1.predict(x_test)
print(f'Accuracy Score: {round(accuracy_score(y_test,y_prd),2)}')
print(f'Precision_score :{round(precision_score(y_test,y_prd,average="micro"),2)}')
print(f'Recall score :{round(recall_score(y_test,y_prd,average="micro"),2)}')


# In[ ]:





# In[ ]:





# In[176]:


#Most important features according to random forest classifier.

imp_fea=pd.DataFrame({"Feature_name":X_Train.columns,"importance":rfc1.feature_importances_})
imp_fea1=imp_fea.sort_values(by='importance',ascending=False)
imp_fea2=imp_fea1.head(7)
plt.figure(figsize=(15,10))
sns.barplot(data=imp_fea2,x="importance",y="Feature_name")
plt.title('Top-7 Importance Features')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# In[177]:


# The confusion matrix 
cm1=confusion_matrix(y_test,y_prd)
sns.heatmap(data=cm1,annot=True,cmap="Blues",linewidths=0.5,fmt="d");
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')


# In[178]:


#plotting the ROC curve.
y_pred_prob=rfc1.predict_proba(x_test)[:][:,1]
df_actual_predicted=pd.concat([pd.DataFrame(np.array(y_test),columns=['y_actual']),
                              pd.DataFrame(y_pred_prob,columns=['y_pred_prob'])],axis=1)


df_actual_predicted.index=y_test.index
fqr,tqr,tr=roc_curve(df_actual_predicted['y_actual'],df_actual_predicted['y_pred_prob'])
auc=roc_auc_score(df_actual_predicted['y_actual'],df_actual_predicted['y_pred_prob'])


plt.plot(fqr,tqr,label='AUC =%0.4f' %auc)
plt.plot(fqr,fqr,linestyle="--",color="k")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve",fontsize=18)
plt.legend();


# As you can see, the random forest classifier is performing only marginally better than the decision tree classifier.

# In[ ]:




