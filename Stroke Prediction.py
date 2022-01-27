#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# In[2]:


stroke_data = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke_data.head(5)


# In[3]:


# Dataframe size
print(stroke_data.shape)


# ### Exploratory Data Analysis

# In[4]:


stroke_data.describe()


# In[5]:


correlation_matrix = stroke_data.corr()
print(correlation_matrix)


# In[6]:


correlation_matrix['stroke'].sort_values(ascending = False)


# In[7]:


plt.figure(figsize=(12,8))
sns.heatmap(data = stroke_data.corr(), annot = True)
plt.show()


# In[8]:


# Gender distribution
print(stroke_data['gender'].value_counts())


# In[9]:


# Target distribution
print(stroke_data['stroke'].value_counts())


# In[10]:


plt.figure(figsize = (12, 8))
sns.countplot(stroke_data['stroke'])
plt.show()


# In[11]:


# Dropping instance with 'other' gender since irrelevant to prediction
stroke_data[stroke_data['gender'] == 'Other']


# ### Feature Extraction

# In[12]:


stroke_data.drop([3116], inplace = True)
stroke_data.drop(['id'], axis = 1, inplace = True)


# In[13]:


stroke_data.head(5)


# In[14]:


# Confirming dropping operation
print(stroke_data['gender'].value_counts())


# In[15]:


# Plot distribution of ages
sns.displot(data = stroke_data, x = 'age', kind = 'kde')


# In[16]:


# Grouping patients with hypertension who developed a stroke
stroke_data.groupby(['hypertension'])['stroke'].sum()


# In[17]:


# Grouping patients with heart disease who developed a stroke
stroke_data.groupby(['heart_disease'])['stroke'].sum()


# In[18]:


# Grouping stroke sufferers based on residential area
stroke_data.groupby(['Residence_type'])['stroke'].sum()


# In[19]:


# Grouping people who smoked, had hypertension and heart disease and suffered a stroke
stroke_data.groupby(['hypertension', 'heart_disease', 'smoking_status'])['stroke'].sum()


# In[20]:


# Obtaining data types
stroke_data.dtypes


# In[21]:


print(f"Different kinds of gender feature is: {stroke_data['gender'].unique()}")
print(f"Different kinds of ever married feature is: {stroke_data['ever_married'].unique()}")
print(f"Different kinds of worktype feature is: {stroke_data['work_type'].unique()}")
print(f"Different kinds of Residence_type feature is: {stroke_data['Residence_type'].unique()}")
print(f"Different kinds of smoking_status feature is: {stroke_data['smoking_status'].unique()}")


# In[22]:


# Encoding Categorical Columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
stroke_data = pd.get_dummies(data = stroke_data, columns = categorical_columns)


# In[23]:


stroke_data.head(20)


# In[24]:


# Dataframe shape after encoding
print(stroke_data.shape)


# In[25]:


# Checking for missing values
stroke_data.isnull().sum()


# In[26]:


# Handling missing values
stroke_data.fillna(method = 'bfill', inplace = True)


# In[27]:


# Confirming handling of missing values
stroke_data.isnull().sum()


# In[28]:


X = stroke_data.drop(['stroke'], axis = 1)
Y = stroke_data['stroke']


# In[29]:


X.head(5)


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)


# In[31]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


# In[32]:


lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear', max_iter = 300, dual = True, random_state = 42)
lsvc = LinearSVC(class_weight = 'balanced', dual = False)
svc = SVC(class_weight = 'balanced', kernel = 'poly')
dtc = DecisionTreeClassifier(max_depth = 8, class_weight = 'balanced')
etc = ExtraTreeClassifier(max_depth = 8, class_weight = 'balanced')
rfc = RandomForestClassifier(n_estimators = 300, max_depth = 8)


# ### Fitting models on imbalanced data

# In[33]:


print(f"fitting logistic regression classiifier : {lr.fit(x_train,y_train)}")
print(f"fitting linear svc classifier : {lsvc.fit(x_train, y_train)}")
print(f"fitting svc kernel classifier : {svc.fit(x_train, y_train)}")
print(f"fitting decision tree classifier : {dtc.fit(x_train, y_train)}")
print(f"fitting extra tree classifier : {etc.fit(x_train, y_train)}")
print(f"fitting random forest classifier : {rfc.fit(x_train, y_train)}")


# In[34]:


y_pred = lr.predict(x_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
cm = confusion_matrix(y_test, y_pred)
print(f"f1_score : {f1_score(y_test, y_pred)}")
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.show()


# In[35]:


y_pred = lsvc.predict(x_test)
print(f"f1_score : {f1_score(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [1, 0])
disp.plot()
plt.show()


# In[36]:


y_pred = svc.predict(x_test)
print(f"f1_score : {f1_score(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [1, 0])
disp.plot()
plt.show()


# In[37]:


y_pred = dtc.predict(x_test)
print(f"f1_score : {f1_score(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [1, 0])
disp.plot()
plt.show()


# In[38]:


y_pred = etc.predict(x_test)
print(f"f1_score : {f1_score(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred, labels = [1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [1, 0])
disp.plot()
plt.show()


# ### Resampling the dataset and fitting

# In[39]:


# pip install -U imbalanced-learn


# In[40]:


x_tr = scaler.fit_transform(X)


# In[41]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
x_tr_adj, y_tr_adj = sm.fit_resample(x_tr, Y)


# In[47]:


total_f1 = []

fold = StratifiedKFold(n_splits = 2)
i = 1
for train_index, test_index in fold.split(x_tr_adj, y_tr_adj):
    X_TRAIN, X_TEST = x_tr_adj[train_index], x_tr_adj[test_index]
    Y_TRAIN, Y_TEST = y_tr_adj[train_index], y_tr_adj[test_index]
    
    lr.fit(X_TRAIN, Y_TRAIN)
    
    Y_PRED = lr.predict(X_TEST)
    
    print(f"--------------------- Fitting Logistic Regression ----------------------")
    
    print(f"fold {i} f1_score : {f1_score(Y_TEST, Y_PRED)}")
    
    i + 1
    
    total_f1.append(f1_score(Y_TEST, Y_PRED))

print(f"-------------------- Mean F1-SCORE ------------------------")
print(np.mean(total_f1))


# In[49]:


total_f1 = []

fold = StratifiedKFold(n_splits = 2)
i = 1
for train_index, test_index in fold.split(x_tr_adj, y_tr_adj):
    X_TRAIN, X_TEST = x_tr_adj[train_index], x_tr_adj[test_index]
    Y_TRAIN, Y_TEST = y_tr_adj[train_index], y_tr_adj[test_index]
    
    lsvc.fit(X_TRAIN, Y_TRAIN)
    
    Y_PRED = lsvc.predict(X_TEST)
    
    print(f"--------------------------- Fitting Linear SVC --------------------------")
    
    print(f"fold {i} f1_score : {f1_score(Y_TEST, Y_PRED)}")
    
    i + 1
    
    total_f1.append(f1_score(Y_TEST, Y_PRED))
    
print(f"-------------------- Mean F1-SCORE ------------------------")
print(np.mean(total_f1))


# In[50]:


total_f1 = []

fold = StratifiedKFold(n_splits = 2)
i = 1
for train_index, test_index in fold.split(x_tr_adj, y_tr_adj):
    X_TRAIN, X_TEST = x_tr_adj[train_index], x_tr_adj[test_index]
    Y_TRAIN, Y_TEST = y_tr_adj[train_index], y_tr_adj[test_index]
    
    svc.fit(X_TRAIN, Y_TRAIN)
    
    Y_PRED = svc.predict(X_TEST)
    
    print(f"-------------------------- Fitting SVC Kernel --------------------------")
    
    print(f"fold {i} f1_score : {f1_score(Y_TEST, Y_PRED)}")
    
    i + 1
    
    total_f1.append(f1_score(Y_TEST, Y_PRED))
    
print(f"-------------------- Mean F1-SCORE ------------------------")
print(np.mean(total_f1))


# In[51]:


total_f1 = []

fold = StratifiedKFold(n_splits = 2)
i = 1
for train_index, test_index in fold.split(x_tr_adj, y_tr_adj):
    X_TRAIN, X_TEST = x_tr_adj[train_index], x_tr_adj[test_index]
    Y_TRAIN, Y_TEST = y_tr_adj[train_index], y_tr_adj[test_index]
    
    dtc.fit(X_TRAIN, Y_TRAIN)
    
    Y_PRED = dtc.predict(X_TEST)
    
    print(f"------------------------ Fitting Decision Tree Classifier ---------------------------")
    
    print(f"fold {i} f1_score : {f1_score(Y_TEST, Y_PRED)}")
    
    i + 1
    
    total_f1.append(f1_score(Y_TEST, Y_PRED))


print(f"-------------------- Mean F1-SCORE ------------------------")
print(np.mean(total_f1))


# In[52]:


total_f1 = []

fold = StratifiedKFold(n_splits = 2)
i = 1
for train_index, test_index in fold.split(x_tr_adj, y_tr_adj):
    X_TRAIN, X_TEST = x_tr_adj[train_index], x_tr_adj[test_index]
    Y_TRAIN, Y_TEST = y_tr_adj[train_index], y_tr_adj[test_index]
    
    etc.fit(X_TRAIN, Y_TRAIN)
    
    Y_PRED = etc.predict(X_TEST)
    
    print(f"------------------------- Fitting Extra Trees Classifier -----------------------")
    
    print(f"fold {i} f1_score : {f1_score(Y_TEST, Y_PRED)}")
    
    i + 1
    
    total_f1.append(f1_score(Y_TEST, Y_PRED))

print(f"-------------------- Mean F1-SCORE ------------------------")
print(np.mean(total_f1))


# In[53]:


total_f1 = []

fold = StratifiedKFold(n_splits = 2)
i = 1
for train_index, test_index in fold.split(x_tr_adj, y_tr_adj):
    X_TRAIN, X_TEST = x_tr_adj[train_index], x_tr_adj[test_index]
    Y_TRAIN, Y_TEST = y_tr_adj[train_index], y_tr_adj[test_index]
    
    rfc.fit(X_TRAIN, Y_TRAIN)
    
    Y_PRED = rfc.predict(X_TEST)
    
    print(f"-------------------------- Fitting Random Forest Classifier ----------------------- ")
    
    print(f"fold {i} f1_score : {f1_score(Y_TEST, Y_PRED)}")
    
    i + 1
    
    total_f1.append(f1_score(Y_TEST, Y_PRED))

print(f"-------------------- Mean F1-SCORE ------------------------")
print(np.mean(total_f1))


# ### In conclusion, Random Forest Classifier produced the best f1-score

# In[ ]:




