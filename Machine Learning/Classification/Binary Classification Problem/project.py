#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss 

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data exploration

# In[2]:


df = pd.read_csv('Credit_card.csv')
df_values = pd.read_csv('Credit_card_label.csv')


# In[3]:


# Exploring first 5 rows of the data

df.head() 


# In[4]:


df_values.head()


# In[5]:


data_1 = df_values.label.value_counts()
data_1


# Let's visualize the ratio of approvals vs declines

# In[91]:


fig, ax = plt.subplots()
explode = (0, 0.15)

names = ['Approves', 'Declines']

ax.pie(data_1, 
       labels=[f'{names[i]}: {data_2.index[i]}' for i in range(len(data_2))],
       explode = explode, 
       #autopct='%1.1f%%', 
       shadow=True, 
       startangle=90, 
       autopct='%1.1f%%',
       # labels = [ 'Approves', 'Declines']
      )

plt.title('Approvals vs Declines')
plt.show()


# In this dataset the target valuables represent application approval with 0 and application rejection with 1. In order for the data to be more intuitive we will reverse these so approval will be represented as 1 and rejection as 0. 

# Our data base is very imbalanced, we have a lot more approval rate than declines, with approximate ratio 1 to 8. 

# In[10]:


# Combining our features with values

df = pd.merge(df, df_values, on = 'Ind_ID')


# In[11]:


df.info()


# In[12]:


df.describe().T


# In[13]:


# Checking unique values in all columns

for col in df.columns:
    print(col)
    
    print(df[col].unique())
    print()


# # Data preparation

# Shuffling the ordering of the rows in the database is an essential step in preparing your data for machine learning training. Let's do that. 

# In[14]:


np.random.seed(1)
df = df.loc[np.random.permutation(len(df))]


# Upon the inspection, we can see that column 'Mobile_phone' contains only one value of 1, thus we can safely remove it as will not affect our data. 

# In[15]:


df.drop(columns = ['Mobile_phone', 'Ind_ID'], axis =1, inplace=True)


# In[16]:


# Checking for duplicate rows 

dups = df.duplicated()
df[dups]


# In[17]:


# There are 162 dublicates which we can remove

df = df.drop_duplicates().reset_index(drop=True)


# In[18]:


# Replacing column names with lowercase

df.rename(columns=lambda x: x.lower(), inplace=True) 


# In[19]:


# Verifying results

df.info()


# In[20]:


# Converting birthday count days into years

df['birthday_count'] = abs(df['birthday_count']/365).round()


# In[21]:


df['birthday_count'].unique()


# In[22]:


# Creating a new column for unemployment

df['unemployed'] = df['employed_days'] >=0

# Converting booleans into integers

df['unemployed'] = np.multiply(df['unemployed'], 1) 


# In[23]:


df['unemployed'].value_counts() # verifying results


# In[24]:


# Creating a new column for employment

df['employed'] = df['employed_days'] < 0 

# Converting booleans into integers

df['employed'] = np.multiply(df['employed'], 1)
df['employed'].value_counts() # verifying results


# In[25]:


# Removing positive values indicating unemployment from 'employed_days' column

df['employed_days'] = df['employed_days'].apply(lambda x: x if x < 0 else None)


# In[26]:


df['employed_days'].value_counts()


# In[27]:


# Converting 'employment_days' into 'employment_years'

df['employed_years'] = abs(df['employed_days']/365).round(1)


# In[28]:


# Removing 'employed_days' column as its data has been converted to 'employed', 'unemployed'
# and 'employed_years' columns

df.drop(['employed_days'], axis =1, inplace=True)


# # Handling Missing Values with Imputation

# In[29]:


df.isnull().sum()


# Let's visualize our missing data

# In[30]:


import seaborn as sns
sns.heatmap(df.isnull(), cbar=False)


# Missing values in 'employed_years' column suggest that the person is currently unemployed, hence we can substitute NaN values with 0

# In[31]:


# Replacing missing values with 0

df['employed_years'] = df['employed_years'].fillna(0)
df['employed_years'].isnull().sum() # Verifyingh results


# In[32]:


# Replacing missing values in non-numeric column 'type_occupation' with 'unknown'

df['type_occupation'] = df['type_occupation'].fillna('unknown')


# In[33]:


df.type_occupation.value_counts()


# For numeric columns, it's very common to replace missing values with the mean. 
# In our case missing values in 'annual_income' and 'birthday_count' columns represent less than 2% of the total data, thus this approach is justifiable.  

# In[34]:


birthday_mean = df['birthday_count'].mean()
df['birthday_count'] = df['birthday_count'].fillna(value = birthday_mean)
df['birthday_count'].isnull().sum() # verifying results


# In[35]:


income_mean = df['annual_income'].mean()
df['annual_income'] = df['annual_income'].fillna(value = income_mean)
df['annual_income'].isnull().sum() # verifying results


# Lastly there is less than 1% of total data is missing in gender column, therefor dropping these rows will not significantly impact our data base. 

# In[36]:


df = df.dropna().reset_index(drop=True)
df.info() # verifying index reset


# In[37]:


# Verifying results

df.isnull().sum()


# # Working with outliers

# In[38]:


import warnings
warnings.filterwarnings("ignore")
sns.histplot(df.annual_income, bins = 50)


# It is common practice to apply logarithmic function to data with long tail distribution. It can help make the distribution more symmetric, potentially stabilizing the variance and making it more suitable for certain types of analysis or modeling.

# In[39]:


df['log_income'] = np.log1p(df.annual_income)
sns.histplot(df.log_income, bins = 50)


# In[40]:


df.drop(['annual_income'], axis = 1, inplace =True)
df.dtypes


# # Working with categorical data

# In[41]:


df.dtypes


# In[42]:


df['propert_owner'].value_counts()


# In[43]:


# Converting categorical 'propert_owner' values into numerical. 1 stands for Yes and 0 for No

df['propert_owner'] = (df['propert_owner'] == 'Y').astype(int)
df['propert_owner'].value_counts() # verifying results


# In[44]:


df['car_owner'].value_counts()


# In[45]:


# Converting categorical 'car_owner' values into numerical. 1 stands for Yes and 0 for No


df['car_owner'] = (df['car_owner'] == 'Y').astype(int)
df['car_owner'].value_counts()


# In[46]:


categorical = df.select_dtypes(include='object').columns.tolist()
categorical


# In[47]:


numerical = df.select_dtypes(exclude='object').columns.tolist()
numerical.remove('label')
numerical


# ### Feature importance
# 
# Feature importance is crucial for understanding the relevance of different features in a predictive model and for feature selection.

# In[48]:


# Calculating mean value of approval rate
global_mean = df.label.mean()
round(global_mean, 3) * 100


# In[49]:


df.label.value_counts()


# In[50]:


df.label.value_counts(normalize=True)


# Let's work out the differences between mean of target variable and mean of categorical features. In addition we will calculate the risk ratio as well. This should help us in identifying the most important categorical features for the model.  
# 
# **diff**: Difference between mean of the target variable and mean of categories for a feature. If this ratio is positive, it indicates a lower probability of approval within the category, and if the gap is negative, the group is more likely to be approved. Larger disparities serve as indications that a variable holds more significance compared to others.
# 
# **risk**: Ratio between mean of categories for a feature and mean of the target variable. If this ratio exceeds 1, the category has a higher likelihood of approval, whereas if it falls below 1, the category is less probable to be approved. This ratio provides a comparative measure of the significance of the feature.
# 

# In[51]:


for col in categorical:
    df_group = df.groupby(by=col).label.agg(['mean'])
    df_group['diff'] = df_group['mean'] - global_mean
    df_group['risk'] = df_group['mean'] / global_mean
    display(round(df_group, 3))


# Upon examining the results we can see that there are no substantial disparities in our **diff** column meaning these features do not hold large significances.
# 

# ### Mutual information
# 
# Mutual information is a measure of the amount of information that one random variable contains about another random variable. A higher mutual information value indicates a higher dependency between the two variables, whereas a lower value suggests less dependency.
# 
# Here we will calculate dependency between our credit approval data and categorical features. 

# In[52]:


def calculate_mi(series):
    return mutual_info_score(series, df.label)

df_mi = df[categorical].apply(calculate_mi)
df_mi = df_mi.to_frame(name='MI').sort_values(by='MI', ascending=False)

display(df_mi)


# The actual score suggests a low dependency between categorical features and approval results. However, we can see which features in comparison to each other are more important for our analysis and can be used in the model. 

# # Working with numerical features

# Pearson correlation coefficient is primarily used for evaluating the linear relationship between two numerical (continuous) features. It measures the strength and direction of the linear association between two variables. 
# 
# Correlation criteria can be used as a model-agnostic feature importance method by calculating the correlation coefficient between each feature and the target variable, and selecting features with the highest correlation. 

# In[53]:


df[numerical].corrwith(df.label).to_frame('correlation').sort_values(by='correlation')


# The coefficient results above show low correlation between target variable and our numerical features.

# In[54]:


df.groupby(by='label')[numerical].mean().T


# # One-Hot encoding
# 
# One-hot encoding is a technique used to convert categorical data into a numerical format that can be used by machine learning algorithms.
# 
# The process of one-hot encoding involves creating binary columns for each category, where each column represents one category and is either 1 (if the category is present) or 0 (if the category is not present). This approach prevents the model from assuming any ordinal relationship between the categories.
# 
# The model also keep numerical values unchanged.

# For this purpose we will use DictVectorizer. In the scikit-learn library it is used to convert a dictionary or an array of dictionaries into a NumPy array. It encodes categorical (string or numerical) features as a one-hot numeric array, which can then be used as input for machine learning models.
# 
# First we need to convert our data frame in multiple dictionaries, following that we will fit DictVectorizer.

# In[56]:


X_dict = df[categorical + numerical].to_dict(orient='records')


# In[57]:


X_dict[0]


# In[58]:


dv = DictVectorizer(sparse=False)

# When sparse is set to False, the resulting array will be a standard 2D NumPy array.
# If sparse is set to True, the resulting array will be a sparse matrix, 
# which is more memory-efficient for large datasets with many categorical features.

dv.fit(X_dict)


# In[59]:


df_encod = dv.transform(X_dict)
df_encod.shape


# In[63]:


dv.get_feature_names_out().tolist()


# # Setting up validation framework

# In[65]:


X_train, X_test, y_train, y_test = train_test_split(df_encod, df.label, test_size=0.2, random_state=1)


# In[92]:


print('Total length of data frame', len(df))
print('Training set', len(X_train))
print('Test set', len(X_test))
print('Sum of train and test sets', len(X_train)+len(X_test))
print('Target variables', len(y_train))


# # Training logistic regression

# In[70]:


model = LogisticRegression(solver='liblinear', random_state=10)
model.fit(X_train, y_train)


# In[71]:


from sklearn.metrics import confusion_matrix, classification_report 
predictions = model.predict(X_test) 
  
print('Classification Report')
print('')
print(classification_report(y_test, predictions)) 


# The imbalanced dataset might lead to misleading precision and recall results. We should take steps to solve this problem.

# ### Working with Imbalanced Data

# SMOTE (Synthetic Minority Over-sampling Technique) is a common method used to tackle imbalanced datasets. Its main job is to balance the number of samples in different classes by creating more copies of the minority class. It does this by  synthesising new examples that are combinations of the existing ones. This helps to ensure that the model doesn't become biased towards the majority class and can learn from the minority class as well.

# In[72]:


sm = SMOTE(random_state = 10) 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel()) 


# In[73]:


lr1 = LogisticRegression() 
lr1.fit(X_train_res, y_train_res.ravel()) 
predictions = lr1.predict(X_test) 

print('Report after oversampling')
print('')
print(classification_report(y_test, predictions)) 


# In[74]:


print("Number of '1' labels before oversampling: {}".format(sum(y_train == 1))) 
print("Number of '0' labels before oversampling: {} \n".format(sum(y_train == 0))) 
  
print("Number of '1' labels after oversampling: {}".format(sum(y_train_res == 1))) 
print("Number of '0' labels after oversampling: {}".format(sum(y_train_res == 0))) 


# In[75]:


nr = NearMiss() 
  
X_train_miss, y_train_miss = nr.fit_resample(X_train, y_train.ravel()) 

print("Number of '1' labels before undersampling: {}".format(sum(y_train == 1))) 
print("Number of '0' labels before undersampling: {} \n".format(sum(y_train == 0))) 

print("Number of '1' labels after undersampling: {}".format(sum(y_train_miss == 1))) 
print("Number of '0' labels after undersampling: {}".format(sum(y_train_miss == 0))) 


# In[76]:


lr2 = LogisticRegression() 
lr2.fit(X_train_miss, y_train_miss.ravel()) 
predictions = lr2.predict(X_test) 
  
print('Report after undersampling')
print('')
print(classification_report(y_test, predictions)) 


# While oversampling produced better results than undersampling, we should consider training alternative models to further evaluate the results.

# # Training tree based models

# In[77]:


# Let's initialize the models

dtc = DecisionTreeClassifier(random_state=10)
rfc = RandomForestClassifier(random_state=10)
xgb = XGBClassifier(random_state=10, use_label_encoder=False, eval_metric='logloss')


# Creating a for loop to train multiple models

# In[78]:


results = {}

models = [('Decision Tree', dtc), ('Random Forest', rfc),('XGBoost', xgb)]


for name, clf in models:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred)

    results[name] = [accuracy, precision, recall, f1 , roc_auc_test]


# In[79]:


results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC Score']).T
results_df


# # Selecting the model

# XGBoost outperformes other models in terms of combination of accuracy, precision, and recall when applied to the dataset.
# 
# In addition XGBoost offers clear indications of the importance of various features in prediction.

# Let's visualize the model using confusion matrix and classification report.
# 
# Confusion matrix is a table used in machine learning to describe the performance of a classification model. It helps visualize the performance of an algorithm by displaying the counts of true and false classifications, specifically for binary classification problems. 

# In[81]:


y_pred_xgb = xgb.predict(X_test)

cm_xgb = confusion_matrix(y_test, y_pred_xgb)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=xgb.classes_)
plt.figure(figsize=(6,6))
disp.plot(cmap='Blues', values_format='.0f')
plt.title("Confusion Matrix for XGBoost")
plt.grid(False)
plt.show()
print("\n")

print("Classification Report for XGBoost:\n")

print(classification_report(y_test, y_pred_xgb))


# # Tuning model's parameters

# Grid search is a hyperparameter tuning technique used to find the best combination of model hyperparameters that yields the optimal performance for a machine learning model. It works by exhaustively searching through a manually specified subset of the hyperparameter space of a learning algorithm.

# In[82]:


from sklearn.model_selection import GridSearchCV


# In[83]:


param_grid = {
   'learning_rate': [0.01, 0.05, 0.1],
   'max_depth': [3, 4, 5, 6],
   'n_estimators': [50, 100, 150, 200],
   'subsample': [0.8, 0.9, 1],
   'colsample_bytree': [0.8, 0.9, 1]



clf_xgb = XGBClassifier(random_state=10, use_label_encoder=False, eval_metric='logloss')

grid_search = GridSearchCV(clf_xgb, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)


# In[95]:


xgb_best = XGBClassifier(
    colsample_bytree=0.9,
    learning_rate=0.1,
    max_depth=4,
    n_estimators=150,
    subsample=0.9,
    random_state=10,
    use_label_encoder=False,
    eval_metric='logloss'
)

best_model = xgb_best.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("Classification Report for the Model")
print(classification_report(y_test, y_pred_best))


# # Using the model

# In[99]:


# Generating random customer to test the model

customer = {
"gender": "female",
 "car_owner": "Y",
 "propert_owner": "Y",
 "children": 0,
 "type_income": "Commercial associate",
 "education":"Higher education",
 "marital_status": "Married",
 "housing_type": "House / apartment",
 "birthday_count": -13557,
 "work_phone": 0,
 "phone": 1,
 "email_id": 1,
 "type_occupation": "Managers",
 "family_members": 2,
 "employed_days": -2418,
 "Annual_income": 95850
}


# We will display the probability of negative class, which is credit card approval in our case

# In[101]:


test = dv.transform([customer])
best_model.predict_proba(test)[0, 0] 
# using the notation [0, 0] to retrieve the probability of the negative class


# # Saving and loading the model

# In[102]:


import pickle


# In[107]:


with open('project.bin', 'wb') as f_out:
    pickle.dump((dv, best_model), f_out)


# In[108]:


with open('project.bin', 'rb') as f_in:
    dv, best_model = pickle.load(f_in)


# In[109]:


dv, best_model

