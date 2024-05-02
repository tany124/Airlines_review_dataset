#!/usr/bin/env python
# coding: utf-8

# In[163]:


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[164]:


df1=pd.read_csv('Airlines Reviews and Rating.csv')


# In[165]:


df1.head()


# In[166]:


df1.shape


# In[167]:


df1.info()


# In[168]:


df1.describe()


# ## Cleaning of data set
# ### Handling missing values

# In[169]:


df1.isnull().sum()


# In[170]:


#Percentage of missing values 
round(df1.isnull().sum()/len(df1.index),2)*100


# In[171]:


#Columns having more than 30% missing values
cols_more_than_equalto_30_percent_missingvalues=(df1.columns[(round(100*(df1.isnull().sum()/len(df1.index)),2))>=30]).tolist()
cols_more_than_equalto_30_percent_missingvalues


# In[172]:


#Although Aircraft type and Inflight Entertainment have more than 30% , we are not removing the variable now as it may be relevant for further analysis
df2=df1.drop(['Wifi & Connectivity'],axis=1)
df2.head()


# In[173]:


#Treating missing values in rows
#Counting number of rows having more than 50% missing values
len(df2[df2.isnull().sum(axis=1)>(len(df2.columns)//2)].index)


# #####  As number of rows having more than 50% missing values is 0 we are not removing any rows
# #### Proceeding to treat missing values for columns
# 

# In[174]:


#Replacing missing values of Aircraft Type to 'Unknown'


# In[175]:


df2['Aircraft Type']=df2['Aircraft Type'].fillna('Unknown')
df2.isnull().sum()


# In[176]:


#Imputing missing value of Feature Inflight Entertainment with mode as it is categorical variable
df2['Inflight Entertainment'].mode()[0]
df2['Inflight Entertainment']=df2['Inflight Entertainment'].fillna(df2['Inflight Entertainment'].mode()[0])
df2['Inflight Entertainment'].isnull().sum()


# In[177]:


df2.isnull().sum()


# In[178]:


#Removing missing value row from country column as number of missing values is very less(1)
df2=df2[~df2['Country'].isnull()]
df2['Country'].isnull().sum()


# In[179]:


#Replacing missing values of Route and Type of travellers to Unknown
df2['Type_of_Travellers']=df2['Type_of_Travellers'].fillna('Unknown')
df2['Route']=df2['Route'].fillna('Unknown')


# In[180]:


df2.isnull().sum()


# In[181]:


#Removing rows for seat type with missing values as they are insignificant in number
df2=df2[~df2['Seat_Types'].isnull()]
df2['Seat_Types'].isnull().sum()


# In[182]:


#Imputing missing values of Seat Comfort,Cabin Staff Service,Ground Service,Food & Beverages with mode
df2['Seat Comfort'].mode()[0]
df2['Seat Comfort']=df2['Seat Comfort'].fillna(df2['Seat Comfort'].mode()[0])
df2['Cabin Staff Service'].mode()[0]
df2['Cabin Staff Service']=df2['Cabin Staff Service'].fillna(df2['Cabin Staff Service'].mode()[0])
df2['Ground Service']=df2['Ground Service'].fillna(df2['Ground Service'].mode()[0])
df2['Food & Beverages']=df2['Food & Beverages'].fillna(df2['Food & Beverages'].mode()[0])
df2.isnull().sum()


# In[183]:


#Removing Date flown column as it is irrelevant for analysis
df3=df2.drop(['Date Flown'],axis=1)
df3.head()


# In[184]:


#Splitting Route to Origin and destination
df3['Origin']=df3['Route'].apply(lambda x:x.split('to')[0])
new = df3['Route'].str.split("to", n=1, expand=True)
df3['Destination']=new[1]
df3.head()


# In[185]:


#Splitting Destination column to Connecting terminal
new1 = df3['Destination'].str.split("via", n=1, expand=True)
df3['Connecting terminal']=new1[1]
df3['Destination']=new1[0]
df3.head()
#Dropping the route column after splitting is done


# In[186]:


#Dropping Route column after splitting
df3.drop(['Route'],axis=1,inplace=True)
df3.head()


# In[187]:


df3['Destination']=df3['Destination'].fillna('Unknown')
df3['Connecting terminal']=df3['Connecting terminal'].fillna('Unknown')
df3.isnull().sum()


# In[188]:


df3.info()


# ### Data Visualization- Categorical Univariate analysis

# In[189]:


#Analysing categorical variables
#Type of travellers
(df3[~(df3['Type_of_Travellers']=="Unknown")]['Type_of_Travellers'].value_counts(normalize=True)*100).plot.barh()
plt.show()


# In[190]:


#Comparison among Seat types
(df3['Seat_Types'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[191]:


#Seat comfort rating
(df3['Seat Comfort'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[192]:


#Cabin Staff Service rating
(df3['Cabin Staff Service'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[193]:


#Ground Service rating
(df3['Ground Service'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[194]:


#Food & Beverages rating
(df3['Food & Beverages'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[195]:


#Inflight Entertainment rating
(df3['Inflight Entertainment'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[196]:


#Value For Money rating
(df3['Value For Money'].value_counts(normalize=True)*100).plot.bar()
plt.show()


# In[197]:


#Recommdations
(df3['Recommended'].value_counts(normalize=True)*100).plot(kind='pie',autopct='%1.0f%%')
plt.show()


# ### Bivariate and Multivariate analysis

# In[198]:


#creating recommdation flag as a separate column that will indicate 1 for yes and 0 for no
df3['Recommendation flag']=np.where(df3.Recommended=='yes',1,0)
df3.head()


# ### Types of travelers vs response rate

# In[199]:


round((df3[~(df3['Type_of_Travellers']=="Unknown")].groupby('Type_of_Travellers')['Recommendation flag'].mean())*100,2).plot.bar()
plt.ylabel('Percentage of passengers with positive response')
plt.show()


# ### Seat type vs Response rate

# In[200]:


round((df3[~(df3['Seat_Types']=="Unknown")].groupby('Seat_Types')['Recommendation flag'].mean())*100,2).plot.barh()
plt.xlabel('Percentage of passengers with positive response')
plt.show()


# ### Seat Type vs Seat comfort vs response

# In[201]:


res=pd.pivot_table(data=df3,index='Seat Comfort',columns='Seat_Types',values='Recommendation flag')
sns.heatmap(res,annot=True,cmap='Greens')
plt.show()


# ### Seat Type vs Cabin Staff service vs response

# In[202]:


res=pd.pivot_table(data=df3,index='Cabin Staff Service',columns='Seat_Types',values='Recommendation flag')
sns.heatmap(res,annot=True,cmap='Greens')
plt.show()


# ### Seat Type vs Ground Service vs response

# In[203]:


res=pd.pivot_table(data=df3,index='Ground Service',columns='Seat_Types',values='Recommendation flag')
sns.heatmap(res,annot=True,cmap='Greens')
plt.show()


# ### Seat Type vs Food & Beverages vs response

# In[204]:


res=pd.pivot_table(data=df3,index='Food & Beverages',columns='Seat_Types',values='Recommendation flag')
sns.heatmap(res,annot=True,cmap='Greens')
plt.show()


# ### Seat Type vs Inflight entertainment vs response

# In[205]:


res=pd.pivot_table(data=df3,index='Inflight Entertainment',columns='Seat_Types',values='Recommendation flag')
sns.heatmap(res,annot=True,cmap='Greens')
plt.show()


# ### Seat Type vs Value for Money vs response

# In[206]:


res=pd.pivot_table(data=df3,index='Value For Money',columns='Seat_Types',values='Recommendation flag')
sns.heatmap(res,annot=True,cmap='Greens')
plt.show()


# #### From the above heat maps we can conclude that factors- Seat comfort,cabin staff service, ground service, food and beverages,value for money are of importance to the passengers in all seat types who have responded 'yes' to recommend airline.

# ### Bar chart comparison of Seat Type vs Flight experiences vs response

# In[207]:


pd.pivot_table(data=df3,index='Seat_Types',columns='Seat Comfort',values='Recommendation flag',aggfunc=sum).plot.bar(figsize=(8,5),rot=45)
plt.xlabel('Seat_Types')
plt.ylabel('response rate')
plt.show()


# In[208]:


pd.pivot_table(data=df3,index='Seat_Types',columns='Value For Money',values='Recommendation flag',aggfunc=sum).plot.bar(figsize=(8,5),rot=45)
plt.xlabel('Cabin Staff Service')
plt.ylabel('response rate')
plt.show()


# In[209]:


pd.pivot_table(data=df3,index='Seat_Types',columns='Food & Beverages',values='Recommendation flag',aggfunc=sum).plot.bar(figsize=(8,5),rot=45)
plt.xlabel('Seat_Types')
plt.ylabel('response rate')
plt.show()


# In[211]:


pd.pivot_table(data=df3,index='Seat_Types',columns='Inflight Entertainment',values='Recommendation flag',aggfunc=sum).plot.bar(figsize=(8,5),rot=45)
plt.xlabel('Seat_Types')
plt.ylabel('response rate')
plt.show()


# In[212]:


pd.pivot_table(data=df3,index='Seat_Types',columns='Ground Service',values='Recommendation flag',aggfunc=sum).plot.bar(figsize=(8,5),rot=45)
plt.xlabel('Seat_Types')
plt.ylabel('response rate')
plt.show()


# In[213]:


pd.pivot_table(data=df3[~(df3['Type_of_Travellers']=="Unknown")],index='Type_of_Travellers',columns='Ground Service',values='Recommendation flag',aggfunc=sum).plot.bar(figsize=(8,5),rot=45)
plt.xlabel('Seat_Types')
plt.ylabel('response rate')
plt.show()


# #### Aircraft type vs Value for money

# In[214]:


#Aircraft type vs Value for money
aircrafts = df3['Aircraft Type'].value_counts()
aircrafts = aircrafts.loc[aircrafts > 10].index
aircrafts = df3.loc[df3['Aircraft Type'].isin(aircrafts)]

ac_value = df3.groupby(aircrafts['Aircraft Type'])['Value For Money'].mean().sort_values(ascending=False)

sns.barplot(x = ac_value.index, y = ac_value, palette='ocean')
plt.xticks(rotation = 90)
plt.title('Aircrafts and their "value for money" relation (5 is the max rating)')
plt.show()


# #### Boeing 777-300ER is the aircraft that has the highest rating for value for money

# #### Seat_Types vs response

# In[215]:


res1=df3.groupby('Seat_Types')['Recommended'].value_counts().unstack(fill_value=0)
res1.plot(kind="bar",colormap="Spectral")
plt.show()


# In[216]:


#Seat Types vs Seat comfort rating
res1=df3.groupby('Seat_Types')['Seat Comfort'].value_counts().unstack(fill_value=0)
res1.plot(kind="bar",colormap="RdYlBu")
plt.show()


# In[217]:


#Seat type vs Cabin Staff service
res1=df3.groupby('Seat_Types')['Cabin Staff Service'].value_counts().unstack(fill_value=0)
res1.plot(kind="bar",colormap="RdYlBu")
plt.show()


# In[218]:


#Seat type vs Ground service
res1=df3.groupby('Seat_Types')['Ground Service'].value_counts().unstack(fill_value=0)
res1.plot(kind="bar",colormap="RdYlBu")
plt.show()


# In[219]:


#Seat type vs Food & Beverages service
res1=df3.groupby('Seat_Types')['Food & Beverages'].value_counts().unstack(fill_value=0)
res1.plot(kind="bar",colormap="RdYlBu")
plt.show()


# In[220]:


#Seat type vs Value For Money service
res1=df3.groupby('Seat_Types')['Value For Money'].value_counts().unstack(fill_value=0)
res1.plot(kind="bar",colormap="RdYlBu")
plt.show()


# In[221]:


for column in ['Aircraft Type', 'Country','Origin', 'Destination']:
    sns.barplot( x= df3[column].value_counts()[:10].index, y = df3[column].value_counts()[:10], palette='autumn')
    plt.title(f'Top {column} in the Data Frame')
    plt.xticks(rotation = 90)
    plt.show()


# ### Conclusion
# #### The factors that are of high importance or priority to the passengers like - Seat comfort,Food Beverages,Ground Service,Value for money,Cabinstaff service have been rated low as feedback by the passengers, resulting in majority of passengers not recommending the airline.
# #### In order to improve the rating these services should be improved or revised.

# ### Building Logistic regression model for predicting flight recommendation

# In[222]:


#import sklearn libraries
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report,recall_score,roc_auc_score,roc_curve,accuracy_score,precision_score,precision_recall_curve,confusion_matrix

#miscellaneous libraries
pd.set_option("display.max_colwidth",200)
pd.set_option("display.max_columns",None)


# In[223]:


#Creating dummy variables for categorical variables
#Categorical variables taken are - Seat_Types,Seat Comfort,Cabin Staff Service,Ground Service,Food & Beverages,
#Inflight Entertainment,Value For Money
df3=pd.get_dummies(df3,columns=['Seat_Types','Seat Comfort','Cabin Staff Service','Ground Service','Food & Beverages','Inflight Entertainment','Value For Money'],drop_first=True)

df3


# In[224]:


#Put all feature variables in X 
X=df3[['Seat_Types_Economy Class',
       'Seat_Types_First Class', 'Seat_Types_Premium Economy',
       'Seat Comfort_2.0', 'Seat Comfort_3.0', 'Seat Comfort_4.0',
       'Seat Comfort_5.0', 'Cabin Staff Service_2.0',
       'Cabin Staff Service_3.0', 'Cabin Staff Service_4.0',
       'Cabin Staff Service_5.0', 'Ground Service_2.0', 'Ground Service_3.0',
       'Ground Service_4.0', 'Ground Service_5.0', 'Food & Beverages_1.0',
       'Food & Beverages_2.0', 'Food & Beverages_3.0', 'Food & Beverages_4.0',
       'Food & Beverages_5.0', 'Inflight Entertainment_1.0',
       'Inflight Entertainment_2.0', 'Inflight Entertainment_3.0',
       'Inflight Entertainment_4.0', 'Inflight Entertainment_5.0',
       'Value For Money_2', 'Value For Money_3', 'Value For Money_4',
       'Value For Money_5']]
X.head()


# In[225]:


#Put target variable in y
y=df3['Recommendation flag']
y


# In[226]:


# Split the dataset into 70% train & 30% test
X_train,X_test,y_train,y_test=train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[227]:


# check the shape
print("X_train Size", X_train.shape)
print("y_train Size", y_train.shape)


# #### Model Building
# 

# In[228]:


#Import the 'LogisticRegression' and create LogisticRegression object
logreg=LogisticRegression()


# In[229]:


# Import the 'RFE' & select 6 variables
# run RFE with 6 variables as output
rfe = RFE(logreg,n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[230]:


# Check the features that are selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[232]:


# Put all the columns selected by RFE in the variable 'col'
col=X_train.columns[rfe.support_]
col


# #### Leveraging the variables selected by Recursive Feature Elimination (RFE), we can now proceed to construct a logistic regression model using statsmodels.

# In[233]:


# Select the columns selected by RFE
X_train = X_train[col]
X_train=X_train.astype('int')


# In[235]:


# Model 1
# Import 'statsmodels'
# Fit a logistic Regression model on X_train after adding a constant and output the summary
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[236]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[237]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Creating dataframe with actual recommendation flag and predicted probabilities

# In[238]:


y_train_pred_final = pd.DataFrame({'Recommendation flag':y_train.values, 'Recommendation_Prob':y_train_pred})
y_train_pred_final['Recommendation_ID'] = y_train.index
y_train_pred_final.head()


# #### Creating new column "Predicted" with 1 if recommendation probability is > 0.5

# In[239]:


y_train_pred_final['predicted'] = y_train_pred_final.Recommendation_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head(20)


# In[240]:


from sklearn import metrics


# In[241]:


#Confusion metrics
confusion = metrics.confusion_matrix(y_train_pred_final['Recommendation flag'], y_train_pred_final['predicted'] )
print(confusion)


# In[242]:


# Predicted          0         1
# Actual
#0                  1340      87
#1                  89        784  


# In[247]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final['Recommendation flag'], y_train_pred_final['predicted']))


# #### Checking VIFs

# In[248]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[249]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### All variables have a good value of VIF. So we need not drop any variables and we can proceed with making predictions using this model only

# In[250]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final['Recommendation flag'], y_train_pred_final.predicted )
confusion


# In[252]:


# Actual/Predicted           0         1
        # 0                1340      87
        # 1                89       784 


# In[253]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final['Recommendation flag'], y_train_pred_final.predicted)


# ### Precision and Recall

# In[254]:


confusion = metrics.confusion_matrix(y_train_pred_final['Recommendation flag'], y_train_pred_final.predicted )
confusion


# #### Precision
# ##### TP/ (TP + FP)

# In[1]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# #### Recall
# ##### TP / TP + FN

# In[256]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# #### Using sklearn utilities for the same

# In[257]:


from sklearn.metrics import precision_score, recall_score


# In[258]:


get_ipython().run_line_magic('pinfo', 'precision_score')


# In[259]:


precision_score(y_train_pred_final['Recommendation flag'], y_train_pred_final.predicted)


# In[260]:


recall_score(y_train_pred_final['Recommendation flag'], y_train_pred_final.predicted)


# #### Precision and recall tradeoff

# In[261]:


from sklearn.metrics import precision_recall_curve


# In[262]:


y_train_pred_final['Recommendation flag'], y_train_pred_final.predicted


# In[263]:


p, r, thresholds = precision_recall_curve(y_train_pred_final['Recommendation flag'], y_train_pred_final['Recommendation_Prob'])


# In[264]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# #### Making predictions on test set

# In[265]:


X_test = X_test[col]
X_test.head()


# In[267]:


X_test_sm = sm.add_constant(X_test)
X_test_sm


# In[280]:


X_test_sm=X_test_sm.astype('int')
y_test_pred=res.predict(X_test_sm)


# In[281]:


y_test_pred[:10]


# In[282]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[283]:


y_pred_1.head()


# In[284]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[285]:


# Putting CustID to index
y_test_df['Recommendation ID'] = y_test_df.index


# In[286]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[287]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[288]:


y_pred_final.head()


# In[289]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Recommendation_Prob'})


# In[292]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Recommendation ID','Recommendation flag','Recommendation_Prob'], axis=1)


# In[293]:


y_pred_final.head()


# In[295]:


y_pred_final['final_predicted'] = y_pred_final['Recommendation_Prob'].map(lambda x: 1 if x > 0.52 else 0)


# In[296]:


y_pred_final.head(20)


# In[297]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final['Recommendation flag'], y_pred_final.final_predicted)


# In[298]:


confusion2 = metrics.confusion_matrix(y_pred_final['Recommendation flag'], y_pred_final.final_predicted )
confusion2


# In[299]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[300]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[301]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:




