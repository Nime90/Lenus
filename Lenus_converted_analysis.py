#importing relevant lybraries
import pandas as pd
import matplotlib.pyplot as plt
from logistic_reg import Logistic_ML, Logistic_stat
import seaborn as sns
pd.set_option('display.max_rows', 500) #in order to better look at results

#importing data
data=pd.read_csv('customer_data_sample.csv')

#converting gender and branch into a numerical value:
female=[]
for g in data.gender:
    if g=='female': female.append(1)
    else: female.append(0)
data['female']=female

branch_n=[]
for b in data.branch:
    if b=='Helsinki': branch_n.append(0)
    elif b=='Tampere': branch_n.append(1)
    elif b=='Turku': branch_n.append(2)
    else: branch_n.append(None)

data['branch_n']=branch_n

#Cleaning data form the "suspicious ages"
data_clean=data#[data['age']>13]
#dropping duplicates
data_clean=data_clean.drop_duplicates()
#dropping empty cells
data_clean=data_clean.dropna()
#rounding the age into an integer
age_r=[int(a) for a in data_clean.age]
data_clean['age_r']=age_r
#cleaning the index
data_clean.reset_index(inplace=True, drop=True)

#Creating a dummy variable for credit_account_id
credit_account_id_d=[]
for c in data_clean['credit_account_id']:
    if c=='9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0':
        credit_account_id_d.append(1)
    else:
        credit_account_id_d.append(0)
data_clean['credit_account_id_dummy']=credit_account_id_d

#show correlation graphs
sns.set_style("whitegrid")
sns.pairplot(data_clean)

# Graphing correlation between columns
sns.heatmap(data_clean[['converted','female','initial_fee_level','customer_segment','credit_account_id_dummy' ]].corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.rcParams['figure.figsize'] = [20, 7]
plt.show()

#Preparing data for logistic regression
#make the dataset ready to the analysis
df=data_clean[['customer_id','converted','female','age_r','initial_fee_level','credit_account_id_dummy']]
variables=['customer_segment','related_customers','family_size','branch']
for v in variables:
    temp=pd.get_dummies(data_clean[v], prefix=v)
    for c in temp.columns:
        df[c]=temp[c]

#running Regression using ML model
Logistic_ML(df)
Logistic_stat(df)

#select the colums you want to use for your regression
df3=df[[ 'converted', 'female', 
         'related_customers_0',#'family_size_1','age_r',#  
         'customer_segment_11',#'credit_account_id_dummy',#'initial_fee_level',#'credit_account_id_dummy',
        'branch_Helsinki']]

#running Regression using ML model
Logistic_ML(df3)
Logistic_stat(df3)