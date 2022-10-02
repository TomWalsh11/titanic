import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df.head()

sns.heatmap(data=df.isnull())
sns.set_style('darkgrid')
plt.figure(figsize=(10,5))
sns.boxplot(x='Pclass',y='Age',data=df)

df[df['Pclass']==1]['Age'].mean()

def enter_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return df[df['Pclass']==1]['Age'].mean()
        elif Pclass == 2:
            return df[df['Pclass']==2]['Age'].mean()
        else:
            return df[df['Pclass']==3]['Age'].mean()
    else:
        return Age

df['Age'] = df[['Age','Pclass']].apply(enter_age,axis=1)

df.head()
df

sns.heatmap(df.isnull())

df.drop('Cabin',axis=1,inplace=True)
df

sex = pd.get_dummies(df['Sex'],drop_first=True)
sex

embarked = pd.get_dummies(df['Embarked'],drop_first=True)
embarked

pclass = pd.get_dummies(df['Pclass'],drop_first=True)
pclass

df = pd.concat([df,pclass,sex,embarked],axis=1)
df.head()

df.drop(['Pclass','Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
df

dft = pd.read_csv('test.csv')
dft.head()

def enter_aget(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return dft[dft['Pclass']==1]['Age'].mean()
        elif Pclass == 2:
            return dft[dft['Pclass']==2]['Age'].mean()
        else:
            return dft[dft['Pclass']==3]['Age'].mean()
    else:
        return Age

dft['Age'] = dft[['Age','Pclass']].apply(enter_aget,axis=1)
dft.head()

sex = pd.get_dummies(dft['Sex'],drop_first=True)
embarked = pd.get_dummies(dft['Embarked'],drop_first=True)
pclass = pd.get_dummies(dft['Pclass'],drop_first=True)

dft = pd.concat([dft,sex,pclass,embarked],axis=1)
dft.head()

dft.drop(['Pclass','Name','Sex','Embarked','Ticket','Cabin'],inplace=True,axis=1)
dft.head()

X_train = df.drop(['PassengerId','Survived'],axis=1)
y_train = df['Survived']
X_test = dft.drop('PassengerId',axis=1)

# Logistic Model
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=5000)

logmodel.fit(X_train,y_train)

X_test
X_test.isna().sum()
X_test['Fare'][152] = X_test['Fare'].mean()
X_test.isna().sum()
predictions = logmodel.predict(X_test)
predictions

dft

output = pd.DataFrame({'PassengerId':dft['PassengerId'], 'Survived':predictions})

output

output.to_csv('submission.csv',index=False)


# Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

rfc_output = pd.DataFrame({'PassengerId':dft['PassengerId'],'Survived':rfc_pred})

rfc_output.to_csv('RfcSubmission.csv',index=False)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)

doutput = pd.DataFrame({'PassengerId':dft['PassengerId'],'Survived':pred})
doutput.to_csv('DTreeSubmission.csv',index=False)

df = pd.read_csv('train.csv')
df.head()

def name_iter(x):
    if 'Mr.' in x:
        return 1
    elif 'Ms.' in x:
        return 2
    elif 'Master' in x:
        return 3
    else:
        return x

df['Name'] = df['Name'].apply(name_iter)
df.head()