import pandas as pd 

df = pd.read_csv('train.csv')

df['Sex'] = df['Sex'].replace(['male','female'],[0,5])
df['Embarked'] = df['Embarked'].replace(['S','C','Q'],[0,1,2])
mean_Pclass=round(df['Pclass'].mean())
df['Pclass'].fillna(value=mean_Pclass, inplace=True)

mean_Age=round(df['Age'].mean())
df['Age'].fillna(value=mean_Age, inplace=True)

mean_Sex=round(df['Sex'].mean())
df['Sex'].fillna(value=mean_Sex, inplace=True)

mean_SibSp=round(df['SibSp'].mean())
df['SibSp'].fillna(value=mean_SibSp, inplace=True)

mean_Parch=round(df['Parch'].mean())
df['Parch'].fillna(value=mean_Parch, inplace=True)

mean_Embarked=round(df['Embarked'].mean())
df['Embarked'].fillna(value=mean_Embarked, inplace=True)
# df.dropna(inplace=True)
# print(df)

y_train = df.iloc[:,1]
# print(y_train)

#Extract features usefull:
X_train = df.filter(items=['Pclass','Sex','Age','SibSp','Parch','Embarked'])
print(X_train)

#X_test:
df_test = pd.read_csv('test.csv')
df_test['Sex'] = df_test['Sex'].replace(['male','female'],[0,5])
df_test['Embarked'] = df_test['Embarked'].replace(['S','C','Q'],[0,1,2])
mean_Pclass=round(df_test['Pclass'].mean())
df_test['Pclass'].fillna(value=mean_Pclass, inplace=True)

mean_Age=round(df_test['Age'].mean())
df_test['Age'].fillna(value=mean_Age, inplace=True)

mean_Sex=round(df_test['Sex'].mean())
df_test['Sex'].fillna(value=mean_Sex, inplace=True)

mean_SibSp=round(df_test['SibSp'].mean())
df_test['SibSp'].fillna(value=mean_SibSp, inplace=True)

mean_Parch=round(df_test['Parch'].mean())
df_test['Parch'].fillna(value=mean_Parch, inplace=True)

mean_Embarked=round(df_test['Embarked'].mean())
df_test['Embarked'].fillna(value=mean_Embarked, inplace=True)

X_test = df_test.filter(items=['Pclass','Sex','Age','SibSp','Parch','Embarked'])
print('X_test',X_test.shape)

#Normalize
# import sklearn.preprocessing
# X_train = sklearn.preprocessing.normalize(X_train)
# print(X_train)
import sklearn
from sklearn.svm import SVC

clf = sklearn.svm.SVC(C=6, kernel='rbf')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('y_pred',y_pred)

df_label = pd.read_csv('gender_submission.csv')
y_test = df_label.iloc[:,1]
print('y_test',y_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))