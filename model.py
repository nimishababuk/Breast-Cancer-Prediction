import pandas as pd

data = pd.read_csv('breast-cancer.csv')
print(data)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['diagnosis'] = encoder.fit_transform(data['diagnosis'])
data.corr()['diagnosis']
data = data.drop(['symmetry_se'],axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

column_names = ['id','radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

plt.figure(figsize=(15,30))
i = 1
for col in column_names:
    plt.subplots_adjust(wspace=0.6,hspace=0.6)
    plt.subplot(5,6,i)
    i+=1
    plt.boxplot(data[col])
    plt.title(col)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scaled_data = scale.fit_transform(data)
pd.DataFrame(scaled_data,columns=data.columns)

x = data.drop('diagnosis',axis=1)
y = data['diagnosis']

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state=5)

from xgboost import XGBClassifier

xg_model = XGBClassifier()
xg_model.fit(xtrain,ytrain)

xg_model.predict(xtest)

score = xg_model.score(xtest,ytest)
print(score)

import pickle
pickle.dump(xg_model,open('model.pkl','wb'))




