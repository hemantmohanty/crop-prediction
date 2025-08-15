import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('crop_recommendation.csv')

data.shape
data.isnull().sum()
x = data.iloc[:,:-1]#Feature
y = data.iloc[:,-1]#Lable
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# X_train.head()
# y_train.head()
model = RandomForestClassifier()
model.fit(X_train,y_train)
# predictions = model.predict(X_test)
# accuracy = model.score(X_test,y_test)
# accuracy = model.score(X_test,y_test)
# print("Accuracy: ",accuracy)

pickle.dump(model, open('model.pkl', 'wb'))

