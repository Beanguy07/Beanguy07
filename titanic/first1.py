import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data['Survived']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=125, max_depth=5,random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions })
output.to_csv('/Users/mac/Downloads/titanic/py/result.csv', index= False)

print('done') 