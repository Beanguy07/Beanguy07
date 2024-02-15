import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data['SalePrice']

X = pd.get_dummies(train_data)
X_test = pd.get_dummies(test_data)


model = RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=20)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame[{'Id': test_data.Id, 'SalePrice': predictions }]
output.to_csv('/Users/mac/downloads/py/result.csv', index=False)

print('done')
