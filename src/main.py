import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

test_data_path = '/workspaces/kaggle-house-prices/src/data/test.csv'
train_data_path = '/workspaces/kaggle-house-prices/src/data/train.csv'

test_data = pd.read_csv(test_data_path)
train_data = pd.read_csv(train_data_path)

features = ['LotArea', 'YearBuilt']

y = train_data.SalePrice
X = train_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

if __name__ == '__main__':
    prediction = model.predict(val_X)
    print(mean_absolute_error(val_y, prediction))
