from numpy import maximum
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mean_absolute_error(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    predicted_values = model.predict(val_X)
    return mean_absolute_error(val_y, predicted_values)

if __name__ == '__main__':
    test_data_path = '/workspaces/kaggle-house-prices/src/data/test.csv'
    train_data_path = '/workspaces/kaggle-house-prices/src/data/train.csv'

    test_data = pd.read_csv(test_data_path)
    train_data = pd.read_csv(train_data_path)

    features = ['LotArea', 'YearBuilt']

    y = train_data.SalePrice
    X = train_data[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    for max_leaf_node in [5, 50, 500, 5000]:
        mae = get_mean_absolute_error(max_leaf_node, train_X, val_X, train_y, val_y)
        print(f"Max leaf nodes: {max_leaf_node} \t\t Mean Absolute Error: {mae}")
