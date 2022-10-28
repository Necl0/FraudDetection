import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle

data = pd.read_csv(r"C:\Users\josmo\Downloads\creditcard.csv")
target = data.pop('Class')

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_columnwise = ColumnTransformer([], remainder=scaler)
tree_reg = DecisionTreeRegressor()
pipeline = make_pipeline(scaler_columnwise, tree_reg)

x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

pipeline.fit(x_train, y_train)

# Testing
pred = pipeline.predict(x_test)


# Cross Validation
scores = cross_val_score(tree_reg, x_test, y_test, scoring="neg_mean_squared_error", cv=10)

# Display Cross Validation results
def display_scores(scores) -> None:
    print(f"Scores: {scores}\nMean: {scores.mean()}\nStandard Deviation: {scores.std()}")


display_scores(scores)

filename = 'model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))