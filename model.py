from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class HousingModel:
    def __init__(self):
        self.model = LinearRegression()

    def train_model(self, df):
        features = ["median_income", "housing_median_age", "rooms_per_household", "bedrooms_per_room"]
        X = df[features]
        y = df["median_house_value"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, df=None):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        return mse, rmse