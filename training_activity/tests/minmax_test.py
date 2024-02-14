from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class min_max_scaler:
    def __init__(self):
        self.maximums = []
        self.minimums = []
    

    def fit(self, features):
        transpose = features.T
        height = transpose.shape[0]
        
        for index in range(height):
            temp = transpose[index]
            self.maximums.append(np.amax(temp))
            self.minimums.append(np.amin(temp))


    def transform(self, data):
        transpose = data.T
        height, width = transpose.shape

        scaled_matrix = np.zeros((height, width), np.float64)
        for i in range(height):
            for j in range(width):
                scaled_matrix[i][j] = (transpose[i][j] - self.minimums[i]) / (self.maximums[i] - self.minimums[i])

        return scaled_matrix.T


if __name__ == "__main__":
    df_train = pd.read_csv("../features/train.csv")

    train_y = np.array(df_train.filter(["256"], axis=1))
    train_x = np.array(df_train.drop(["dataset", "weather", "date", "hour", "parking_space", "256"], axis=1))

    scaler = min_max_scaler();
    sklearn_scaler = MinMaxScaler();

    scaler.fit(train_x)
    sklearn_scaler.fit(train_x)

    my_scaler_result = scaler.transform(train_x)
    sklearn_scaler_result = sklearn_scaler.transform(train_x)

    print("--------------------------------------------------------------------")
    print("My min max scaler: ")
    print(my_scaler_result)
    print("--------------------------------------------------------------------")
    print("Sklearn min max scaler: ")
    print(sklearn_scaler_result)
    print("--------------------------------------------------------------------")