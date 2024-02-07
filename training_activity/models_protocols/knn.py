import numpy as np

class KNearestNeighbors:
    def __init__(self):
        self.k = 3
    

    def __euclidian_dist(self, point_a, point_b):
        temp = point_a - point_b
        temp = np.dot(temp.T, temp)
        return np.sqrt(temp)


    def fit(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


    def predict(self, x_pred):
        preds = []

        point1 = self.x_data[0]
        point2 = x_pred[0]

        print(self.__euclidian_dist(point1, point2))

        return preds