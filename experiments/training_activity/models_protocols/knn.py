import numpy as np
import heapq

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


    def predict(self, features):
        preds = []

        for feature in features:
            distances = []
            for index, x in enumerate(self.x_data):
                heapq.heappush(distances, (self.__euclidian_dist(x, feature), index))
            
            k_nearest = heapq.nsmallest(self.k, distances)
            classes = np.array([self.y_data[nearest[1]] for nearest in k_nearest])

            ones_qtd = np.count_nonzero(classes)
            zeros_qtd = len(classes) - ones_qtd
            preds.append(1 if ones_qtd >= zeros_qtd else 0)

        return preds