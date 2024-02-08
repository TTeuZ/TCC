from sklearn.metrics import classification_report
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

        print("-----------------------------------------------------------------------------------------")
        print(f"Dados de treino: {self.x_data}")
        print(f"classes de treino: {self.y_data}")
        print("-----------------------------------------------------------------------------------------")

        for feature in features:
            distances = []
            for index, x in enumerate(self.x_data):
                heapq.heappush(distances, (self.__euclidian_dist(x, feature), index))
            
            k_nearest = heapq.nsmallest(self.k, distances)
            classes = np.array([self.y_data[nearest[1]] for nearest in k_nearest])

            ones_qtd = np.count_nonzero(classes)
            zeros_qtd = len(classes) - ones_qtd
            preds.append(1 if ones_qtd >= zeros_qtd else 0)

            print(f"distancias: {distances}")
            print(f"Proximos: {k_nearest}")
            print(f"classes dos mais proximos: {classes}")
            print(f"classificação: {1 if ones_qtd >= zeros_qtd else 0}")
            print("-----------------------------------------------------------------------------------------")

        return preds
    

if __name__ == "__main__":
    x_train = np.array([[1, 1], [2, 2], [3, 1], [2, 0], [0, 4], [2, 5], [2, 1], [1, 5], [3, 3], [2, 4]])
    y_train = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

    x_test = np.array([[3, 4], [2, 3], [4, 1]])
    y_test = np.array([0, 0, 1])

    model = KNearestNeighbors()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print(classification_report(y_test, preds))
