class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_matrix(self, weights, gradients):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] -= (
                    self.learning_rate
                    * gradients[i][j]
                )

    def update_vector(self, values, gradients):
        for i in range(len(values)):
            values[i] -= (
                self.learning_rate
                * gradients[i]
            )
