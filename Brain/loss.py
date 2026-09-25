import math


class CrossEntropyLoss:
    def softmax(self, logits):
        if not logits:
            return []

        maximum = max(logits)

        exp_values = [
            math.exp(value - maximum)
            for value in logits
        ]

        total = sum(exp_values)

        if total == 0:
            return [0.0] * len(logits)

        return [
            value / total
            for value in exp_values
        ]

    def loss(self, logits, target_id):
        if not logits:
            raise ValueError("Logits cannot be empty")

        if target_id < 0 or target_id >= len(logits):
            raise ValueError("Invalid target token ID")

        probabilities = self.softmax(logits)

        probability = probabilities[target_id]

        # Prevent log(0)
        probability = max(
            probability,
            1e-12
        )

        return -math.log(probability)

    def gradient(self, logits, target_id):
        if not logits:
            raise ValueError("Logits cannot be empty")

        if target_id < 0 or target_id >= len(logits):
            raise ValueError("Invalid target token ID")

        probabilities = self.softmax(logits)

        gradients = probabilities.copy()

        gradients[target_id] -= 1.0

        return gradients
