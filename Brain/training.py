from Brain.backprop import OutputBackprop
from Brain.optimizer import SGD


class Trainer:
    def __init__(
        self,
        decoder,
        loss_function,
        learning_rate=0.01
    ):
        self.decoder = decoder
        self.loss_function = loss_function

        self.backprop = OutputBackprop()

        self.optimizer = SGD(
            learning_rate=learning_rate
        )

    def train_step(
        self,
        vectors,
        target_id
    ):
        # -------------------------
        # Forward pass
        # -------------------------

        decoder_output = self.decoder.forward(
            vectors
        )

        logits = self.decoder.logits(
            decoder_output
        )

        # -------------------------
        # Loss
        # -------------------------

        loss = self.loss_function.loss(
            logits[-1],
            target_id
        )

        # -------------------------
        # Gradient
        # -------------------------

        output_gradient = (
            self.loss_function.gradient(
                logits[-1],
                target_id
            )
        )

        # -------------------------
        # Backpropagation
        # -------------------------

        gradients = (
            self.backprop.calculate_gradients(
                decoder_output[-1],
                self.decoder.output_weights,
                output_gradient
            )
        )

        # -------------------------
        # Weight update
        # -------------------------

        self.optimizer.update_matrix(
            self.decoder.output_weights,
            gradients["weights"]
        )

        self.optimizer.update_vector(
            self.decoder.output_bias,
            gradients["bias"]
        )

        return loss
