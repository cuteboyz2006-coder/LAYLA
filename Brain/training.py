from Brain.backprop import OutputBackprop
from Brain.optimizer import SGD
from Brain.embedding_backprop import DecoderInputBackprop
from Brain.decoder_backprop import DecoderBackprop


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
        self.input_backprop = DecoderInputBackprop()
        self.decoder_backprop = DecoderBackprop()

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
        # Output gradient
        # -------------------------

        output_gradient = (
            self.loss_function.gradient(
                logits[-1],
                target_id
            )
        )

        # -------------------------
        # Output layer gradients
        # -------------------------

        output_gradients = (
            self.backprop.calculate_gradients(
                decoder_output[-1],
                self.decoder.output_weights,
                output_gradient
            )
        )

        # -------------------------
        # Gradient to decoder input
        # -------------------------

        decoder_input_gradient = (
            self.input_backprop.calculate_gradient(
                self.decoder.output_weights,
                output_gradient
            )
        )

        # -------------------------
        # Current attention vector
        # -------------------------

        attention_output = (
            self.decoder.causal_attention(
                vectors
            )
        )

        attention_vector = attention_output[-1]

        # -------------------------
        # Feed-forward gradients
        # -------------------------

        feed_forward_gradients = (
            self.decoder_backprop.feed_forward_gradient(
                attention_vector=attention_vector,
                decoder_gradient=decoder_input_gradient,
                decoder=self.decoder
            )
        )

        # -------------------------
        # Update output weights
        # -------------------------

        self.optimizer.update_matrix(
            self.decoder.output_weights,
            output_gradients["weights"]
        )

        # -------------------------
        # Update output bias
        # -------------------------

        self.optimizer.update_vector(
            self.decoder.output_bias,
            output_gradients["bias"]
        )

        # -------------------------
        # Update W1
        # -------------------------

        self.optimizer.update_matrix(
            self.decoder.w1,
            feed_forward_gradients["w1"]
        )

        # -------------------------
        # Update B1
        # -------------------------

        self.optimizer.update_vector(
            self.decoder.b1,
            feed_forward_gradients["b1"]
        )

        # -------------------------
        # Update W2
        # -------------------------

        self.optimizer.update_matrix(
            self.decoder.w2,
            feed_forward_gradients["w2"]
        )

        # -------------------------
        # Update B2
        # -------------------------

        self.optimizer.update_vector(
            self.decoder.b2,
            feed_forward_gradients["b2"]
        )

        # -------------------------
        # Gradient verification
        # -------------------------

        print(
            "Decoder input gradient size:",
            len(decoder_input_gradient)
        )

        print(
            "Feed-forward input gradient size:",
            len(
                feed_forward_gradients["input"]
            )
        )

        return loss
