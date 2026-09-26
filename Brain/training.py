from Brain.backprop import OutputBackprop
from Brain.optimizer import SGD
from Brain.embedding_backprop import (
    DecoderInputBackprop,
    EmbeddingBackprop
)
from Brain.decoder_backprop import DecoderBackprop
from Brain.attention_backprop import AttentionBackprop


class Trainer:

    def __init__(
        self,
        decoder,
        loss_function,
        embedding,
        learning_rate=0.01,
        max_gradient=1.0
    ):
        self.decoder = decoder
        self.loss_function = loss_function
        self.embedding = embedding

        self.max_gradient = max_gradient

        self.backprop = OutputBackprop()

        self.input_backprop = (
            DecoderInputBackprop()
        )

        self.embedding_backprop = (
            EmbeddingBackprop()
        )

        self.decoder_backprop = (
            DecoderBackprop()
        )

        self.attention_backprop = (
            AttentionBackprop()
        )

        self.optimizer = SGD(
            learning_rate=learning_rate
        )

    # --------------------------------
    # Gradient clipping
    # --------------------------------

    def clip_value(self, value):
        if value > self.max_gradient:
            return self.max_gradient

        if value < -self.max_gradient:
            return -self.max_gradient

        return value

    def clip_vector(self, vector):
        return [
            self.clip_value(value)
            for value in vector
        ]

    def clip_matrix(self, matrix):
        return [
            [
                self.clip_value(value)
                for value in row
            ]
            for row in matrix
        ]

    # --------------------------------
    # Train one token prediction
    # --------------------------------

    def train_step(
        self,
        vectors,
        target_id,
        token_ids
    ):

        if not vectors:
            return 0.0

        # --------------------------------
        # Forward pass
        # --------------------------------

        decoder_output = self.decoder.forward(
            vectors
        )

        if not decoder_output:
            return 0.0

        logits = self.decoder.logits(
            decoder_output
        )

        if not logits:
            return 0.0

        last_logits = logits[-1]

        # --------------------------------
        # Loss
        # --------------------------------

        loss = self.loss_function.loss(
            last_logits,
            target_id
        )

        # --------------------------------
        # Output gradient
        # --------------------------------

        output_gradient = (
            self.loss_function.gradient(
                last_logits,
                target_id
            )
        )

        output_gradient = (
            self.clip_vector(
                output_gradient
            )
        )

        # --------------------------------
        # Output layer backprop
        # --------------------------------

        output_gradients = (
            self.backprop.calculate_gradients(
                decoder_output[-1],
                self.decoder.output_weights,
                output_gradient
            )
        )

        output_weight_gradients = (
            self.clip_matrix(
                output_gradients["weights"]
            )
        )

        output_bias_gradients = (
            self.clip_vector(
                output_gradients["bias"]
            )
        )

        # --------------------------------
        # Gradient into decoder
        # --------------------------------

        decoder_input_gradient = (
            self.input_backprop.calculate_gradient(
                self.decoder.output_weights,
                output_gradient
            )
        )

        decoder_input_gradient = (
            self.clip_vector(
                decoder_input_gradient
            )
        )

        # --------------------------------
        # Recalculate causal attention
        # --------------------------------

        attention_output = (
            self.decoder.causal_attention(
                vectors
            )
        )

        if not attention_output:
            return loss

        attention_vector = (
            attention_output[-1]
        )

        # --------------------------------
        # Feed-forward backprop
        # --------------------------------

        feed_forward_gradients = (
            self.decoder_backprop.feed_forward_gradient(
                attention_vector=attention_vector,
                decoder_gradient=decoder_input_gradient,
                decoder=self.decoder
            )
        )

        # --------------------------------
        # Residual connection
        #
        # decoder_output =
        # attention + feed_forward
        # --------------------------------

        attention_gradient = [
            decoder_input_gradient[i]
            + feed_forward_gradients["input"][i]
            for i in range(
                len(decoder_input_gradient)
            )
        ]

        attention_gradient = (
            self.clip_vector(
                attention_gradient
            )
        )

        # --------------------------------
        # Attention backprop
        # --------------------------------

        embedding_gradients = (
            self.attention_backprop.calculate_input_gradient(
                embeddings=vectors,
                output_gradient=attention_gradient,
                embedding_size=self.decoder.embedding_size
            )
        )

        embedding_gradients = [
            self.clip_vector(
                gradient
            )
            for gradient in embedding_gradients
        ]

        # --------------------------------
        # Embedding weight gradients
        # --------------------------------

        embedding_weight_gradients = (
            self.embedding_backprop.update_embedding_gradient(
                embedding_weights=self.embedding.weights,
                token_ids=token_ids,
                input_gradients=embedding_gradients
            )
        )

        embedding_weight_gradients = (
            self.clip_matrix(
                embedding_weight_gradients
            )
        )

        # --------------------------------
        # Feed-forward gradients
        # --------------------------------

        w1_gradients = (
            self.clip_matrix(
                feed_forward_gradients["w1"]
            )
        )

        b1_gradients = (
            self.clip_vector(
                feed_forward_gradients["b1"]
            )
        )

        w2_gradients = (
            self.clip_matrix(
                feed_forward_gradients["w2"]
            )
        )

        b2_gradients = (
            self.clip_vector(
                feed_forward_gradients["b2"]
            )
        )

        # --------------------------------
        # Apply updates
        # --------------------------------

        self.optimizer.update_matrix(
            self.decoder.output_weights,
            output_weight_gradients
        )

        self.optimizer.update_vector(
            self.decoder.output_bias,
            output_bias_gradients
        )

        self.optimizer.update_matrix(
            self.decoder.w1,
            w1_gradients
        )

        self.optimizer.update_vector(
            self.decoder.b1,
            b1_gradients
        )

        self.optimizer.update_matrix(
            self.decoder.w2,
            w2_gradients
        )

        self.optimizer.update_vector(
            self.decoder.b2,
            b2_gradients
        )

        self.optimizer.update_matrix(
            self.embedding.weights,
            embedding_weight_gradients
        )

        return loss
