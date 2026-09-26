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
        learning_rate=0.01
    ):
        self.decoder = decoder
        self.loss_function = loss_function
        self.embedding = embedding

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

    def train_step(
        self,
        vectors,
        target_id,
        token_ids
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
        # Attention output
        # -------------------------

        attention_output = (
            self.decoder.causal_attention(
                vectors
            )
        )

        attention_vector = (
            attention_output[-1]
        )

        # -------------------------
        # Feed-forward backprop
        # -------------------------

        feed_forward_gradients = (
            self.decoder_backprop.feed_forward_gradient(
                attention_vector=attention_vector,
                decoder_gradient=decoder_input_gradient,
                decoder=self.decoder
            )
        )

        # -------------------------
        # Gradient to attention
        # -------------------------

        attention_gradient = [
            decoder_input_gradient[i]
            + feed_forward_gradients["input"][i]
            for i in range(
                len(decoder_input_gradient)
            )
        ]

        # -------------------------
        # Attention backprop
        # -------------------------

        embedding_gradients = (
            self.attention_backprop.calculate_input_gradient(
                embeddings=vectors,
                output_gradient=attention_gradient,
                embedding_size=self.decoder.embedding_size
            )
        )

        # -------------------------
        # Embedding gradients
        # -------------------------

        embedding_weight_gradients = (
            self.embedding_backprop.update_embedding_gradient(
                embedding_weights=self.embedding.weights,
                token_ids=token_ids,
                input_gradients=embedding_gradients
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
        # Update embeddings
        # -------------------------

        self.optimizer.update_matrix(
            self.embedding.weights,
            embedding_weight_gradients
        )

        # -------------------------
        # Verification
        # -------------------------

        print(
            "Embedding gradient positions:",
            len(embedding_gradients)
        )

        print(
            "Embedding vocabulary size:",
            len(self.embedding.weights)
        )

        print(
            "Attention gradient positions:",
            len(embedding_gradients)
        )

        print(
            "Attention gradient size:",
            len(embedding_gradients[0])
            if embedding_gradients
            else 0
        )

        return loss
