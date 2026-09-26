from Brain.feed_forward_backprop import FeedForwardBackprop


class DecoderBackprop:
    def __init__(self):
        self.feed_forward_backprop = FeedForwardBackprop()

    def residual_gradient(
        self,
        decoder_gradient
    ):
        # Residual connection:
        #
        # output = attention + feed_forward(attention)
        #
        # Direct residual path
        # gradient = decoder gradient

        return [
            value
            for value in decoder_gradient
        ]

    def feed_forward_gradient(
        self,
        attention_vector,
        decoder_gradient,
        decoder
    ):
        # Recalculate the hidden layer
        # used by the feed-forward network.

        hidden = decoder._linear(
            attention_vector,
            decoder.w1,
            decoder.b1
        )

        hidden = decoder._relu(
            hidden
        )

        gradients = (
            self.feed_forward_backprop.calculate(
                input_vector=attention_vector,
                hidden_vector=hidden,
                output_gradient=decoder_gradient,
                w1=decoder.w1,
                w2=decoder.w2
            )
        )

        return gradients
