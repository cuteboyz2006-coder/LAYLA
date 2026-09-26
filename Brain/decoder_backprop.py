class DecoderBackprop:
    def residual_gradient(
        self,
        decoder_gradient
    ):
        # Residual connection:
        #
        # output = input + feed_forward(input)
        #
        # Is step mein direct residual path
        # ka gradient pass kar rahe hain.

        return [
            value
            for value in decoder_gradient
        ]
