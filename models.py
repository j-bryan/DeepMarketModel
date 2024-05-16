import torch


LAYER_TYPES = {
    "LSTM": torch.nn.LSTM,
    "RNN": torch.nn.RNN,
    "GRU": torch.nn.GRU
}


class SequenceModel(torch.nn.Module):
    """
    A base class for sequence models. This class does not implement the forward method, so it should
    not be used directly. Instead, use one of the subclasses, SequenceEncoder or SequenceDecoder.
    """
    conditional = False  # If the model requires a condition vector

    def __init__(self,
                 layer_type: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 batch_first: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.layer_type = layer_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        Layer = LAYER_TYPES[layer_type]
        self.model = Layer(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           dropout=dropout,
                           bidirectional=bidirectional)


class SequenceEncoder(SequenceModel):
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.model(X)  # Initial hidden state (and cell state for LSTM) defaults to zero
        return output, hidden  # For LSTM, hidden is a tuple of hidden and cell state


class SequenceDecoder(SequenceModel):
    def __init__(self,
                 layer_type: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 batch_first: bool,
                 dropout: float,
                 bidirectional: bool,
                 output_model: torch.nn.Module,
                 feed_previous: bool = True):
        """
        :param layer_type: str: The type of recurrent layer to use (LSTM, RNN, or GRU)
        :param input_size: int: The number of expected features in the input X
        :param hidden_size: int: The number of features in the hidden state h
        :param num_layers: int: The number of recurrent layers
        :param batch_first: bool: If True, the input and output tensors are provided as (batch, seq, feature)
        :param dropout: float: If non-zero, introduces a dropout layer on the outputs of each RNN layer
        :param bidirectional: bool: If True, the RNN is bidirectional
        :param output_model: torch.nn.Module: A model to pass the decoder outputs through
        :param feed_previous: bool: If True, use the previous output of the decoder as input to the next step.
                                    If False, use a zero tensor as the input to the decoder. Must be
                                    True if a target tensor is provided (teacher forcing).
        """
        assert batch_first
        super().__init__(layer_type, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional)
        self.output_model = output_model
        self.feed_previous = feed_previous

    def forward(self,
                encoder_outputs: torch.Tensor,
                encoder_hidden: torch.Tensor | tuple,
                target_tensor: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param encoder_outputs: torch.Tensor: The output of the encoder, shape (batch_size, seq_len, hidden_size)
        :param encoder_hidden: torch.Tensor | tuple: The hidden state of the encoder, shape (num_layers * num_directions, batch_size, hidden_size)
                               If the decoder model is bidirectional, num_directions is 2. If the decoder
                               is an LSTM, encoder_hidden is a tuple of (hidden state, cell state), each
                               with shape (num_layers * num_directions, batch_size, hidden_size).
        :param target_tensor: torch.Tensor | None: The target tensor, shape (batch_size, seq_len, input_size)
                              Used for training only for teacher forcing. If None, the decoder will
                              use its own outputs as inputs.
        """
        use_teacher_forcing = target_tensor is not None
        if use_teacher_forcing and not self.feed_previous:
            raise ValueError("The feed_previous parameter must be True if a target tensor is provided.")

        # Initialize the decoder hidden state with the encoder hidden state
        batch_size = encoder_outputs.size(0)

        decoder_input = torch.zeros(batch_size, 1, 1).to(encoder_outputs.device)  # No outputs available at the first step, so use a zero tensor
        decoder_hidden = encoder_hidden

        # Loop over the sequence length to retrieve the decoder outputs at each time step
        decoder_outputs = []
        seq_len = encoder_outputs.size(1)
        for i in range(seq_len):
            output, decoder_hidden = self.model(decoder_input, decoder_hidden)
            output = self.output_model(output)
            decoder_outputs.append(output)

            # We allow for three different modes of operation for the decoder:
            #   1. (Teacher forcing) Use the target tensor as input to the decoder. This is useful during
            #      training to help the model learn to generate the correct output.
            #   2. (Feed previous) Use the previous output of the decoder as input to the next step. This
            #      is useful during inference when the target tensor is not available.
            #   3. (No feedback) Use a zero tensor as input to the decoder. This is a simple baseline.
            if use_teacher_forcing:
                decoder_input = target_tensor[:, i, :].unsqueeze(1)
            elif self.feed_previous:
                decoder_input = output
            else:
                decoder_input = torch.zeros(batch_size, 1, 1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden


class ContextHiddenDecoder(SequenceDecoder):
    """
    Decodes a sequence with a vector of conditional inputs that are appended to the hidden state
    at the first time step.
    """
    conditional = True  # If the model requires a condition vector

    def __init__(self,
                 layer_type: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 conditional_size: int,
                 batch_first: bool,
                 dropout: float,
                 bidirectional: bool,
                 output_model: torch.nn.Module,
                 feed_previous: bool = True):
        """
        :param layer_type: str: The type of recurrent layer to use (LSTM, RNN, or GRU)
        :param input_size: int: The number of expected features in the input X
        :param hidden_size: int: The number of features in the hidden state h
        :param num_layers: int: The number of recurrent layers
        :param conditional_size: int: The size of the condition vector
        :param batch_first: bool: If True, the input and output tensors are provided as (batch, seq, feature)
        :param dropout: float: If non-zero, introduces a dropout layer on the outputs of each RNN layer
        :param bidirectional: bool: If True, the RNN is bidirectional
        :param output_model: torch.nn.Module: A model to pass the decoder outputs through
        :param feed_previous: bool: If True, use the previous output of the decoder as input to the next step.
                                    If False, use a zero tensor as the input to the decoder. Must be
                                    True if a target tensor is provided (teacher forcing).
        """
        self.conditional_size = conditional_size
        self.orig_input_size = input_size
        self.orig_hidden_size= hidden_size
        super().__init__(layer_type, input_size, hidden_size + conditional_size, num_layers, batch_first, dropout, bidirectional, output_model, feed_previous)

    def forward(self,
                encoder_outputs: torch.Tensor,
                encoder_hidden: torch.Tensor | tuple,
                condition: torch.Tensor,
                target_tensor: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param encoder_outputs: torch.Tensor: The output of the encoder, shape (batch_size, seq_len, hidden_size)
        :param encoder_hidden: torch.Tensor | tuple: The hidden state of the encoder, shape (num_layers * num_directions, batch_size, hidden_size)
                               If the decoder model is bidirectional, num_directions is 2. If the decoder
                               is an LSTM, encoder_hidden is a tuple of (hidden state, cell state), each
                               with shape (num_layers * num_directions, batch_size, hidden_size).
        :param condition: torch.Tensor: The condition to add to the hidden state
        :param target_tensor: torch.Tensor | None: The target tensor, shape (batch_size, seq_len, input_size)
                              Used for training only for teacher forcing. If None, the decoder will
                              use its own outputs as inputs.
        """
        condition = condition.unsqueeze(0).repeat(encoder_hidden.shape[0], 1, 1)
        augmented_hidden = torch.concat([encoder_hidden, condition], dim=2)
        return super().forward(encoder_outputs, augmented_hidden, target_tensor)

class ContextInputDecoder(SequenceDecoder):
    """
    Decodes a sequence using a vector of conditional inputs at each time step in addition to the
    output of the previous time step.
    """
    conditional = True  # If the model requires a condition vector

    def __init__(self,
                 layer_type: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 conditional_size: int,
                 batch_first: bool,
                 dropout: float,
                 bidirectional: bool,
                 output_model: torch.nn.Module,
                 feed_previous: bool = True):
        """
        :param layer_type: str: The type of recurrent layer to use (LSTM, RNN, or GRU)
        :param input_size: int: The number of expected features in the input X
        :param hidden_size: int: The number of features in the hidden state h
        :param num_layers: int: The number of recurrent layers
        :param conditional_size: int: The size of the condition vector
        :param batch_first: bool: If True, the input and output tensors are provided as (batch, seq, feature)
        :param dropout: float: If non-zero, introduces a dropout layer on the outputs of each RNN layer
        :param bidirectional: bool: If True, the RNN is bidirectional
        :param output_model: torch.nn.Module: A model to pass the decoder outputs through
        :param feed_previous: bool: If True, use the previous output of the decoder as input to the next step.
                                    If False, use a zero tensor as the input to the decoder. Must be
                                    True if a target tensor is provided (teacher forcing).
        """
        self.conditional_size = conditional_size
        self.orig_input_size = input_size
        self.orig_hidden_size= hidden_size
        super().__init__(layer_type, input_size + conditional_size, hidden_size, num_layers, batch_first, dropout, bidirectional, output_model, feed_previous)

    def forward(self,
                encoder_outputs: torch.Tensor,
                encoder_hidden: torch.Tensor | tuple,
                condition: torch.Tensor,
                target_tensor: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param encoder_outputs: torch.Tensor: The output of the encoder, shape (batch_size, seq_len, hidden_size)
        :param encoder_hidden: torch.Tensor | tuple: The hidden state of the encoder, shape (num_layers * num_directions, batch_size, hidden_size)
                               If the decoder model is bidirectional, num_directions is 2. If the decoder
                               is an LSTM, encoder_hidden is a tuple of (hidden state, cell state), each
                               with shape (num_layers * num_directions, batch_size, hidden_size).
        :param condition: torch.Tensor: The condition to add to the hidden state
        :param target_tensor: torch.Tensor | None: The target tensor, shape (batch_size, seq_len, input_size)
                              Used for training only for teacher forcing. If None, the decoder will
                              use its own outputs as inputs.
        """
        use_teacher_forcing = target_tensor is not None
        if use_teacher_forcing and not self.feed_previous:
            raise ValueError("The feed_previous parameter must be True if a target tensor is provided.")

        # Initialize the decoder hidden state with the encoder hidden state
        batch_size = encoder_outputs.size(0)
        condition = condition.unsqueeze(1)
        decoder_input = torch.concat([condition, torch.zeros(batch_size, 1, 1).to(encoder_outputs.device)], dim=2)
        decoder_hidden = encoder_hidden

        # Loop over the sequence length to retrieve the decoder outputs at each time step
        decoder_outputs = []
        seq_len = encoder_outputs.size(1)
        for i in range(seq_len):
            output, decoder_hidden = self.model(decoder_input, decoder_hidden)
            output = self.output_model(output)
            decoder_outputs.append(output)

            # We allow for three different modes of operation for the decoder:
            #   1. (Teacher forcing) Use the target tensor as input to the decoder. This is useful during
            #      training to help the model learn to generate the correct output.
            #   2. (Feed previous) Use the previous output of the decoder as input to the next step. This
            #      is useful during inference when the target tensor is not available.
            #   3. (No feedback) Use a zero tensor as input to the decoder. This is a simple baseline.
            if use_teacher_forcing:
                decoder_input = target_tensor[:, i, :].unsqueeze(1)
            elif self.feed_previous:
                decoder_input = output
            else:
                decoder_input = torch.zeros(batch_size, 1, 1)
            decoder_input = torch.concat([condition, decoder_input], dim=2)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden


class Seq2Seq(torch.nn.Module):
    """
    An encoder-decoder sequence-to-sequence model.
    """
    def __init__(self, encoder: SequenceEncoder, decoder: SequenceDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X: torch.Tensor, target_tensor: torch.Tensor | None = None) -> torch.Tensor:
        encoded, hidden = self.encoder(X)  # For LSTM, hidden is a tuple of hidden and cell state
        decoded, _ = self.decoder(encoded, hidden, target_tensor)  # Ignore the second output, which is the hidden state (and cell state if LSTM)
        return decoded


class ContextSeq2Seq(torch.nn.Module):
    """
    Sequence-to-sequence encoder-decoder model with an additional conditional input that is added
    to the hidden state transfer between the encoder and decoder.
    """
    def __init__(self, encoder: SequenceEncoder, decoder: SequenceDecoder, encoder_context: bool, decoder_context: bool, context_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_context = encoder_context
        self.decoder_context = decoder_context
        self.context_size = context_size

    def forward(self, X: torch.Tensor, target_tensor: torch.Tensor | None = None) -> torch.Tensor:
        if self.encoder_context or self.decoder_context:
            context = X[:, 0, -self.context_size:]
        if not self.encoder_context and self.context_size > 0:
            X = X[:, :, :-self.context_size]  # Remove the context from the input tensor

        # Pass through encoder
        encoded, hidden = self.encoder(X)

        if self.decoder_context:
            decoded, _ = self.decoder(encoded, hidden, context, target_tensor)
        else:
            decoded, _ = self.decoder(encoded, hidden, target_tensor)

        return decoded


class MLP(torch.nn.Module):
    """
    A simple multi-layer perceptron model.
    """
    def __init__(self,
                 input_size: int,
                 hidden_layers: list[int],
                 output_size: int,
                 activation: str = "ReLU",
                 output_activation: str = "Identity"):
        super().__init__()
        if not len(hidden_layers) > 0:
            raise ValueError(f"At least one hidden layer is required. Received {hidden_layers}. "
                             "Otherwise, just use torch.nn.Linear directly.")

        Activation = getattr(torch.nn, activation)
        OutputActivation = getattr(torch.nn, output_activation)

        layers = [torch.nn.Linear(input_size, hidden_layers[0]), Activation()]
        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(Activation())
        layers.append(torch.nn.Linear(hidden_layers[-1], output_size))
        layers.append(OutputActivation())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
