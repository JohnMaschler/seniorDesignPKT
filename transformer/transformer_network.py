import torch
import torch.nn as nn

# ======================
# CustomEmbedding Class
# ======================
# This class is a simple embedding layer for mapping input indices into dense vectors
# of a fixed size (embedding dimension).
class CustomEmbedding(nn.Module):
    # input_dim is the total number of unique indices the embedding layer should expect
    # embed_dim is the size of each embedding vector (output dimension), which represents how dense each input index is encoded
    def __init__(self, input_dim, embed_dim):
        # Calls the constructor of the parent class nn.Module. 
        # This is necessary to initialize internal PyTorch-specific functionality, such as registering the parameters of the module.
        super(CustomEmbedding, self).__init__()
        
        """
        Initializes a PyTorch nn.Embedding layer. This layer creates a lookup table where each input index (integer) maps to a dense vector of size embed_dim.
        For example, if input_dim = 1000 (1000 unique tokens) and embed_dim = 256, 
        the embedding layer will create a matrix of size [1000, 256], 
        where each row corresponds to a vector representation of a token.
        """
        self.embedding = nn.Embedding(input_dim, embed_dim)

    def forward(self, x):
        # x: Input tensor of shape (batch_size, seq_length)
        return self.embedding(x)  # Returns embedded tensor of shape (batch_size, seq_length, embed_dim)
        """
        How It Works:
        Each integer in x is treated as an index to the embedding matrix created in __init__.
        The layer retrieves the corresponding dense vector for each index, resulting in an output tensor of shape (batch_size, seq_length, embed_dim).
        For example:
        If x = [[1, 3, 2], [0, 4, 5]] with batch_size = 2 and seq_length = 3, and embed_dim = 256, the output will be a tensor of shape [2, 3, 256].
        """

# ======================
# SelfAttention Class
# ======================
# This class implements the multi-head self-attention mechanism.
# It allows the model to focus on different parts of the input sequence simultaneously.
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure embedding size is divisible by the number of heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Linear transformations for query, key, and value
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        """
        values, keys, query: Input tensors of shape (batch_size, seq_length, embed_size)
        mask: Optional mask tensor to prevent attending to certain positions
        """
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Transform and reshape for multi-head attention
        values = self.values(values).reshape(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).reshape(N, query_len, self.heads, self.head_dim)

        # Compute attention scores (energy)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # Shape: (N, heads, query_len, key_len)

        # Apply mask (if provided)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Compute attention weights and apply softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Combine attention weights with values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)  # Final output after linear transformation

# ======================
# TransformerBlock Class
# ======================
# This class represents a single block of the transformer model,
# consisting of a self-attention layer followed by a feed-forward network.
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        value, key, query: Input tensors of shape (batch_size, seq_length, embed_size)
        mask: Mask tensor to control attention
        """
        # Self-attention + skip connection + layer normalization
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        # Feed-forward network + skip connection + layer normalization
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# ======================
# Encoder Class
# ======================
# This class encodes the input sequence using a stack of transformer blocks.
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Embedding layers
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout, forward_expansion
                ) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: Input tensor of shape (batch_size, seq_length)
        mask: Mask tensor for attention
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# ======================
# DecoderBlock Class
# ======================
# This class represents a single block of the transformer decoder,
# which includes self-attention, encoder-decoder attention, and feed-forward layers.
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        x: Target sequence
        value, key: Encoder outputs
        src_mask, trg_mask: Attention masks
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

# ======================
# Decoder Class
# ======================
# This class decodes the target sequence using a stack of decoder blocks.
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)  # Predict token probabilities
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        x: Target sequence tokens
        enc_out: Encoder output
        src_mask, trg_mask: Masks for attention
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)  # Output token probabilities
        return out


# ======================
# NetworkTrafficTransformer Class
# ======================
# This class combines the encoder and decoder to form the complete transformer model.
class NetworkTrafficTransformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, num_layers, heads, device, max_length, src_pad_idx, trg_pad_idx):
        super(NetworkTrafficTransformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size, embed_size, num_layers, heads, device, forward_expansion=4, dropout=0.1, max_length=max_length
        )
        self.decoder = Decoder(
            trg_vocab_size, embed_size, num_layers, heads, forward_expansion=4, dropout=0.1, device=device, max_length=max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Create a mask for the source sequence to ignore padding tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # Create a triangular mask for the target sequence to prevent attending to future tokens
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg, mode="train"):
        """
        src: Source sequence
        trg: Target sequence
        mode: "train" or "infer"
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)

        if mode == "train":
            out = self.decoder(trg, enc_src, src_mask, trg_mask)
        elif mode == "infer":
            # Autoregressive decoding
            N = trg.shape[0]  # Batch size
            max_len = trg.shape[1]  # Maximum sequence length
            inputs = trg[:, 0].unsqueeze(1)  # Start decoding with <SOS> token
            outputs = inputs.clone()  # Initialize outputs with <SOS>

            for _ in range(max_len - 1):  # Decode iteratively
                out = self.decoder(inputs, enc_src, src_mask, self.make_trg_mask(inputs))
                next_token = out[:, -1, :].argmax(dim=1, keepdim=True)  # Predict next token
                inputs = torch.cat([inputs, next_token], dim=1)  # Append prediction to inputs
                outputs = torch.cat([outputs, next_token], dim=1)  # Append to outputs

            out = outputs  # Return the generated sequence

        return out
