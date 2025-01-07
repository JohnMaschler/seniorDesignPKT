import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


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
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
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
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

with open("transformed_packets.json", "r") as f:
    data = json.load(f)

print("Sample data entry:")
print(data[0])


import re

class TextTokenizer:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size = 4  # starting count

    def build_vocab(self, text_list):
        # Split each text into tokens
        for text in text_list:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def tokenize(self, text):
        # Simple: split on non-alphanumeric
        tokens = re.findall(r"[A-Za-z0-9.]+", text.lower())
        return tokens

    def encode(self, text):
        # Add <SOS> at the start and <EOS> at the end
        tokens = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) 
                  for t in self.tokenize(text)]
        return [self.token_to_id["<SOS>"]] + tokens + [self.token_to_id["<EOS>"]]

    def decode(self, tokens):
        return [self.id_to_token.get(t, "<UNK>") for t in tokens]


# Output packet is a dictionary like:
# {
#   "src_ip": "192.168.202.79",
#   "dst_ip": "192.168.229.254",
#   ...
# }

class PacketTokenizer:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size = 4  # starting count

    def build_vocab(self, packets):
        # Each packet is a dict of fields.
        # We can combine "key:value" into a single token, or treat keys/values separately.
        for packet in packets:
            for key, value in packet.items():
                token = f"{key}:{value}"
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, packet_dict):
        # Convert a dictionary to a list of tokens
        tokens = []
        for key, value in packet_dict.items():
            token = f"{key}:{value}"
            tokens.append(self.token_to_id.get(token, self.token_to_id["<UNK>"]))
        # Optionally add <SOS> and <EOS>
        return [self.token_to_id["<SOS>"]] + tokens + [self.token_to_id["<EOS>"]]

    def decode(self, token_ids):
        return [self.id_to_token.get(t, "<UNK>") for t in token_ids]


# Separate inputs (texts) and outputs (packet dicts)
input_texts = [item["input"] for item in data]
output_packets = [item["output"] for item in data]

text_tokenizer = TextTokenizer()
text_tokenizer.build_vocab(input_texts)

packet_tokenizer = PacketTokenizer()
packet_tokenizer.build_vocab(output_packets)

print("Text vocab size:", text_tokenizer.vocab_size)
print("Packet vocab size:", packet_tokenizer.vocab_size)


class PacketDataset(Dataset):
    def __init__(self, data, text_tokenizer, packet_tokenizer):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.packet_tokenizer = packet_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["input"]
        packet = item["output"]

        # Encode text as source tokens
        src = self.text_tokenizer.encode(text)

        # Encode packet as target tokens
        trg = self.packet_tokenizer.encode(packet)

        return torch.tensor(src), torch.tensor(trg)

def collate_fn(batch):
    # Separate src and trg from the batch
    src_batch = [item[0] for item in batch]
    trg_batch = [item[1] for item in batch]

    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)

    return src_padded, trg_padded

dataset = PacketDataset(data, text_tokenizer, packet_tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


src_vocab_size = text_tokenizer.vocab_size
trg_vocab_size = packet_tokenizer.vocab_size
src_pad_idx = text_tokenizer.token_to_id["<PAD>"]
trg_pad_idx = packet_tokenizer.token_to_id["<PAD>"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_size=128,
    num_layers=2,
    forward_expansion=2,
    heads=2,
    dropout=0.1,
    device=device,
    max_length=100
).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        # We'll use teacher forcing approach: feed trg[:, :-1] into the model
        trg_input = trg[:, :-1]
        trg_expected = trg[:, 1:]

        # Forward pass
        output = model(src, trg_input)

        # Reshape to compute loss
        # output shape: (batch_size, seq_len, vocab_size)
        # trg_expected shape: (batch_size, seq_len)
        output = output.reshape(-1, output.shape[2])
        trg_expected = trg_expected.reshape(-1)

        loss = criterion(output, trg_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")


def generate_packet(model, user_input_text, max_length=30):
    model.eval()

    # Encode user input with text_tokenizer
    src = text_tokenizer.encode(user_input_text)
    src_tensor = torch.tensor(src).unsqueeze(0).to(device)  # shape: (1, seq_len)

    # Start token for target
    trg_tokens = [packet_tokenizer.token_to_id["<SOS>"]]
    for _ in range(max_length):
        trg_tensor = torch.tensor(trg_tokens).unsqueeze(0).to(device)  # (1, len_so_far)
        
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        
        # output shape: (1, trg_len, vocab_size)
        next_token = output[0, -1, :].argmax(dim=-1).item()
        trg_tokens.append(next_token)
        
        # If <EOS> is generated, stop
        if next_token == packet_tokenizer.token_to_id["<EOS>"]:
            break
    
    # Exclude the first <SOS> and last <EOS>
    return trg_tokens[1:-1] if trg_tokens[-1] == packet_tokenizer.token_to_id["<EOS>"] else trg_tokens[1:]

user_text = "Send a TCP packet from 192.168.229.254 (port 443) to 192.168.202.79 (port 46117), with flags PUSH,ACK and payload size 47 bytes."
generated_token_ids = generate_packet(model, user_text)
decoded_tokens = packet_tokenizer.decode(generated_token_ids)
print("Generated Packet Tokens:", decoded_tokens)
