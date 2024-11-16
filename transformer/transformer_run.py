import pandas as pd
import torch
from transformer_network import NetworkTrafficTransformer

# Load and process data
df = pd.read_csv("packets.csv")

# Prepare special tokens for the vocabularies
SPECIAL_TOKENS = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2}

# Tokenize and create vocabularies for all features
def tokenize_and_build_vocab(values):
    """Create a mapping for unique values to indices."""
    unique_values = sorted(set(values))
    vocab = {v: i + len(SPECIAL_TOKENS) for i, v in enumerate(unique_values)}
    vocab.update(SPECIAL_TOKENS)  # Add special tokens
    return vocab

# Build vocabularies
src_ip_vocab = tokenize_and_build_vocab(df['src_ip'])
dst_ip_vocab = tokenize_and_build_vocab(df['dst_ip'])
protocol_vocab = tokenize_and_build_vocab(df['protocol'])
port_vocab = tokenize_and_build_vocab(pd.concat([df['src_port'], df['dst_port']]))

# Map the original data to tokenized sequences
df['src_ip'] = df['src_ip'].map(src_ip_vocab)
df['dst_ip'] = df['dst_ip'].map(dst_ip_vocab)
df['protocol'] = df['protocol'].map(protocol_vocab)
df['src_port'] = df['src_port'].map(port_vocab)
df['dst_port'] = df['dst_port'].map(port_vocab)

# Prepare source (src) and target (trg) sequences
# <SOS> at the beginning, <EOS> at the end for trg
df['SOS'] = SPECIAL_TOKENS["<SOS>"]
df['EOS'] = SPECIAL_TOKENS["<EOS>"]

src = df[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']].values
trg = df[['SOS', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'EOS']].values

# Convert to tensors
src_tensor = torch.tensor(src, dtype=torch.long)
trg_tensor = torch.tensor(trg, dtype=torch.long)

# Model parameters
src_pad_idx = SPECIAL_TOKENS["<PAD>"]
trg_pad_idx = SPECIAL_TOKENS["<PAD>"]
src_vocab_size = max(src_ip_vocab.values()) + 1

# Find the maximum value from each vocabulary dictionary
max_src_ip = max(src_ip_vocab.values())
max_dst_ip = max(dst_ip_vocab.values())
max_protocol = max(protocol_vocab.values())
max_port = max(port_vocab.values())

# Calculate the target vocabulary size
trg_vocab_size = max(max_src_ip, max_dst_ip, max_protocol, max_port) + 1

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NetworkTrafficTransformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    embed_size=256,
    num_layers=4,
    heads=8,
    device=device,
    max_length=trg.shape[1],  # Target sequence length
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx
).to(device)

# Prepare shifted trg_input for teacher forcing
trg_input = trg_tensor[:, :-1]  # All tokens except the last (shifted for input)
trg_target = trg_tensor[:, 1:]  # All tokens except the first (target for loss calculation)

# Move tensors to device
src = src_tensor.to(device)
trg_input = trg_input.to(device)
trg_target = trg_target.to(device)

# Run the model
output = model(src, trg_input)

# Convert output to predicted token sequences
predicted_tokens = output.argmax(dim=-1)  # Get the token with the highest probability

# Map tokens back to human-readable values using reverse vocabularies
reverse_src_ip_vocab = {v: k for k, v in src_ip_vocab.items()}
reverse_dst_ip_vocab = {v: k for k, v in dst_ip_vocab.items()}
reverse_protocol_vocab = {v: k for k, v in protocol_vocab.items()}
reverse_port_vocab = {v: k for k, v in port_vocab.items()}

# Decode sequences for visualization
decoded_predictions = []
decoded_true_values = []
for pred_seq, true_seq in zip(predicted_tokens, trg_target):
    decoded_pred = [
        reverse_src_ip_vocab.get(token, f"<UNK:{token}>")
        if idx == 1 else
        reverse_dst_ip_vocab.get(token, f"<UNK:{token}>")
        if idx == 1 else
        reverse_port_vocab.get(token, f"<UNK:{token}>")
    ]

    # Append for analysis
    decoded_predictions.append(decoded_pred)

# Visualize predictions vs true sequences
for pred, true_seq in zip(decoded_predictions[:5], decoded_true_values[:5]):
    print(pred)
