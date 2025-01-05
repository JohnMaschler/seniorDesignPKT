#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformerModel import Transformer

from collections import Counter
import matplotlib.pyplot as plt

from pcap_tokenization2 import (
    MultiPacketStreamDataset, TextTokenizer, PacketTokenizer, create_input_description, decode_tokens_to_packets
)

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
PACKET_START = "<PACKET_START>"
PACKET_END = "<PACKET_END>"

###############################################################################
# 2) Define a Collate Function
###############################################################################
def collate_fn(batch, text_tokenizer, packet_tokenizer):
    src_tensors = []
    trg_tensors = []

    for (prompt, pkt_tokens) in batch:
        src_ids = text_tokenizer.encode(prompt)
        pkt_ids = packet_tokenizer.encode(pkt_tokens)
        pkt_ids.append(packet_tokenizer.token2id[EOS])

        src_t = torch.tensor(src_ids, dtype=torch.long)
        trg_t = torch.tensor(pkt_ids, dtype=torch.long)

        src_tensors.append(src_t)
        trg_tensors.append(trg_t)

    src_padded = pad_sequence(
        src_tensors, batch_first=True, 
        padding_value=text_tokenizer.word2id[PAD]
    )
    trg_padded = pad_sequence(
        trg_tensors, batch_first=True, 
        padding_value=packet_tokenizer.token2id[PAD]
    )

    # Debugging output for collate function
    # print("\n[DEBUG] Source Padded Shape:", src_padded.shape)
    # print("[DEBUG] Target Padded Shape:", trg_padded.shape)

    return src_padded, trg_padded

###############################################################################
# Training Loop with Debugging
###############################################################################
def train_transformer(
    model, 
    data_loader, 
    packet_tokenizer, 
    text_tokenizer, 
    epochs=10, 
    lr=1e-3, 
    device="cpu"
):
    criterion = nn.CrossEntropyLoss(
        ignore_index=packet_tokenizer.token2id[PAD], label_smoothing=0.1
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    best_loss = float('inf')
    losses = []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:].contiguous().view(-1)

            # Debugging: Validate trg_input
            # if (trg_input >= packet_tokenizer.vocab_size).any():
            #     invalid_indices = (trg_input >= packet_tokenizer.vocab_size).nonzero(as_tuple=True)
            #     print(f"\n[ERROR] Invalid indices in trg_input at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            #     print("[DEBUG] trg_input shape:", trg_input.shape)
            #     print("[DEBUG] trg_input max value:", trg_input.max().item())
            #     print("[DEBUG] trg_input invalid values:", trg_input[invalid_indices])
            #     print("[DEBUG] Packet Tokenizer Vocab Size:", packet_tokenizer.vocab_size)
            #     raise ValueError("Invalid indices detected in trg_input.")

            # Debugging: Log shapes
            # print(f"[DEBUG] Epoch {epoch + 1}, Batch {batch_idx + 1}")
            # print("[DEBUG] src shape:", src.shape)
            # print("[DEBUG] trg_input shape:", trg_input.shape)

            output = model(src, trg_input)
            output = output.view(-1, output.shape[2])

            loss = criterion(output, trg_expected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            scheduler.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_tokenizer': text_tokenizer.word2id,
                'packet_tokenizer': packet_tokenizer.token2id
            }, 'model_with_tokenizers.pth')
            print(f"Model saved with loss {avg_loss:.4f}")

    # if we want to plot the loss
    # plt.plot(range(1, epochs + 1), losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.show()

def penalized_generation(logits, token_counts, penalty_factor=2.0):
    for token, count in token_counts.items():
        logits[0, token] -= penalty_factor * count
    return logits


def generate_packets(model, input_text, text_tokenizer, packet_tokenizer, device, max_len=100):
    model.eval()
    with torch.no_grad():
        src = text_tokenizer.encode(input_text)
        src_tensor = torch.tensor([src], dtype=torch.long).to(device)

        trg_indices = [packet_tokenizer.token2id[SOS]]
        token_counts = Counter()

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(device)
            output = model(src_tensor, trg_tensor)
            logits = output[:, -1, :]

            penalized_logits = penalized_generation(logits, token_counts)
            next_token = penalized_logits.argmax(dim=1).item()

            if next_token == packet_tokenizer.token2id[EOS]:
                break

            trg_indices.append(next_token)
            token_counts[next_token] += 1

        return packet_tokenizer.decode(trg_indices[1:])


###############################################################################
# Main Function with Debugging
###############################################################################
def main():
    JSON_PATH = "streams.json"
    dataset = MultiPacketStreamDataset(
        json_file=JSON_PATH,
        max_packets_per_stream=30
    )

    if len(dataset) == 0:
        print("No valid samples found in streams.json.")
        return

    text_prompts = [sample[0] for sample in dataset]
    packet_data = [sample[1] for sample in dataset]

    text_tokenizer = TextTokenizer()
    text_tokenizer.build_vocab(text_prompts)

    packet_tokenizer = PacketTokenizer()
    flat_packet_data = [token for pkt_tokens in packet_data for token in pkt_tokens]
    packet_tokenizer.build_vocab([flat_packet_data])

    # Debugging: Print tokenizer vocab sizes and mapping
    # print("\n[DEBUG] Text Tokenizer Vocab Size:", text_tokenizer.vocab_size)
    # print("[DEBUG] Packet Tokenizer Vocab Size:", packet_tokenizer.vocab_size)
    # print("[DEBUG] Packet Tokenizer Mapping:", packet_tokenizer.token2id)

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, text_tokenizer, packet_tokenizer)
    )

    src_vocab_size = text_tokenizer.vocab_size
    trg_vocab_size = packet_tokenizer.vocab_size
    src_pad_idx = text_tokenizer.word2id[PAD]
    trg_pad_idx = packet_tokenizer.token2id[PAD]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=512,
        num_layers=8,   # change this
        forward_expansion=4, # change this
        heads=4, #change this
        dropout=0.4, # change this
        device=device,
        max_length=1024
    ).to(device)


    train_transformer(
        model=model,
        data_loader=data_loader,
        packet_tokenizer=packet_tokenizer,
        text_tokenizer=text_tokenizer,
        epochs=20,
        lr=1e-3, # change this 
        device=device
    )

    # user_prompt = input("Enter your text prompt for packet generation: ")
    user_prompt = "Generate traffic for Example 2"
    generated_tokens = generate_packets(model, user_prompt, text_tokenizer, packet_tokenizer, device, max_len=100)
    print("\nGenerated Packet Tokens:")
    print(generated_tokens)

    decoded_packets = decode_tokens_to_packets(generated_tokens)
    print("\nDecoded Packets:")
    for idx, pkt in enumerate(decoded_packets):
        print(f"Packet {idx + 1}: {pkt}")

if __name__ == "__main__":
    main()
