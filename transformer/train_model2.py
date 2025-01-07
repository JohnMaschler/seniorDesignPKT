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
    MultiPacketStreamDataset, TextTokenizer, PacketTokenizer, decode_tokens_to_packets
)

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
PACKET_START = "<PACKET_START>"
PACKET_END = "<PACKET_END>"
STREAM_START = "<STREAM_START>"
STREAM_END = "<STREAM_END>"


def collate_fn(batch, text_tokenizer, packet_tokenizer):
    src_tensors = []
    trg_tensors = []

    for (prompt, pkt_tokens) in batch:
        # Encode the prompt and tokens
        src_ids = text_tokenizer.encode(prompt)
        pkt_ids = packet_tokenizer.encode(pkt_tokens)
        pkt_ids.append(packet_tokenizer.token2id[EOS]) 

        # Convert to tensors for padding
        src_t = torch.tensor(src_ids, dtype=torch.long)
        trg_t = torch.tensor(pkt_ids, dtype=torch.long)

        src_tensors.append(src_t)
        trg_tensors.append(trg_t)

    # Pad sequences 
    src_padded = pad_sequence(
        src_tensors, batch_first=True, 
        padding_value=text_tokenizer.word2id[PAD]
    )
    trg_padded = pad_sequence(
        trg_tensors, batch_first=True, 
        padding_value=packet_tokenizer.token2id[PAD]
    )

    return src_padded, trg_padded


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

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:].contiguous().view(-1)

            output = model(src, trg_input)
            output = output.view(-1, output.shape[2])

            loss = criterion(output, trg_expected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

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

# TODO: figure out the best way to predict the next token
# other options might be argmax, multinomial, beam search...
def top_k_logits(logits, k):
    values, indices = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, float('-inf'), logits)


def generate_packets(model, input_text, text_tokenizer, packet_tokenizer, device, max_len=100, k=5):
    model.eval()
    with torch.no_grad():
        src = text_tokenizer.encode(input_text)
        src_tensor = torch.tensor([src], dtype=torch.long).to(device)

        # is it bad to use the stream start and end tokens?
        trg_indices = [packet_tokenizer.token2id[STREAM_START]]
        # trg_indices = []
        token_counts = Counter()

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(device)
            output = model(src_tensor, trg_tensor)
            logits = output[:, -1, :]
            
            top_k_filtered_logits = top_k_logits(logits, k)
            probabilities = torch.softmax(top_k_filtered_logits, dim=1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            if next_token == packet_tokenizer.token2id[EOS]:
                break

            trg_indices.append(next_token)
            token_counts[next_token] += 1

        trg_indices.append(packet_tokenizer.token2id[STREAM_END])
        return packet_tokenizer.decode(trg_indices[1:-1])

def main():
    JSON_PATH = "streams.json"
    dataset = MultiPacketStreamDataset(
        json_file=JSON_PATH,
        max_packets_per_stream=30
    )

    if len(dataset) == 0:
        print(f"No valid samples found in {JSON_PATH}.")
        return

    text_prompts = [sample[0] for sample in dataset]
    packet_data = [sample[1] for sample in dataset]

    text_tokenizer = TextTokenizer()
    text_tokenizer.build_vocab(text_prompts)

    packet_tokenizer = PacketTokenizer()

    flat_packet_data = []
    for pkt_tokens in packet_data:
        for token in pkt_tokens:
            flat_packet_data.append(token)
    packet_tokenizer.build_vocab([flat_packet_data])

    # Debugging: Print tokenizer vocab sizes and mapping
    # print("\nText Tokenizer Vocab Size:", text_tokenizer.vocab_size)
    # print("Packet Tokenizer Vocab Size:", packet_tokenizer.vocab_size)
    # print("Packet Tokenizer Mapping:", packet_tokenizer.token2id)

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
    
    src_vocab_size = text_tokenizer.vocab_size
    trg_vocab_size = packet_tokenizer.vocab_size


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=512,
        num_layers=2,   # change this
        forward_expansion=2, # change this
        heads=4, #change this
        dropout=0.2, # change this
        device=device,
        max_length=1024
    ).to(device)


    train_transformer(
        model=model,
        data_loader=data_loader,
        packet_tokenizer=packet_tokenizer,
        text_tokenizer=text_tokenizer,
        epochs=3, # change this
        lr=1e-3, # change this 
        device=device
    )

    # user_prompt = input("Enter your text prompt for packet generation: ")
    user_prompt = "Generate traffic for Example 1 with src_ip: 192.168.202.68 and dst_ip: 192.168.24.100"
    generated_tokens = generate_packets(model, user_prompt, text_tokenizer, packet_tokenizer, device, max_len=100)
    print("\nGenerated Packet Tokens:")
    print(generated_tokens)

    decoded_packets = decode_tokens_to_packets(generated_tokens)
    print("\nDecoded Packets:")
    for idx, pkt in enumerate(decoded_packets):
        print(f"Packet {idx + 1}: {pkt}")

if __name__ == "__main__":
    main()
