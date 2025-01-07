import dpkt
import socket
import re
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import json
from torch.utils.data import Dataset



PACKET_START = "<PACKET_START>"
PACKET_END   = "<PACKET_END>"
PAYLOAD_START= "<PAYLOAD_START>"
PAYLOAD_END  = "<PAYLOAD_END>"
STREAM_START = "<STREAM_START>"
STREAM_END = "<STREAM_END>"
SOS = "<SOS>"  # Start Of Sequence (decoder input)
EOS = "<EOS>"  # End Of Sequence (decoder target)
PAD = "<PAD>"

SPECIAL_TOKENS = [
    PACKET_START, PACKET_END, STREAM_END, STREAM_START,
    PAYLOAD_START, PAYLOAD_END,
    SOS, EOS, PAD
]


def decode_tokens_to_packets(tokens):
    packets = []
    # TODO: the STREAM_START token is always missing from the generated tokens
    # if STREAM_START not in tokens or STREAM_END not in tokens:
    #     print("ERROR: Incomplete stream detected. Returning empty packets.")
    #     return packets

    # Remove the stream start and end tokens for now
    tokens = tokens[1:-1]
    current_packet = {}
    inside_payload = False
    payload = []

    for token in tokens:
        if token == PACKET_START:
            current_packet = {}
            payload = []
        elif token == PACKET_END:
            if payload:
                current_packet["payload_hex"] = payload
            packets.append(current_packet)
        elif token == PAYLOAD_START:
            inside_payload = True
        elif token == PAYLOAD_END:
            inside_payload = False
        elif ":" in token and not inside_payload:
            key, value = token.split(":", 1)
            current_packet[key] = value
        elif inside_payload:
            payload.append(token)
    return packets


class MultiPacketStreamDataset(Dataset):
    """
    Each dataset entry is a tuple: (user_prompt, packet_stream_tokens).
    Grouping packets and tokenizing them into a list of tokens.
    use data_preprocessing2.py to generate the json file. (streams.json)
    """
    def __init__(self, json_file, max_packets_per_stream=None):
        self.samples = []

        with open(json_file, 'r') as f:
            streams = json.load(f)

        for stream in streams:
            stream_id = stream.get("stream_id", {})
            stream_id_info = " ".join(f"{key}: {value}" for key, value in stream_id.items())
            # print(f"Stream ID Info: {stream_id_info}")
            prompt = f"{stream['prompt']} [Stream Info: {stream_id_info}]"

            packets = stream["packets"]

            # Limit the number of packets per stream ?
            if max_packets_per_stream:
                packets = packets[:max_packets_per_stream]

            # Tokenize the stream packets
            stream_tokens = multi_packets_to_tokens(packets)
            
            self.samples.append((prompt, stream_tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

"""
build_vocab: Build the vocabulary from a list of texts that are given by the user.
tokenize: Split a text into tokens.
encode: Convert a text into a list of token IDs.
decode: Convert a list of token IDs into a text.
"""
class TextTokenizer:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size = 4
        self.word2id = self.token_to_id

    def build_vocab(self, text_list):
        # Split each text into tokens
        for text in text_list:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def tokenize(self, text):
        # for now, just splitting by space as our tokenization
        tokens = re.findall(r"[A-Za-z0-9.]+", text.lower())
        return tokens

    def encode(self, text):
        # Add <SOS> at the start and <EOS> at the end
        tokens = []
        tokens.append(self.token_to_id["<SOS>"])
        for t in self.tokenize(text):
            if t in self.token_to_id:
                tokens.append(self.token_to_id[t])
            else:
                tokens.append(self.token_to_id["<UNK>"])
        tokens.append(self.token_to_id["<EOS>"])
        return tokens

    def decode(self, tokens):
        return [self.id_to_token.get(t, "<UNK>") for t in tokens]


""" 
PacketTokenizer class for tokenizing packets into a list of tokens.
encode: Convert a list of tokens to a list of token IDs.
build_vocab: Build the vocabulary from a list of token lists.
decode: Convert a list of token IDs to a list of tokens.
"""
class PacketTokenizer:
    def __init__(self):
        self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, token_list):
        encoded = []
        for tok in token_list:
            if tok in self.token2id:
                encoded.append(self.token2id[tok])
            else:
                # Map unknown tokens to <UNK> token
                if "<UNK>" in self.token2id:
                    encoded.append(self.token2id["<UNK>"])
                else:
                    print(f"[WARNING] Unknown token (OOV): {tok}")
        return encoded

    def build_vocab(self, list_of_token_lists):
        unique_tokens = set()
        for token_list in list_of_token_lists:
            unique_tokens.update(token_list)

        for tok in sorted(unique_tokens):
            if tok not in self.token2id:
                self.token2id[tok] = self.vocab_size
                self.id2token[self.vocab_size] = tok
                self.vocab_size += 1

    def decode(self, ids):
        return [self.id2token.get(i, "<UNK>") for i in ids]


# def packet_to_tokens(packet_dict):
#     tokens = []
#     tokens.append(PACKET_START)
#     tokens.append(f"protocol:{packet_dict['protocol']}")
#     tokens.append(f"src_ip:{packet_dict['src_ip']}")
#     tokens.append(f"dst_ip:{packet_dict['dst_ip']}")
#     tokens.append(f"src_port:{packet_dict['src_port']}")
#     tokens.append(f"dst_port:{packet_dict['dst_port']}")
#     tokens.append(f"flags:{packet_dict['flags']}")
#     tokens.append(f"payload_size:{packet_dict['payload_size']}")

#     # tokens.append(PAYLOAD_START)
#     # tokens.extend(packet_dict.get("payload_hex", []))
#     # tokens.append(PAYLOAD_END)

#     tokens.append(PACKET_END)
#     return tokens


def packet_to_tokens(packet_dict):
    """
    Convert a single packet (dict) to a list of tokens.
    """
    tokens = []
    tokens.append(PACKET_START)

    fields_order = [
        "src_ip", "dst_ip", "protocol", "src_port", 
        "dst_port", "timestamp", "flags", "payload_size"
    ]

    for field in fields_order:
        if field in packet_dict:
            token = f"{field}:{packet_dict[field]}"
            tokens.append(token)
            # print(f"Tokenized Field: {token}")

    tokens.append(PACKET_END)
    return tokens



def multi_packets_to_tokens(packet_list):
    """
    Tokenize a list of packets the stream with STREAM_BEGIN and STREAM_END.
    """
    all_tokens = [STREAM_START]
    for pkt in packet_list:
        all_tokens.extend(packet_to_tokens(pkt))
    all_tokens.append(STREAM_END)

    # print("Tokenized Stream:", all_tokens)
    return all_tokens
