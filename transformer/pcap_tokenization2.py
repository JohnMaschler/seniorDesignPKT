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
SOS = "<SOS>"  # Start Of Sequence (decoder input)
EOS = "<EOS>"  # End Of Sequence (decoder target)
PAD = "<PAD>"  # Padding token
PACKET_HEADER_START = "<PACKET_HEADER_START>"
PACKET_HEADER_END = "<PACKET_HEADER_END>"

SPECIAL_TOKENS = [
    PACKET_START, PACKET_END,
    PACKET_HEADER_START, PACKET_HEADER_END,
    PAYLOAD_START, PAYLOAD_END,
    SOS, EOS, PAD
]


def decode_tokens_to_packets(tokens):
    packets = []
    current_packet = {}
    inside_header = False
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
        elif token == PACKET_HEADER_START:
            inside_header = True
        elif token == PACKET_HEADER_END:
            inside_header = False
        elif token == PAYLOAD_START:
            inside_payload = True
        elif token == PAYLOAD_END:
            inside_payload = False
        elif inside_header and ":" in token:
            key, value = token.split(":", 1)
            current_packet[key] = value
        elif inside_payload:
            payload.append(token)

    return packets



###############################################################################
# Minimal Dataset that Groups Packets and Associates a Prompt
###############################################################################

class MultiPacketStreamDataset(Dataset):
    """
    Each dataset entry is a tuple: (user_prompt, packet_stream_tokens).
    """
    def __init__(self, json_file, max_packets_per_stream=None):
        self.samples = []

        # Load streams from JSON
        with open(json_file, 'r') as f:
            streams = json.load(f)

        for stream in streams:
            prompt = stream["prompt"]
            packets = stream["packets"]

            # Optionally limit packets per stream
            if max_packets_per_stream:
                packets = packets[:max_packets_per_stream]

            # Tokenize stream
            stream_tokens = multi_packets_to_tokens(packets)
            self.samples.append((prompt, stream_tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_input_description(packet_group):
    """
    Generate a specific description for a group of packets.
    """
    descriptions = []
    for pkt in packet_group:
        descriptions.append(
            f"Send a {pkt['protocol']} packet from {pkt['src_ip']} (port {pkt['src_port']}) "
            f"to {pkt['dst_ip']} (port {pkt['dst_port']}), with flags {pkt['flags']} "
            f"and payload size {pkt['payload_size']} bytes."
        )
    return " ".join(descriptions)

###############################################################################
# Tokenizers for Text and Packet Sequences
###############################################################################
class TextTokenizer:
    """
    Tokenizes user instructions (e.g., "Generate 3 TCP packets...") 
    into basic word pieces or subwords.
    """
    def __init__(self):
        self.word2id = {PAD: 0, SOS: 1, EOS: 2}
        self.id2word = {0: PAD, 1: SOS, 2: EOS}
        self.vocab_size = 3

    def build_vocab(self, text_list):
        for text in text_list:
            tokens = self.tokenize(text)
            for tok in tokens:
                if tok not in self.word2id:
                    self.word2id[tok] = self.vocab_size
                    self.id2word[self.vocab_size] = tok
                    self.vocab_size += 1

    def tokenize(self, text):
        # Very naive: split on non-alphanumeric
        return re.findall(r"[A-Za-z0-9.]+", text.lower())

    def encode(self, text):
        # Convert text to tokens, map to IDs, add <SOS> and <EOS>
        tokens = [self.word2id.get(tok, 0) for tok in self.tokenize(text)]
        return [self.word2id[SOS]] + tokens + [self.word2id[EOS]]

    def decode(self, ids):
        return [self.id2word.get(i, "<UNK>") for i in ids]


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
                # Map unknown tokens to a special <UNK> token or raise an error
                if "<UNK>" in self.token2id:
                    encoded.append(self.token2id["<UNK>"])
                else:
                    print(f"[WARNING] Unknown token: {tok}")
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
#     """
#     Convert a single packet (dict) to a list of tokens with special delimiters.
#     """
#     tokens = []
#     tokens.append(PACKET_START)
#     tokens.append(PACKET_HEADER_START)
#     tokens.append(f"protocol:{packet_dict['protocol']}")
#     tokens.append(f"src_ip:{packet_dict['src_ip']}")
#     tokens.append(f"dst_ip:{packet_dict['dst_ip']}")
#     tokens.append(f"src_port:{packet_dict['src_port']}")
#     tokens.append(f"dst_port:{packet_dict['dst_port']}")
#     tokens.append(f"flags:{packet_dict['flags']}")
#     tokens.append(f"payload_size:{packet_dict['payload_size']}")
#     tokens.append(PACKET_HEADER_END)

#     # Optionally include payload tokens
#     # tokens.append(PAYLOAD_START)
#     # tokens.extend(packet_dict.get("payload_hex", []))
#     # tokens.append(PAYLOAD_END)

#     tokens.append(PACKET_END)
#     return tokens

def packet_to_tokens(packet_dict):
    """
    Convert a single packet (dict) to a list of tokens with a consistent order.
    """
    tokens = []
    tokens.append(PACKET_START)
    tokens.append(PACKET_HEADER_START)
    
    # Use a consistent order for fields
    for field in ['protocol', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'flags', 'payload_size']:
        if field in packet_dict:
            tokens.append(f"{field}:{packet_dict[field]}")
    
    tokens.append(PACKET_HEADER_END)
    tokens.append(PACKET_END)
    return tokens



def multi_packets_to_tokens(packet_list):
    """
    Concatenate tokens for multiple packets into one flat sequence.
    """
    all_tokens = []
    for pkt in packet_list:
        all_tokens.extend(packet_to_tokens(pkt))
    return all_tokens