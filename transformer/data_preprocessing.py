import dpkt
import socket
import json
import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.append(parent_dir)

# Import database functions
from data_scripts.db import insert_traffic_metadata, insert_traffic_flows

SPECIAL_TOKENS = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2}

def tokenize_and_build_vocab(values):
    """Create a mapping for unique values to indices."""
    unique_values = sorted(set(values))
    vocab = {v: i + len(SPECIAL_TOKENS) for i, v in enumerate(unique_values)}
    vocab.update(SPECIAL_TOKENS)  # Add special tokens
    return vocab

def process_pcap(pcap_file_path, description, prompt, tags, output_json="packets.json", max_packets=1000):
    # Initialize data storage
    packet_data = []
    src_ips, dst_ips, protocols, ports = set(), set(), set(), set()

    # Read the pcap file
    with open(pcap_file_path, 'rb') as f:
        pcap_reader = dpkt.pcap.Reader(f)

        for i, (ts, buf) in enumerate(pcap_reader):
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    transport_layer = ip.data
                    protocol = 'TCP' if isinstance(transport_layer, dpkt.tcp.TCP) else 'UDP'
                    src_port = transport_layer.sport
                    dst_port = transport_layer.dport
                    payload_size = len(transport_layer.data)
                    timestamp = ts

                    # Handle TCP flags
                    flags = ""
                    if protocol == 'TCP':
                        tcp_flags = {
                            dpkt.tcp.TH_FIN: "FIN",
                            dpkt.tcp.TH_SYN: "SYN",
                            dpkt.tcp.TH_RST: "RST",
                            dpkt.tcp.TH_PUSH: "PUSH",
                            dpkt.tcp.TH_ACK: "ACK",
                            dpkt.tcp.TH_URG: "URG",
                        }
                        flags = ",".join(name for bit, name in tcp_flags.items() if transport_layer.flags & bit)

                    # Add unique values for vocabularies
                    src_ips.add(src_ip)
                    dst_ips.add(dst_ip)
                    protocols.add(protocol)
                    ports.add(src_port)
                    ports.add(dst_port)

                    # Append packet information
                    packet_data.append({
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "protocol": protocol,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "timestamp": timestamp,
                        "flags": flags,  # Add flags, even if empty
                        "payload_size": payload_size,
                    })

            if i >= max_packets - 1:
                break

    # ============================
    # Append to existing JSON file
    # ============================
    if os.path.exists(output_json):
        with open(output_json, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    # Extend the existing data with new packet_data
    existing_data.extend(packet_data)

    # Write the combined data back to JSON
    with open(output_json, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)
    print(f"Appended {len(packet_data)} packets from {pcap_file_path} to {output_json}")

    # Insert metadata into the database
    metadata_id = insert_traffic_metadata(
        description=description,
        prompt=prompt,
        pcap_file_path=pcap_file_path,
        json_file_path=output_json,
        tags=tags
    )

    # Insert packet data into the database
    insert_traffic_flows(metadata_id, packet_data)

    print(f"Metadata and flows inserted into the database for {description}")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '..')  # Directory containing PCAP files
    output_json = "packets.json"

    # Ensure the output JSON exists or create it with an empty list
    if not os.path.exists(output_json):
        with open(output_json, 'w') as json_file:
            json.dump([], json_file)

    # List of PCAP files to process
    pcap_files = [
        {"path": os.path.join(data_dir, "maccdc2012_00001.pcap"), "description": "Example 2", "tags": "Normal,DNS"},
        {"path": os.path.join(data_dir, "maccdc2012_00000.pcap"), "description": "Example 1", "tags": "Normal,DNS"},
    ]

    for pcap_info in pcap_files:
        process_pcap(
            pcap_file_path=pcap_info["path"],
            description=pcap_info["description"],
            prompt=f"Generate traffic for {pcap_info['description']}",
            tags=pcap_info["tags"],
            output_json=output_json  # Use the same JSON file to append data
        )

