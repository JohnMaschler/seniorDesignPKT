import sqlite3
import os
import json

DB_FILE = "network_traffic.db"

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # create traffic_metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS traffic_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        prompt TEXT NOT NULL,
        pcap_file_path TEXT NOT NULL,
        json_file_path TEXT NOT NULL,
        tags TEXT
    );
    """)

    # create traffic_flows table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS traffic_flows (
        id INTEGER,
        src_ip TEXT,
        dst_ip TEXT,
        protocol TEXT,
        src_port INTEGER,
        dst_port INTEGER,
        timestamp REAL,
        flags TEXT,
        payload_size INTEGER,
        FOREIGN KEY (id) REFERENCES traffic_metadata (id)
    );
    """)

    conn.commit()
    conn.close()

# Insert into traffic_metadata
def insert_traffic_metadata(description, prompt, pcap_file_path, json_file_path, tags):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO traffic_metadata (description, prompt, pcap_file_path, json_file_path, tags)
    VALUES (?, ?, ?, ?, ?);
    """, (description, prompt, pcap_file_path, json_file_path, tags))

    conn.commit()
    metadata_id = cursor.lastrowid  # Get the auto-incremented ID
    conn.close()
    return metadata_id

# Insert into traffic_flows
def insert_traffic_flows(metadata_id, flows):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.executemany("""
    INSERT INTO traffic_flows (id, src_ip, dst_ip, protocol, src_port, dst_port, timestamp, flags, payload_size)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, [(metadata_id, flow['src_ip'], flow['dst_ip'], flow['protocol'], flow['src_port'], flow['dst_port'], 
           flow['timestamp'], flow['flags'], flow['payload_size']) for flow in flows])

    conn.commit()
    conn.close()

# Query metadata and flows
def query_database(description_filter=None, tags_filter=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    query = "SELECT * FROM traffic_metadata"
    conditions = []
    params = []

    if description_filter:
        conditions.append("description LIKE ?")
        params.append(f"%{description_filter}%")
    if tags_filter:
        conditions.append("tags LIKE ?")
        params.append(f"%{tags_filter}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    cursor.execute(query, params)
    metadata_results = cursor.fetchall()

    # here we will fetch related flows for each metadata entry
    for metadata in metadata_results:
        metadata_id = metadata[0]
        cursor.execute("SELECT * FROM traffic_flows WHERE id = ?", (metadata_id,))
        flows = cursor.fetchall()
        print(f"Metadata ID: {metadata_id}")
        print("Flows:")
        for flow in flows:
            print(flow)

    conn.close()

if __name__ == "__main__":
    initialize_database()