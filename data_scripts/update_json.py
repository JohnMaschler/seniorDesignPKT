import json

# Load your existing JSON file
with open("packets.json", "r") as file:
    packets = json.load(file)

# Function to create a natural language input description
def create_input_description(packet):
    return (
        f"Send a {packet['protocol']} packet from {packet['src_ip']} (port {packet['src_port']}) "
        f"to {packet['dst_ip']} (port {packet['dst_port']}), with flags {packet['flags']} "
        f"and payload size {packet['payload_size']} bytes."
    )

# Transform into desired format
transformed_data = []
for packet in packets:
    transformed_data.append({
        "input": create_input_description(packet),
        "output": packet
    })

# Save the transformed data to a new JSON file
with open("transformed_packets.json", "w") as output_file:
    json.dump(transformed_data, output_file, indent=4)

print("Transformed JSON saved as 'transformed_packets.json'.")
