import json

input_file = "./datasets_validation/minif2f/dataset.jsonl"   # your original jsonl file
output_file = "./datasets_validation/minif2f/goldStandard_output.jsonl" # new jsonl file

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)

        # Keep only the needed keys
        new_data = {
            "name": data.get("name"),
            "split": data.get("split"),
            "id": data.get("id"),
            "formal_statement": data.get("formal_statement"),
            "formal_proof": data.get("formal_proof")
        }

        # Write back as jsonl
        outfile.write(json.dumps(new_data, ensure_ascii=True) + "\n")

print(f"Processed file written to {output_file}")
