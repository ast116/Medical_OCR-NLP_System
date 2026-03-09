import json
import os

def save_json(data, filename, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, filename + ".json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
