import json
import os


def load_lectures(lectures_dir):
    """
    Load all lecture JSON files from the given directory.

    Args:
        lectures_dir (str): Path to the lectures directory

    Returns:
        list: A list of dictionaries with keys 'id' and 'text'
    """
    lectures = []

    for filename in sorted(os.listdir(lectures_dir)):
        if filename.endswith(".json"):
            file_path = os.path.join(lectures_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                lectures.append({
                    "id": data["id"],
                    "text": data["text"]
                })

    return lectures
