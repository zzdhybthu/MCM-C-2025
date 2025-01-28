from tqdm import tqdm
import orjson

def safe_eval(data):
    data = data.replace('null', 'None')
    return eval(data)

def read_jsonl(file_path):
    try:
        with open(file_path, 'rb') as f:
            lines = []
            for line in tqdm(f, leave=False):
                lines.append(orjson.loads(line))
            return lines
    except Exception as e:
        print(f"Error in read {file_path}: {e}")
        return []

def read_json(file_path):
    try:
        with open(file_path, 'rb') as f:
            return orjson.loads(f.read())
    except Exception as e:
        print(f"Error in read {file_path}: {e}")
        return {}

def write_jsonl(data, file_path):
    with open(file_path, 'wb') as f:
        for d in tqdm(data, leave=False):
            f.write(orjson.dumps(d) + b'\n')

def write_json(data, file_path):
    with open(file_path, 'wb') as f:
        f.write(orjson.dumps(data))