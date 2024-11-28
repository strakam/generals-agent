import subprocess
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

base_url = "https://generalsio-replays-na.s3.amazonaws.com/"

id_filename = "replay_ids/humanexe_replay_ids.txt"
replay_storage_folder = "replays/humanexe_replays"

json_keys = [
    "version",
    "id",
    "mapWidth",
    "mapHeight",
    "usernames",
    "stars",
    "cities",
    "cityArmies",
    "generals",
    "mountains",
    "moves",
    "afks",
    "teams"
]

def decompress_with_js(url):
    try:
        # Run the Node.js script with the URL as an argument
        result = subprocess.run(
            ['node', 'decompress.js', url], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Check for any error in stderr
        if result.stderr:
            print("Error in decompression:", result.stderr.decode('utf-8'))
            return None

        # Return the decompressed data from stdout
        return result.stdout.decode('utf-8')

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def add_json_keys(game_json):
    new_json = {}
    for key, values in zip(json_keys, game_json):
        new_json[key] = values
    return new_json

def download_replay(replay_id: str, folder_name: str, filters: list = []):
    url = base_url + replay_id + ".gior"
    game_json = decompress_with_js(url)
    game_json = json.loads(game_json)
    game_json = add_json_keys(game_json)
    with open(f"{folder_name}/{replay_id}.json", "w") as f:
        json.dump(game_json, f)

if __name__ == "__main__":
    with open(id_filename, "r") as f:
        replay_ids = f.read().splitlines()

        # download the replays in parallel using threading
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(download_replay, replay_id, replay_storage_folder): replay_id for replay_id in replay_ids
            }

            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing a future: {e}")
        

