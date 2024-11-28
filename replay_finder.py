import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from functools import partial


base_url_general = "https://generals.io/api/replays?"
base_url_user = "https://generals.io/api/replaysForUsername?"

count = 200  # This is max
n_workers = 12
game_type = "1v1"


def filter_by_name(replays, name: str):
    result = []
    for replay in replays:
        player_list = [player["currentName"] for player in replay.get("ranking", [])]
        if name in player_list:
            result.append(replay)
    return result


def filter_by_stars(replays, min_stars: int = 0, max_stars: int = 100):
    result = []
    for replay in replays:
        player_list = [player["stars"] for player in replay.get("ranking", [])]
        if all(min_stars <= stars <= max_stars for stars in player_list):
            result.append(replay)
    return result


def filter_by_type(replays, game_types: list[str]):
    result = []
    for replay in replays:
        if replay["type"] in game_types:
            result.append(replay)
    return result


def filter_by_turns(replays, min_turns: int = 0, max_turns: int = 1000):
    result = []
    for replay in replays:
        if min_turns <= replay["turns"] <= max_turns:
            result.append(replay)
    return result


def fetch_replay_batch(url: str, filters: list = []):
    """
    Fetch a batch of replays starting at the given offset and filter them.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        replays = response.json()
        for f in filters:
            replays = f(replays)
        return len(replays)
    except Exception as e:
        print(f"Error fetching url {url}: {e}")
        return 0


def fetch_replays(username: str = None):
    """
    Fetch replays for a user in parallel using threading.
    """
    offsets = range(0, 4000, 200)
    total = 0

    # make partial functions for filters
    filters = [
        partial(filter_by_type, game_types=[game_type]),
        partial(filter_by_turns, min_turns=0, max_turns=1e5),
        # partial(filter_by_stars, min_stars=600, max_stars=1e5),
    ]

    if username:
        url = f"https://generals.io/api/replaysForUsername?u={username}&"
    else:
        url = "https://generals.io/api/replays?"

    urls = [f"{url}offset={offset}&count=200" for offset in offsets]

    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit tasks for all offsets
        futures = {
            executor.submit(fetch_replay_batch, url, filters): url for url in urls
        }

        # Use tqdm for a progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                total += future.result()
            except Exception as e:
                print(f"Error processing a future: {e}")

    return total


# Run the scraper
if __name__ == "__main__":
    print(fetch_replays("Human.exe"))
