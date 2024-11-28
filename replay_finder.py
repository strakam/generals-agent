import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from functools import partial


base_url_general = "https://generals.io/api/replays?"
base_url_user = "https://generals.io/api/replaysForUsername?"

n_workers = 12

def filter_by_name(replays, name: str):
    result = []
    for replay in replays:
        player_list = [player["currentName"] for player in replay.get("ranking", [])]
        if name in player_list:
            result.append(replay)
    return result


def filter_by_elo_game(replays, min_stars: int = 0, max_stars: int = 100):
    result = []
    for replay in replays:
        player_list = [player["stars"] for player in replay.get("ranking", [])]
        if all(min_stars <= stars <= max_stars for stars in player_list):
            result.append(replay)
    return result

def filter_by_elo_player(replays, min_stars: int = 0, max_stars: int = 100):
    result = []
    for replay in replays:
        player_list = [player["stars"] for player in replay.get("ranking", [])]
        if any(min_stars <= stars <= max_stars for stars in player_list):
            result.append(replay)
    return result


def filter_by_n_players(replays, n_players: int = 2):
    result = []
    for replay in replays:
        if len(replay.get("ranking", [])) == n_players:
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
        return replays
    except Exception as e:
        print(f"Error fetching url {url}: {e}")


def fetch_replays(username: str = None):
    """
    Fetch replays for a user in parallel using threading.
    """
    offsets = range(0, 550_000, 200)

    # make partial functions for filters
    filters = [
        partial(filter_by_n_players, n_players=2),
        # partial(filter_by_turns, min_turns=0, max_turns=1e5),
        # partial(filter_by_elo_game, min_stars=65, max_stars=1e5),
    ]

    if username:
        url = f"https://generals.io/api/replaysForUsername?&l=duel&u={username}&"
    else:
        url = "https://generals.io/api/replays?&l=duel&"

    urls = [f"{url}offset={offset}&count=200" for offset in offsets]

    replay_ids = []
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit tasks for all offsets
        futures = {
            executor.submit(fetch_replay_batch, url, filters): url for url in urls
        }

        # Use tqdm for a progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                replays = future.result()
                if replays:
                    replay_ids.extend([replay["id"] for replay in replays])
            except Exception as e:
                print(f"Error processing a future: {e}")

    with open("replay_ids/all_duels.txt", "w") as f:
        f.write("\n".join(replay_ids))
        f.write("\n")

# Run the scraper
if __name__ == "__main__":
    fetch_replays()
