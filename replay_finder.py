import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm


base_url_general = "https://generals.io/api/replays?"
base_url_user = "https://generals.io/api/replaysForUsername?"

count = 200  # This is max
n_workers = 12
game_type = "1v1"

def filter_by_name(replays, names: list[str]):
    pass

def filter_by_stars(replays, min_stars: int = 0, max_stars: int = 100):
    pass

def filter_by_type(replays, game_type: str):
    pass

def filter_by_turns(replays, min_turns: int = 0, max_turns: int = 1000):
    pass

base_url_user = "https://generals.io/api/replaysForUsername?"

count = 200  # This is max
game_type = "1v1"


def get_replay_batch(username: str, offset: int = 0):
    """
    Fetch a batch of replays starting at the given offset and filter them.
    """
    url = f"{base_url_user}u={username}&offset={offset}&count={count}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        data = response.json()
        counter = 0
        for replay in data:
            player_list = replay.get("ranking", [])
            if len(player_list) != 2:
                continue
            if player_list[0]["stars"] > 60 and player_list[1]["stars"] > 60:
                counter += 1
        return counter
    except Exception as e:
        print(f"Error fetching offset {offset}: {e}")
        return 0


def scrape(username: str, max_offset: int = 400_000, batch_size: int = 200):
    """
    Fetch replays for a user in parallel using threading.
    """
    offsets = range(0, max_offset, batch_size)
    total = 0

    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit tasks for all offsets
        futures = {
            executor.submit(get_replay_batch, username, offset): offset
            for offset in offsets
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
    print(scrape("Human.exe"))
