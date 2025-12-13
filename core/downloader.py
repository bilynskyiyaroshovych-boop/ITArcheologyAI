import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.environ.get(
    "DATA_DIR", r"C:\python\Archeological_Dataset"
)

HEADERS = {
    "User-Agent": "dataset-downloader/1.0 (contact: antonbylunskui@gmail.com)"
}


def fix_thumb_url(url: str) -> str:
    if not url or "/thumb/" not in url:
        return url
    prefix, rest = url.split("/thumb/", 1)
    orig_path = rest.rsplit("/", 1)[0]
    return f"{prefix}/{orig_path}"


def api_get_with_retries(session, url, params, max_retries=4):
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", backoff)))
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    raise RuntimeError("API retries exhausted")


def download_with_retries(session, url, dst_path, max_retries=6):
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=30, stream=True)

            if r.status_code == 403:
                tqdm.write(f"Access denied: {url}")
                return False

            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", backoff)))
                backoff = min(backoff * 2, 60)
                continue

            r.raise_for_status()

            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True

        except Exception:
            if attempt == max_retries - 1:
                return False
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    return False


def download_images(args):
    query, category, limit, position_index = args

    per_request = 50
    sleep_between_requests = 1.0

    os.makedirs(BASE_DIR, exist_ok=True)
    save_folder = os.path.join(BASE_DIR, category)
    os.makedirs(save_folder, exist_ok=True)

    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": min(per_request, limit),
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    }

    session = requests.Session()
    session.headers.update(HEADERS)

    collected_pages = {}

    while True:
        try:
            data = api_get_with_retries(session, api_url, params)
        except Exception:
            break

        if "query" in data and "pages" in data["query"]:
            collected_pages.update(data["query"]["pages"])

        if len(collected_pages) >= limit:
            break

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(sleep_between_requests)
        else:
            break

    pages_list = list(collected_pages.values())

    if not pages_list:
        tqdm.write(f"No results for: {category}")
        return

    bar = tqdm(
        total=min(limit, len(pages_list)),
        desc=f"{category:<15}",
        position=position_index,
        leave=True,
        colour="green",
    )

    downloaded = 0

    for page in pages_list:
        if downloaded >= limit:
            break

        if "imageinfo" not in page or not page["imageinfo"]:
            continue

        info = page["imageinfo"][0]
        img_url = info.get("url") or info.get("thumburl")
        if not img_url:
            continue

        img_url = fix_thumb_url(img_url)
        img_name = img_url.split("/")[-1].split("?")[0]

        if img_name.lower().endswith(
            (".pdf", ".djvu", ".tif", ".tiff", ".ogg", ".ogv", ".webm", ".stl")
        ):
            continue

        dst_path = os.path.join(save_folder, img_name)

        if os.path.exists(dst_path):
            downloaded += 1
            bar.update(1)
            continue

        if download_with_retries(session, img_url, dst_path):
            downloaded += 1
            bar.update(1)

        time.sleep(sleep_between_requests)

    bar.close()


def run_downloader(tasks_data):
    """
    tasks_data = [
        ("ancient pottery", "ceramics", 400),
        ...
    ]
    """
    tasks_with_pos = [
        task + (i,) for i, task in enumerate(tasks_data)
    ]

    with ThreadPoolExecutor(max_workers=len(tasks_data)) as executor:
        executor.map(download_images, tasks_with_pos)


if __name__ == "__main__":
    tasks = [
        ("ancient pottery", "ceramics", 600),
        ("bronze age jewelry", "jewelry", 600),
        ("neolithic tools", "tools", 600),
        ("archaeological pottery fragments", "fragments", 600),
        ("ancient beads", "beads", 600),
    ]

    print("Starting download...")
    run_downloader(tasks)
    print("Done.")
