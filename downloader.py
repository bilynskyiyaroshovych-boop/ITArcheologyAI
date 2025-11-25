# ...existing code...
import os
import time
import requests
from tqdm import tqdm

BASE_DIR = os.environ.get("DATA_DIR", "/data/Archaeological_Dataset")

HEADERS = {"User-Agent": "dataset-downloader/1.0 (contact: your_email@example.com)"}


def fix_thumb_url(url):
    """
    Convert Wikimedia 'thumb' URL to original file URL.
    Example:
      
    """
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
                ra = r.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else backoff
                time.sleep(wait)
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
    """
    Download with retries and backoff
    Returns True or False
    """
    backoff = 1
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=30, stream=True)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else backoff
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except requests.RequestException:
            if attempt == max_retries - 1:
                return False
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    return False


def download_images(query, category, limit=100, per_request=50, sleep_between_requests=1.0):
    """
    Download images from Wikimedia Commons into BASE_DIR/<category>
    """
    os.makedirs(BASE_DIR, exist_ok=True)
    save_folder = os.path.join(BASE_DIR, category)
    os.makedirs(save_folder, exist_ok=True)

    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": min(per_request, max(1, limit)),
        "gsrnamespace": "6",  # File: namespace
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }

    session = requests.Session()
    session.headers.update(HEADERS)

    collected_pages = {}
    print(f"\nSearching: '{query}' (target: {limit} files)")

    # pagination with retries
    while True:
        try:
            data = api_get_with_retries(session, api_url, params)
        except Exception as e:
            print("API request failed:", str(e))
            break

        if "query" in data and "pages" in data["query"]:
            collected_pages.update(data["query"]["pages"])

        if len(collected_pages) >= limit:
            break

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(sleep_between_requests)
            continue
        else:
            break

    if not collected_pages:
        print("No results for:", query)
        return

    pages_list = list(collected_pages.values())
    print(f"Found {len(pages_list)} pages. Starting download (max {limit}).\n")

    downloaded = 0
    for page in tqdm(pages_list):
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
        dst_path = os.path.join(save_folder, img_name)
        if os.path.exists(dst_path):
            downloaded += 1
            continue

        ok = download_with_retries(session, img_url, dst_path, max_retries=6)
        if ok:
            downloaded += 1
        else:
            print("Failed to download:", img_url)

        time.sleep(sleep_between_requests)

    print(f"\nDone. Saved {downloaded} files to {save_folder}")
    print("Open folder:", f'explorer "{save_folder}"')


if __name__ == "__main__":
    download_images("ancient pottery", "ceramics", limit=100)
    download_images("bronze age jewelry", "jewelry", limit=100)
    download_images("neolithic tools", "tools", limit=100)
    download_images("archaeological pottery fragments", "fragments", limit=100)
    download_images("ancient beads", "beads", limit=100)
# ...existing code...