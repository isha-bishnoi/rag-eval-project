import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

DOC_URLS = [
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
    "https://fastapi.tiangolo.com/tutorial/body-fields/",
    "https://fastapi.tiangolo.com/tutorial/body-multiple-params/",
    "https://fastapi.tiangolo.com/tutorial/body-nested-models/",
    "https://fastapi.tiangolo.com/tutorial/body-updates/",
    "https://fastapi.tiangolo.com/tutorial/response-model/",
    "https://fastapi.tiangolo.com/tutorial/response-status-code/",
    "https://fastapi.tiangolo.com/tutorial/request-forms/",
    "https://fastapi.tiangolo.com/tutorial/request-files/",
    "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/",
    "https://fastapi.tiangolo.com/tutorial/security/",
    "https://fastapi.tiangolo.com/tutorial/security/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
    "https://fastapi.tiangolo.com/tutorial/cors/",
    "https://fastapi.tiangolo.com/tutorial/middleware/",
    "https://fastapi.tiangolo.com/tutorial/background-tasks/",
    "https://fastapi.tiangolo.com/tutorial/bigger-applications/",
    "https://fastapi.tiangolo.com/tutorial/metadata/",
    "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/",
    "https://fastapi.tiangolo.com/tutorial/testing/",
    "https://fastapi.tiangolo.com/tutorial/encoder/",
    "https://fastapi.tiangolo.com/tutorial/extra-data-types/",
]

def fetch_page(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    return soup

def save_document(document, filename):
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")

def discover_links(url):
    soup = fetch_page(url)

    links = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]

        full_url = urljoin(url, href)

        parsed_url = urlparse(full_url)

        # Only keep links from FastAPI documentation
        if parsed_url.netloc != "fastapi.tiangolo.com":
            continue

        # Remove #section fragments
        full_url = full_url.split("#")[0]

        # Only keep tutorial/documentation pages
        if "/tutorial/" in full_url:
            links.add(full_url)

    return sorted(links)

def extract_content(soup, url):
    article = soup.find("article")

    if article is None:
        raise ValueError("Could not find documentation article")

    for element in article.find_all(["script", "style", "nav", "footer"]):
        element.decompose()

    title = soup.title.get_text(strip=True)

    # Preserve code blocks before extracting normal text
    for code_block in article.find_all("pre"):
        code_text = code_block.get_text("", strip=False)
        code_block.replace_with(f"\n\n{code_text}\n\n")

    text = article.get_text("\n", strip=True)

    return {
        "title": title,
        "url": url,
        "text": text,
    }

def validate_urls(urls):
    valid_urls = []

    for url in urls:
        try:
            response = requests.head(url, timeout=10)

            if response.status_code == 200:
                valid_urls.append(url)
                print(f"✓ {url}")
            else:
                print(f"✗ {url} -> {response.status_code}")

        except requests.RequestException as exc:
            print(f"✗ {url} -> {exc}")

    return valid_urls

def scrape_documents(urls):
    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Scraping: {url}")

        try:
            soup = fetch_page(url)
            document = extract_content(soup, url)

            path_parts = urlparse(url).path.strip("/").split("/")

            if len(path_parts) > 2:
                filename = "-".join(path_parts[1:]) + ".json"
            else:
                filename = path_parts[-1] + ".json"

            save_document(document, filename)

        except requests.RequestException as exc:
            print(f"Failed to scrape {url}: {exc}")
            
if __name__ == "__main__":
    valid_urls = validate_urls(DOC_URLS)

    print()
    print(f"Valid URLs: {len(valid_urls)}/{len(DOC_URLS)}")

    scrape_documents(valid_urls)