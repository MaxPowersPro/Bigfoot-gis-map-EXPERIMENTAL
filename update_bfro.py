import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

CSV_PATH = "data/bfro_reports_clean.csv"
RECENT_URL = "https://www.bfro.net/gdb/newadd.asp?Show=AB"
BASE_REPORT_URL = "https://www.bfro.net/GDB/show_report.asp?id={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MaxquestGISUpdater/1.0)"}


def geocode_county_state(county, state):
    """Geocode County + State using free OpenStreetMap Nominatim (with polite rate limit)."""
    query_parts = [p for p in [county, state, "USA"] if p]
    if not query_parts:
        return None, None
    query = ", ".join(query_parts)
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        time.sleep(1.1)  # Respect Nominatim 1 req/sec rule
        if resp.status_code == 200 and resp.json():
            hit = resp.json()[0]
            return round(float(hit["lat"]), 5), round(float(hit["lon"]), 5)
    except Exception as e:
        print(f"Geocode error for '{query}': {e}")
    return None, None


def fetch_recent_report_ids():
    """Scrape report IDs from BFRO's Recent Additions page."""
    resp = requests.get(RECENT_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    ids = []
    for a in soup.find_all("a", href=True):
        match = re.search(r"show_report\.asp\?id=(\d+)", a["href"], re.IGNORECASE)
        if match:
            rid = match.group(1)
            if rid not in ids:
                ids.append(rid)
    return ids


def parse_report_page(report_id):
    """Fetch and parse a single BFRO report HTML page."""
    url = BASE_REPORT_URL.format(report_id)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip=True)

    # Extract Class rating
    cls_match = re.search(r"\((Class\s+[ABC])\)", text)
    class_rating = cls_match.group(1) if cls_match else "Class B"

    # Extract Title
    title_tag = soup.find("span", class_="reportheader")
    title = title_tag.get_text(strip=True) if title_tag else f"Report #{report_id}"

    # Extract County & State
    state_match = re.search(r"STATE:\s*([A-Za-z\s]+)", text)
    county_match = re.search(r"COUNTY:\s*([A-Za-z\s\.\-]+)", text)
    state = state_match.group(1).strip().split("\n")[0] if state_match else ""
    county = county_match.group(1).strip().split("\n")[0] if county_match else ""

    # Extract Observed narrative
    obs_match = re.search(r"OBSERVED:\s*(.+?)(?:\n[A-Z\s]{4,}:|\Z)", text, re.DOTALL)
    summary = obs_match.group(1).strip().replace("\n", " ") if obs_match else ""

    # Extract Date
    date_match = re.search(r"YEAR:\s*(\d{4})", text)
    event_date = f"{date_match.group(1)}-01-01" if date_match else "N/A"

    lat, lon = geocode_county_state(county, state)
    if lat is None or lon is None:
        print(f"Skipping Report #{report_id} — could not geocode ({county}, {state})")
        return None

    return {
        "report_id": str(report_id),
        "title": f"Report {report_id}: {title}",
        "summary": summary[:2000],
        "latitude": lat,
        "longitude": lon,
        "event_date": event_date,
        "class_rating": class_rating,
        "county": county,
    }


def main():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(
            columns=[
                "report_id",
                "title",
                "summary",
                "latitude",
                "longitude",
                "event_date",
                "class_rating",
                "county",
            ]
        )

    existing_ids = set(df["report_id"].astype(str).str.strip())
    recent_ids = fetch_recent_report_ids()
    new_ids = [rid for rid in recent_ids if rid not in existing_ids]

    print(f"Found {len(recent_ids)} recent BFRO reports; {len(new_ids)} are brand new.")
    if not new_ids:
        return

    new_rows = []
    for rid in new_ids:
        print(f"Fetching Report #{rid}...")
        row = parse_report_page(rid)
        if row:
            new_rows.append(row)
        time.sleep(1.0)

    if new_rows:
        updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df.to_csv(CSV_PATH, index=False)
        print(f"Added {len(new_rows)} new reports to {CSV_PATH}.")


if __name__ == "__main__":
    main()
