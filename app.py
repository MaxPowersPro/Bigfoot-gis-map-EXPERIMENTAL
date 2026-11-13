import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
from streamlit_js_eval import get_geolocation
import random
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import os
import io
import wave
import json
import urllib.parse
from data.data_loader import load_and_standardize_dataset

st.set_page_config(
    page_title="Bigfoot Field Analysis Platform",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_bfro_clean_data(path="data/bfro_reports_clean.csv"):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return []

    records = []
    for _, row in df.iterrows():
        real_terrain = row.get("terrain_roughness_score")
        records.append({
            "id": str(row.get("report_id", "")),
            "report_id": str(row.get("report_id", "")),
            "title": str(row.get("title", "BFRO Report")) if pd.notna(row.get("title")) else "BFRO Report",
            "summary": str(row.get("summary", "")) if pd.notna(row.get("summary")) else "",
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "event_date": str(row.get("event_date", "N/A")) if pd.notna(row.get("event_date")) else "N/A",
            "class_rating": str(row.get("class_rating", "Unclassified")) if pd.notna(row.get("class_rating")) else "Unclassified",
            "county": str(row.get("county", "")) if pd.notna(row.get("county")) else "",
            "source": "BFRO",
            "real_terrain_roughness": float(real_terrain) if pd.notna(real_terrain) else None,
        })
    return records

raw_sightings_bfro = load_bfro_clean_data()

def load_junk_drawer_data(path="data/junk_drawer.csv"):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return []
    return df.to_dict("records")

def load_researcher_archives_data(path="data/researcher_archives.csv"):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return []
    return df.to_dict("records")

def load_john_green_data(path="data/john_green_incidents_clean.csv"):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return []

    records = []
    for _, row in df.iterrows():
        town = str(row.get("i_nearest_town", "")) if pd.notna(row.get("i_nearest_town")) else ""
        state = str(row.get("i_state_prov", "")) if pd.notna(row.get("i_state_prov")) else ""
        title = f"Historical Account near {town}, {state}" if town else f"Historical Account, {state}"
        has_tracks = pd.notna(row.get("i_tracks")) or pd.notna(row.get("i_cast_made"))
        citation = str(row.get("i_name", "")) if pd.notna(row.get("i_name")) else "John Green Historical Archive"
        narrative = str(row.get("i_account_of_incident", "")) if pd.notna(row.get("i_account_of_incident")) else ""
        summary_text = narrative if narrative else f"(No narrative on file — citation: {citation})"

        real_terrain = row.get("terrain_roughness_score")
        records.append({
            "id": f"JG-{row.get('i_incident_id', '')}",
            "report_id": f"JG-{row.get('i_incident_id', '')}",
            "title": title,
            "summary": summary_text,
            "citation": citation,
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "event_date": str(row.get("i_observation_date", "N/A")),
            "class_rating": "Historical Account",
            "has_physical_evidence": bool(has_tracks),
            "county": str(row.get("i_county", "")) if pd.notna(row.get("i_county")) else "",
            "state": state,
            "source": "John Green Historical Archive",
            "real_terrain_roughness": float(real_terrain) if pd.notna(real_terrain) else None,
        })
    return records

def load_renner_bfro_data(
    local_path="data/bfro_reports_geocoded.csv",
    fallback_url="https://raw.githubusercontent.com/timothyrenner/bfro_sightings_data/master/bfro_reports_geocoded.csv",
):
    df = None
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
        except Exception:
            try:
                df = pd.read_csv(local_path, engine="python", on_bad_lines="skip")
            except Exception:
                df = None

    if df is None:
        try:
            df = pd.read_csv(fallback_url)
        except Exception:
            try:
                df = pd.read_csv(fallback_url, engine="python", on_bad_lines="skip")
            except Exception:
                return []

    records = []
    for _, row in df.iterrows():
        lat_val = row.get("latitude")
        lon_val = row.get("longitude")
        if pd.isna(lat_val) or pd.isna(lon_val):
            continue

        raw_num = row.get("number", row.get("report_id", ""))
        rep_id = str(int(float(raw_num))) if pd.notna(raw_num) and str(raw_num).replace(".", "", 1).isdigit() else str(raw_num).strip()

        title = str(row.get("title", "BFRO Report")) if pd.notna(row.get("title")) else "BFRO Report"
        observed = str(row.get("observed", row.get("summary", ""))) if pd.notna(row.get("observed", row.get("summary"))) else ""
        ev_date = str(row.get("date", row.get("event_date", "N/A"))) if pd.notna(row.get("date", row.get("event_date"))) else "N/A"
        cls_raw = str(row.get("classification", row.get("class_rating", "Unclassified"))) if pd.notna(row.get("classification", row.get("class_rating"))) else "Unclassified"
        if cls_raw in ["Class A", "Class B", "Class C"] or "Class" in cls_raw:
            class_rating = cls_raw
        elif cls_raw in ["A", "B", "C"]:
            class_rating = f"Class {cls_raw}"
        else:
            class_rating = cls_raw

        records.append({
            "id": rep_id,
            "report_id": rep_id,
            "title": title,
            "summary": observed,
            "latitude": float(lat_val),
            "longitude": float(lon_val),
            "event_date": ev_date,
            "class_rating": class_rating,
            "county": str(row.get("county", "")) if pd.notna(row.get("county")) else "",
            "state": str(row.get("state", "")) if pd.notna(row.get("state")) else "",
            "source": "BFRO (Renner Geocoded)",
            "real_terrain_roughness": None,
        })
    return records


def merge_bfro_deduplicated(primary_bfro, renner_bfro):
    seen_ids = set()
    seen_coord_dates = set()
    merged = []

    for rec in primary_bfro:
        rid = str(rec.get("report_id", "")).strip()
        if rid and rid.lower() != "nan":
            seen_ids.add(rid)
        lat_v, lon_v = rec.get("latitude"), rec.get("longitude")
        if lat_v is not None and lon_v is not None:
            seen_coord_dates.add((round(float(lat_v), 3), round(float(lon_v), 3), str(rec.get("event_date", "N/A")).strip()))
        merged.append(rec)

    for rec in renner_bfro:
        rid = str(rec.get("report_id", "")).strip()
        if rid and rid in seen_ids:
            continue
        lat_v, lon_v = rec.get("latitude"), rec.get("longitude")
        if lat_v is None or lon_v is None:
            continue
        coord_key = (round(float(lat_v), 3), round(float(lon_v), 3), str(rec.get("event_date", "N/A")).strip())
        if coord_key in seen_coord_dates:
            continue
        if rid:
            seen_ids.add(rid)
        seen_coord_dates.add(coord_key)
        merged.append(rec)

    return merged


# --- Local CSV Storage Helpers (replacing Supabase) ---
ACOUSTIC_CSV_PATH = "data/acoustic_reports_rows.csv"
INVESTIGATOR_LOGS_CSV_PATH = "data/investigator_logs_rows.csv"
SAVED_SEARCHES_CSV_PATH = "data/saved_searches_rows.csv"

def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    return str(val).strip().lower() in ("true", "1", "t", "yes")

def load_local_csv_records(path):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return []
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            rec[col] = None if pd.isna(val) else val
        records.append(rec)
    return records

def save_local_csv_records(path, records, default_columns):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not records:
        df = pd.DataFrame(columns=default_columns)
    else:
        df = pd.DataFrame(records)
        for col in default_columns:
            if col not in df.columns:
                df[col] = None
    df.to_csv(path, index=False)

@st.cache_data(ttl=3600, show_spinner=False)
def load_wind_turbines_for_sector(center_lat, center_lon, radius_mi):
    """Load wind turbines from local data/wind_turbines.csv (if present) or live USGS USWTDB API for the sector."""
    turbines = []
    local_candidates = ["data/wind_turbines.csv", "data/uswtdb.csv", "data/wind_turbines_rows.csv"]
    local_file = next((p for p in local_candidates if os.path.exists(p)), None)

    if local_file:
        try:
            df = pd.read_csv(local_file)
            lat_col = next((c for c in df.columns if c.lower() in ["ylat", "latitude", "lat"]), None)
            lon_col = next((c for c in df.columns if c.lower() in ["xlong", "longitude", "lon", "lng"]), None)
            name_col = next((c for c in df.columns if c.lower() in ["p_name", "project_name", "name", "facility_name"]), None)
            cap_col = next((c for c in df.columns if c.lower() in ["t_cap", "p_cap", "capacity_mw"]), None)
            if lat_col and lon_col:
                for _, row in df.iterrows():
                    t_lat, t_lon = row.get(lat_col), row.get(lon_col)
                    if pd.isna(t_lat) or pd.isna(t_lon):
                        continue
                    turbines.append({
                        "latitude": float(t_lat),
                        "longitude": float(t_lon),
                        "project_name": str(row.get(name_col, "Wind Facility")) if name_col and pd.notna(row.get(name_col)) else "Wind Facility",
                        "capacity": row.get(cap_col) if cap_col and pd.notna(row.get(cap_col)) else None,
                    })
                return turbines
        except Exception:
            pass

    # Fallback: query official USGS U.S. Wind Turbine Database (USWTDB) API within bounding box
    deg_buf = (radius_mi + 15.0) / 69.0
    min_lat, max_lat = center_lat - deg_buf, center_lat + deg_buf
    min_lon, max_lon = center_lon - deg_buf, center_lon + deg_buf
    api_url = (
        "https://energy.usgs.gov/api/uswtdb/v1/turbines"
        f"?ylat=gte.{min_lat:.4f}&ylat=lte.{max_lat:.4f}"
        f"&xlong=gte.{min_lon:.4f}&xlong=lte.{max_lon:.4f}"
        "&select=case_id,p_name,ylat,xlong,t_cap,p_tnum&limit=5000"
    )
    try:
        resp = requests.get(api_url, timeout=8)
        if resp.status_code == 200:
            for item in resp.json():
                t_lat, t_lon = item.get("ylat"), item.get("xlong")
                if t_lat is not None and t_lon is not None:
                    turbines.append({
                        "latitude": float(t_lat),
                        "longitude": float(t_lon),
                        "project_name": str(item.get("p_name") or "Wind Facility"),
                        "capacity": item.get("t_cap"),
                    })
    except Exception:
        pass
    return turbines


raw_sightings_renner = load_renner_bfro_data()
raw_sightings_bfro_merged = merge_bfro_deduplicated(raw_sightings_bfro, raw_sightings_renner)
raw_sightings_john_green = load_john_green_data()
raw_sightings = raw_sightings_bfro_merged + raw_sightings_john_green
raw_lore = load_and_standardize_dataset("data/indigenous_lore.csv")
raw_news = load_and_standardize_dataset("data/press_archives.csv")
raw_camps = load_and_standardize_dataset("data/campsites.csv")
raw_junk = load_junk_drawer_data()
all_junk_records = raw_junk

raw_researcher_archives = load_researcher_archives_data()
KRANTZ_PDF_URL = "https://raw.githubusercontent.com/MaxPowersPro/Bigfoot-gis-map-EXPERIMENTAL/main/data/krantz_finding_aid.pdf"

if "user_lat" not in st.session_state or "user_lon" not in st.session_state:
    device_loc = get_geolocation()
    if device_loc and "coords" in device_loc:
        st.session_state.user_lat = device_loc["coords"]["latitude"]
        st.session_state.user_lon = device_loc["coords"]["longitude"]
        st.session_state.location_name = "Detected Local Sector"
    else:
        st.session_state.user_lat = 41.4384
        st.session_state.user_lon = -123.7060
        st.session_state.location_name = "Bluff Creek, CA (Patterson-Gimlin Film Site)"

if "user_state" not in st.session_state:
    st.session_state.user_state = "California"
if "user_county" not in st.session_state:
    st.session_state.user_county = "Humboldt County"

if "pending_saved_search" in st.session_state:
    _pending = st.session_state.pop("pending_saved_search")
    st.session_state.user_lat = float(_pending["latitude"])
    st.session_state.user_lon = float(_pending["longitude"])
    st.session_state.location_name = _pending.get("location_name", "Loaded Search")
    st.session_state.radius_miles_key = int(_pending.get("radius_miles", 100))
    _layers_raw = _pending.get("layers")
    if isinstance(_layers_raw, str):
        try:
            _layers_raw = json.loads(_layers_raw)
        except Exception:
            _layers_raw = {}
    for _layer_key, _layer_val in (_layers_raw or {}).items():
        st.session_state[_layer_key] = bool(_layer_val)

lat = float(st.session_state.user_lat)
lon = float(st.session_state.user_lon)
loc_name = str(st.session_state.location_name)
active_state = str(st.session_state.user_state)
active_county = str(st.session_state.user_county)

try:
    st.image("image.png", use_container_width=True)
except Exception:
    try:
        st.image("header_banner.png", use_container_width=True)
    except Exception:
        st.title("Maxquest GIS")

st.caption("Site-Specific Spatial Map & Predictive Multi-Criteria Analysis Engine")

with st.expander("📱 How to Use Maxquest & Master Field Navigation Guide", expanded=False):
    st.markdown("### 🎓 App Navigation & Field Controls")
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown("#### 📍 Searching & Device GPS")
        st.write("""
        * **Target Search Area:** Type any city, county, or landmark in the sidebar search box to re-center the analysis map.
        * **Device GPS:** Tap **📲 Device GPS** in the sidebar to lock onto your phone's live location in the field.
        * **Field Radius:** Select a radius (25--500 miles) to set your operational search perimeter.
        """)
    with col_g2:
        st.markdown("#### 🗺️ Master Map Key")
        st.write("""
        * **👣📜 Footprint Markers:** Blue = standard biological sighting. Purple = report text includes documented high-strangeness language (UFO, mystery lights, mind-speak, etc.) — a content flag, not a claim that anything paranormal occurred. Footprint vs. scroll icon shows BFRO vs. John Green historical source.
        * **🚨 Red Dotted Rings:** Hot Zones where reports, wildlife density, cover, and water overlap.
        * **🪹 Orange Dotted Rings:** Predictive Refuges — high inferred habitat quality with no direct sightings nearby (could mean it's genuinely quiet, or that reports aren't reaching outside databases).
        * **🌲 Green Channels:** Larson transit corridors (≤20 miles) connecting nearby Hot Zones along natural terrain gaps.
        * **🔊 Purple Marker & Wind Farm Clusters:** Infrasound sources & wind turbine clusters. Bold inner ring = where it can actually be *felt*; light outer ring = where it's still *detectable*. Full source details are in the Local Intel drawer below the map.
        * **🏕️ Green Campgrounds:** Dispersed campsites and backcountry staging points.
        """)
    with col_g3:
        st.markdown("#### ⚙️ Sidebar Layer Toggles")
        st.write("""
        * Use the **Active Map Layers** in the sidebar to show or hide specific data overlays in real time.
        * Turn layers off to declutter dense search sectors when analyzing high-density sighting clusters.
        * **📝 Submitting a Field Log:** Enter a Researcher Name in the sidebar first, then use the "Submit Investigator Field Log" drawer near the bottom of the page. Full instructions are inside that drawer.
        """)
    st.markdown("---")

def apply_jitter(lat_val, lon_val, offset_seed=0):
    random.seed(int(lat_val * 1000) + int(lon_val * 1000) + offset_seed)
    return lat_val + random.uniform(-0.003, 0.003), lon_val + random.uniform(-0.003, 0.003)

def get_season(date_str):
    if not date_str or date_str == 'N/A':
        return 'Unknown'
    try:
        month = int(str(date_str).split('-')[1])
        if month in [12, 1, 2]: return '❄️ Winter'
        elif month in [3, 4, 5]: return '🌸 Spring'
        elif month in [6, 7, 8]: return '☀️ Summer'
        elif month in [9, 10, 11]: return '🍂 Autumn'
    except Exception:
        pass
    return 'Unknown'

def filter_urban(check_lat, check_lon):
    urban_bounds = [
        {"min_lat": 35.5, "max_lat": 35.7, "min_lon": -82.65, "max_lon": -82.45},
        {"min_lat": 27.8, "max_lat": 28.1, "min_lon": -82.55, "max_lon": -82.30},
        {"min_lat": 28.4, "max_lat": 28.65, "min_lon": -81.50, "max_lon": -81.20},
        {"min_lat": 38.0, "max_lat": 38.2, "min_lon": -84.6, "max_lon": -84.4},
    ]
    for b in urban_bounds:
        if b["min_lat"] <= check_lat <= b["max_lat"] and b["min_lon"] <= check_lon <= b["max_lon"]:
            return True
    return False

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

INFRASOUND_TYPES = {
    "waterfall": {"label": "🌊 Waterfall / Hydrological", "cause": "High-volume water impact at waterfalls, rapids, or river spillways.", "freq_range": "3 – 15 Hz", "character": "Continuous rumble.", "felt_miles": 12, "travel_miles": 80, "effect": "Falls mostly in the vestibular resonance band (1-7 Hz) — dizziness, pressure headaches, and loss of balance are the most commonly reported effects within the felt zone."},
    "wind": {"label": "🌬️ Wind / Mountain Pass", "cause": "High-velocity wind funneling through narrow ridge gaps, saddles, or mountain passes.", "freq_range": "0.5 – 5 Hz", "character": "Continuous, weather- and pressure-system dependent.", "felt_miles": 8, "travel_miles": 55, "effect": "Sits at the low end of the vestibular band — dizziness and disorientation are most likely, especially during active frontal weather passing through the notch."},
    "manmade": {"label": "⚙️ Man-Made (Dam / Mine / Wind Farm)", "cause": "Dam spillway turbulence, wind-turbine blade-pass harmonics (0.5–12 Hz), or mine/quarry blasting.", "freq_range": "0.5 – 15 Hz", "character": "Dams/turbines: steady rhythmic hum. Mines/quarries: sharp, punctuated blast pulses.", "felt_miles": 8, "travel_miles": 60, "effect": "Wind turbine blade-pass harmonics and dam hum overlap the 7–12 Hz CNS alpha-wave band and can trigger unease, ear pressure, or dread within the felt zone."},
    "biological": {"label": "🦍 Biological", "cause": "Large thoracic resonance structures and specialized laryngeal folds in large-bodied animals producing sub-audible vocal bursts.", "freq_range": "8 – 18 Hz", "character": "Vocal burst, not continuous.", "felt_miles": 3, "travel_miles": 8, "effect": "Overlaps the human alpha-wave / 'being watched' band (7-12 Hz) most directly, but the felt zone is small since source amplitude is far below geological or man-made sources."},
}

RANGE_BASED_BIOLOGICAL_SPECIES = {
    "American Alligator": {
        "states": {
            "Texas", "Oklahoma", "Louisiana", "Arkansas", "Mississippi",
            "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina",
        },
        "frequency": "16–25 Hz",
        "description": (
            "Male American alligators produce infrasound (16-25 Hz) during "
            "breeding-season bellows (roughly April-May); the sub-audible "
            "component makes water on their backs visibly 'dance' in Faraday "
            "wave patterns, felt from 50+ ft away. Documented by Kent Vliet "
            "(Univ. of Florida)."
        ),
        "citation": "Range: USFWS Federal Register (2021). Frequency: Vliet, Univ. of Florida.",
    },
    "Ruffed Grouse": {
        "states": {
            "Maine", "New Hampshire", "Vermont", "Massachusetts", "Connecticut",
            "New York", "Pennsylvania", "New Jersey",
            "Virginia", "West Virginia", "Ohio", "Maryland",
            "North Carolina", "Tennessee", "Kentucky", "Georgia",
            "Michigan", "Wisconsin", "Minnesota", "Illinois", "Indiana", "Iowa",
            "South Dakota", "North Dakota",
            "Montana", "Idaho", "Wyoming", "Utah", "Washington", "Oregon", "Alaska",
        },
        "frequency": "~45 Hz peak (partial true infrasound per Freeman 2012)",
        "description": (
            "Male ruffed grouse perform a wing-beat 'drumming' display during "
            "breeding season, striking the air up to 50 times in ~10 seconds "
            "fast enough to create a small sonic boom; often felt as much as "
            "heard. Peak frequency (~45 Hz, Garcia et al. 2012) sits just "
            "above the strict 20 Hz infrasound cutoff, but the display's "
            "energy extends into true infrasound (Freeman 2012)."
        ),
        "citation": "Range: Ruffed Grouse Society. Frequency: Garcia et al. 2012; Freeman 2012.",
    },
}

BIO_SPECIES_LISTEN_LINKS = {
    "Ruffed Grouse": "https://search.macaulaylibrary.org/catalog?taxonCode=rufgro",
}

def classify_infrasound_type(event_type_str):
    s = str(event_type_str).lower()
    if any(k in s for k in ["waterfall", "falls", "rapid", "hydro", "niagara", "snoqualmie"]):
        return "waterfall"
    if any(k in s for k in ["wind", "pass", "ridge", "saddle", "gap", "aeolian"]) and "turbine" not in s and "farm" not in s:
        return "wind"
    if any(k in s for k in ["dam", "mine", "quarry", "blast", "turbine", "wind farm"]):
        return "manmade"
    if any(k in s for k in ["biotic", "biological", "animal", "vocal", "call"]):
        return "biological"
    return None

def cluster_wind_turbines(turbines, cluster_radius_miles=3.5):
    """Group individual wind turbines into wind farm clusters and build one merged polygon per cluster."""
    if not turbines:
        return []
    pts = np.array([[t["latitude"], t["longitude"]] for t in turbines])
    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS spatial clustering within cluster_radius_miles
        queue = [i]
        visited[i] = True
        member_indices = []
        while queue:
            curr = queue.pop(0)
            member_indices.append(curr)
            dlat = np.radians(pts[:, 0] - pts[curr, 0])
            dlon = np.radians(pts[:, 1] - pts[curr, 1])
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(pts[curr, 0])) * np.cos(np.radians(pts[:, 0])) * np.sin(dlon / 2) ** 2
            dists = 3958.8 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            neighbors = np.where((~visited) & (dists <= cluster_radius_miles))[0]
            for nb in neighbors:
                visited[nb] = True
                queue.append(int(nb))

        members = [turbines[idx] for idx in member_indices]
        c_lat = float(np.mean([m["latitude"] for m in members]))
        c_lon = float(np.mean([m["longitude"] for m in members]))
        # Pick most frequent non-generic project name in the cluster
        names = [m["project_name"] for m in members if m.get("project_name") and m["project_name"] != "Wind Facility"]
        farm_name = max(set(names), key=names.count) if names else "Wind Farm Cluster"

        # Build one merged buffered hull (~2.0 mi buffer around turbines) so overlapping turbine zones fuse into one shape
        buf_deg = 2.0 / 69.0
        buffered_shapes = [Point(m["longitude"], m["latitude"]).buffer(buf_deg, resolution=12) for m in members]
        merged_geom = unary_union(buffered_shapes)
        polygons_latlon = []
        if merged_geom.geom_type == "Polygon":
            polygons_latlon.append([[y, x] for x, y in merged_geom.exterior.coords])
        elif merged_geom.geom_type == "MultiPolygon":
            for poly in merged_geom.geoms:
                polygons_latlon.append([[y, x] for x, y in poly.exterior.coords])

        clusters.append({
            "name": farm_name,
            "turbine_count": len(members),
            "latitude": c_lat,
            "longitude": c_lon,
            "polygons_latlon": polygons_latlon,
            "felt_radius_miles": 6.0,
            "prop_radius_miles": 35.0,
        })
    return clusters

EVIDENCE_CATEGORIES = {
    "Tracks / Footprints": ["track", "footprint", "print", "cast"],
    "Vocalizations": ["whoop", "howl", "scream", "whistle", "vocal", "holler", "yell", "call"],
    "Tree Knocks": ["knock", "banging", "wood knock"],
    "Rock Throwing": ["rock", "stone", "thrown", "throw"],
    "Distinctive Odor": ["smell", "odor", "stench", "stink", "foul"],
}

def scan_evidence_patterns(sightings):
    results = {cat: [] for cat in EVIDENCE_CATEGORIES}
    for s in sightings:
        text = f"{s.get('title', '')} {s.get('summary', '')}".lower()
        season = get_season(s.get("event_date", "N/A"))
        for category, keywords in EVIDENCE_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                results[category].append({"report": s, "season": season})
    return results

PARANORMAL_KEYWORDS = [
    "ufo", "mystery light", "orb", "telepath", "mind-speak", "mindspeak", "mind speak",
    "vanished", "disappeared without", "teleport", "portal", "shapeshift", "shape-shift",
    "glowing eyes", "no tracks after", "tracks stopped", "psychic",
    "shadow figure", "shadow person", "disembodied", "apparition", "interdimensional",
    "cloaked", "cloak", "shimmering", "predator-like", "invisible", "poltergeist",
    "time slip", "lost time", "missing time", "sixth sense", "astral",
]

SENSORY_DREAD_KEYWORDS = [
    "sense of dread", "overwhelming dread", "feeling of dread", "unexplained dread",
    "being watched", "feeling watched", "eyes on me", "heavy silence", "forest went quiet",
    "woods went quiet", "dead silent", "sudden silence", "hair stood up", "hair on my neck",
    "nausea", "nauseous", "dizzy", "dizziness", "vertigo", "ear pressure", "head pressure",
    "chest pressure", "vibration", "vibrating", "low hum", "sub-audible", "felt it in my chest",
    "peripheral shadow", "out of the corner of my eye", "panicked", "overwhelming fear",
    "paralyzed with fear", "couldn't move", "frozen in fear",
]

def scan_paranormal_content(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in PARANORMAL_KEYWORDS)

def scan_sensory_dread(text: str) -> list:
    low = text.lower()
    return [kw for kw in SENSORY_DREAD_KEYWORDS if kw in low]

def calculate_human_effort_factor(dist_to_road_miles: float, pop_density_sq_mi: float) -> float:
    safe_dist = max(0.01, dist_to_road_miles)
    pop_scalar = max(0.1, pop_density_sq_mi / 50.0)
    proximity_friction = 1.0 / (safe_dist + 0.1)
    effort_factor = pop_scalar * proximity_friction
    return round(effort_factor, 3)

def calculate_adjusted_evidence_weight(report_class: str, has_physical_evidence: bool, effort_factor: float, lore_boost: bool = False, k: float = 0.5) -> dict:
    if has_physical_evidence:
        base_weight = 3.0
    elif "Class A" in str(report_class) or "Historical Account" in str(report_class):
        base_weight = 1.5
    elif "Class B" in str(report_class):
        base_weight = 0.8
    else:
        base_weight = 0.3
    if lore_boost:
        base_weight += 0.25
    adjusted_weight = base_weight / (1.0 + (k * effort_factor))
    final_weight = max(0.1, min(4.0, adjusted_weight))
    return {
        "base_weight": base_weight,
        "effort_factor": effort_factor,
        "final_weight": round(final_weight, 2),
        "audit_explanation": f"Base ({base_weight}x) / (1 + {k} * Effort({effort_factor})) = {round(final_weight, 2)}x"
    }

def calculate_seasonal_cover_index(event_month: int, prop_evergreen: float, prop_deciduous: float, has_persistent_understory: bool = True) -> float:
    if 5 <= event_month <= 10:
        leaf_status = 1.0
    else:
        leaf_status = 0.20
    understory_bonus = 0.25 if has_persistent_understory else 0.0
    sc_m = prop_evergreen + (prop_deciduous * leaf_status) + understory_bonus
    return round(min(1.0, max(0.0, sc_m)), 2)

def calculate_environmental_suitability_index(sc_m: float, dist_to_water_miles: float, terrain_roughness_score: float, ungulate_biomass_score: float) -> float:
    water_score = 1.0 if dist_to_water_miles <= 0.5 else (0.7 if dist_to_water_miles <= 2.0 else 0.3)
    esi = (0.35 * sc_m) + (0.25 * water_score) + (0.20 * terrain_roughness_score) + (0.20 * ungulate_biomass_score)
    return round(min(1.0, max(0.0, esi)), 3)

def generate_gpx(target_lat, target_lon, loc_title, sightings, camps, audio, community_logs):
    gpx = ET.Element("gpx", version="1.1", creator="BigfootFieldPlatform", xmlns="http://www.topografix.com/GPX/1/1")
    wpt_target = ET.SubElement(gpx, "wpt", lat=str(target_lat), lon=str(target_lon))
    ET.SubElement(wpt_target, "name").text = f"TARGET: {loc_title}"
    ET.SubElement(wpt_target, "sym").text = "Cross-Hair"
    for s in sightings:
        wpt = ET.SubElement(gpx, "wpt", lat=str(s.get("latitude")), lon=str(s.get("longitude")))
        ET.SubElement(wpt, "name").text = f"Sighting: {s.get('title', 'BFRO Report')}"
        ET.SubElement(wpt, "desc").text = f"Date: {s.get('event_date', 'N/A')} | Summary: {s.get('summary', '')}"
        ET.SubElement(wpt, "sym").text = "Footprint"
    for c in camps:
        wpt = ET.SubElement(gpx, "wpt", lat=str(c.get("latitude")), lon=str(c.get("longitude")))
        ET.SubElement(wpt, "name").text = f"Camp: {c.get('name', 'Campsite')}"
        ET.SubElement(wpt, "sym").text = "Campground"
    for a in audio:
        wpt = ET.SubElement(gpx, "wpt", lat=str(a.get("latitude")), lon=str(a.get("longitude")))
        ET.SubElement(wpt, "name").text = f"Audio: {a.get('event_type', 'Infrasound Log')}"
        ET.SubElement(wpt, "desc").text = str(a.get('notes', '') or '')
        ET.SubElement(wpt, "sym").text = "Sound"
    for log in community_logs:
        wpt = ET.SubElement(gpx, "wpt", lat=str(log.get("latitude")), lon=str(log.get("longitude")))
        ET.SubElement(wpt, "name").text = f"Field Log: {log.get('observation_type', 'Unvetted Log')}"
        ET.SubElement(wpt, "desc").text = f"Facts: {log.get('physical_evidence_notes', '')} | Narrative: {log.get('field_narrative', '')}"
        ET.SubElement(wpt, "sym").text = "Pin"
    return ET.tostring(gpx, encoding="utf-8", method="xml")

def geocode_mapbox(query):
    token = st.secrets.get("MAPBOX_TOKEN", "")
    if not token:
        return None
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(query)}.json"
    try:
        resp = requests.get(url, params={"access_token": token, "limit": 1, "types": "place,locality,address,district,region"}, timeout=5)
        if resp.status_code == 200 and resp.json().get("features"):
            feature = resp.json()["features"][0]
            center = feature["center"]
            place_name = feature.get("place_name", query)
            state, county = "Massachusetts", ""
            for ctx in feature.get("context", []):
                if "region" in ctx.get("id", ""): state = ctx.get("text", state)
                elif "district" in ctx.get("id", ""): county = ctx.get("text", county)
            return center[1], center[0], place_name, state, county
    except Exception:
        pass
    return None

with st.sidebar:
    st.header("⚙️ Field Controls")

    if "researcher_name" not in st.session_state:
        st.session_state.researcher_name = ""
    st.session_state.researcher_name = st.text_input(
        "🧑‍🔬 Researcher Name / Callsign",
        value=st.session_state.researcher_name,
        placeholder="e.g. Max P.",
        help="Type a name once per visit. It gets attached to anything you save below, so you can find your own Field Log entries and Saved Searches later. Not a password -- just a label."
    )
    researcher_name = st.session_state.researcher_name.strip()
    if not researcher_name:
        st.caption("⚠️ Enter a name above to save or view your own Field Log entries and Saved Searches.")

    if not st.secrets.get("MAPBOX_TOKEN", ""):
        st.caption("⚠️ No MAPBOX_TOKEN set in Secrets — area search won't work until this is added. Device GPS still works.")

    loc_search = st.text_input("📍 Target Search Area", value=loc_name)
    radius_miles = st.selectbox("Field Radius (Miles)", [25, 50, 100, 250, 500], index=2, key="radius_miles_key")
    deg_delta = radius_miles / 69.0

    col_s1, col_s2 = st.columns(2)
    if col_s1.button("🔎 Search Area", use_container_width=True) and loc_search:
        with st.spinner("Searching..."):
            res = geocode_mapbox(loc_search)
        if res:
            st.session_state.user_lat, st.session_state.user_lon, st.session_state.location_name, st.session_state.user_state, st.session_state.user_county = res
            st.rerun()
        else:
            st.warning("Search didn't return a result — check MAPBOX_TOKEN in Secrets, or try Device GPS instead.")

    if col_s2.button("📲 Device GPS", use_container_width=True):
        with st.spinner("Getting your location..."):
            loc_data = get_geolocation()
        if loc_data and "coords" in loc_data:
            st.session_state.user_lat = loc_data["coords"]["latitude"]
            st.session_state.user_lon = loc_data["coords"]["longitude"]
            st.session_state.location_name = "Device GPS"
            st.rerun()

    with st.expander("💾 Saved Searches"):
        st.caption("Save this exact location, radius, and layer setup under your name, or jump back to one you saved before (stored locally in data/saved_searches_rows.csv).")
        current_layers = {
            "layer_bfro": st.session_state.get("layer_bfro", True),
            "layer_lore": st.session_state.get("layer_lore", True),
            "layer_news": st.session_state.get("layer_news", True),
            "layer_hotspots": st.session_state.get("layer_hotspots", True),
            "layer_audio": st.session_state.get("layer_audio", True),
            "layer_turbines": st.session_state.get("layer_turbines", True),
            "layer_user_logs": st.session_state.get("layer_user_logs", True),
            "layer_camps": st.session_state.get("layer_camps", True),
        }

        all_saved_searches = load_local_csv_records(SAVED_SEARCHES_CSV_PATH)

        if not researcher_name:
            st.info("Enter a Researcher Name above to save or load searches.")
        else:
            save_label = st.text_input("Label for this search", value=loc_name, key="save_search_label")
            if st.button("Save This Search", use_container_width=True):
                try:
                    existing_ids = [int(s["id"]) for s in all_saved_searches if str(s.get("id", "")).isdigit()]
                    next_id = (max(existing_ids) + 1) if existing_ids else 1
                    all_saved_searches.append({
                        "id": next_id,
                        "created_at": datetime.now().isoformat(),
                        "researcher_name": researcher_name,
                        "label": save_label,
                        "location_name": loc_name,
                        "latitude": lat,
                        "longitude": lon,
                        "radius_miles": radius_miles,
                        "layers": json.dumps(current_layers),
                    })
                    save_local_csv_records(
                        SAVED_SEARCHES_CSV_PATH,
                        all_saved_searches,
                        ["id", "created_at", "researcher_name", "label", "location_name", "latitude", "longitude", "radius_miles", "layers"],
                    )
                    st.success(f"Saved: {save_label}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't save: {e}")

            st.markdown("---")
            my_searches_data = [
                s for s in all_saved_searches
                if str(s.get("researcher_name", "")).strip() == researcher_name
            ]

            if not my_searches_data:
                st.caption("No saved searches yet under this name.")
            else:
                search_labels = {f"{s.get('label', 'Untitled')} ({s.get('id')})": s for s in my_searches_data}
                chosen_label = st.selectbox("Your saved searches:", list(search_labels.keys()), key="load_search_select")
                col_j1, col_j2 = st.columns(2)
                if col_j1.button("Jump to this search", use_container_width=True):
                    saved = search_labels[chosen_label]
                    st.session_state.pending_saved_search = saved
                    st.rerun()
                if col_j2.button("🗑️ Delete", use_container_width=True):
                    saved = search_labels[chosen_label]
                    try:
                        updated_searches = [s for s in all_saved_searches if str(s.get("id")) != str(saved.get("id"))]
                        save_local_csv_records(
                            SAVED_SEARCHES_CSV_PATH,
                            updated_searches,
                            ["id", "created_at", "researcher_name", "label", "location_name", "latitude", "longitude", "radius_miles", "layers"],
                        )
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Couldn't delete: {e}")

    st.markdown("---")
    st.subheader("🗺️ Active Map Layers")
    show_bfro = st.checkbox("1. 👣 Sightings (Dual Footprints)", value=True, key="layer_bfro")
    show_lore = st.checkbox("2. 🪶 Native American Lore Net", value=True, key="layer_lore")
    show_news = st.checkbox("3. 📰 Press Archives Net", value=True, key="layer_news")
    show_hotspots = st.checkbox("4. 🚨 Hot Zones, Refuges & The Larson Hypothesis", value=True, key="layer_hotspots")
    show_audio = st.checkbox("5. 🔊 Infrasound / Acoustic Masking", value=True, key="layer_audio")
    show_turbines = st.checkbox("6. 🌀 Wind Turbine Farm Clusters", value=True, key="layer_turbines")
    show_user_logs = st.checkbox("7. ⚠️ Community Field Logs", value=True, key="layer_user_logs")
    show_camps = st.checkbox("8. 🏕️ Camping & Access Points", value=True, key="layer_camps")

PARAM_DEFAULTS = {
    "param_hotzone_radius_mi": 15.0,
    "param_report_gap_mi": 10.0,
    "param_esi_threshold": 0.55,
    "param_corridor_max_mi
