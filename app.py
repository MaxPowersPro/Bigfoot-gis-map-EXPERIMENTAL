import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import sqlite3
from streamlit_js_eval import get_geolocation
import random
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import numpy as np
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

# --- Data Loaders ---
def load_bfro_clean_data(path="data/bfro_reports_clean.csv"):
    import pandas as pd
    import os
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

@st.cache_data(ttl=86400)
def load_bfro_renner_data(local_path="data/bfro_reports_geocoded.csv"):
    import pandas as pd
    import os

    fallback_urls = [
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-11-15/bfro_reports_geocoded.csv",
        "https://raw.githubusercontent.com/timothyrenner/bfro_sightings_data/master/bfro_reports_geocoded.csv",
    ]

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
        for url in fallback_urls:
            try:
                df = pd.read_csv(url)
                if df is not None and not df.empty:
                    break
            except Exception:
                continue

    if df is None or df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        lat_val = row.get("latitude")
        lon_val = row.get("longitude")
        if pd.isna(lat_val) or pd.isna(lon_val):
            continue
        raw_num = row.get("number", row.get("report_id", ""))
        rep_id = str(int(float(raw_num))) if pd.notna(raw_num) and str(raw_num).replace(".", "", 1).isdigit() else str(raw_num)
        title_val = str(row.get("title", "")) if pd.notna(row.get("title")) else f"BFRO Report #{rep_id}"
        observed_val = str(row.get("observed", row.get("summary", ""))) if pd.notna(row.get("observed", row.get("summary", ""))) else ""
        date_val = str(row.get("date", row.get("event_date", "N/A"))) if pd.notna(row.get("date", row.get("event_date", "N/A"))) else "N/A"
        class_val = str(row.get("classification", row.get("class_rating", "Unclassified"))) if pd.notna(row.get("classification", row.get("class_rating", "Unclassified"))) else "Unclassified"
        county_val = str(row.get("county", "")) if pd.notna(row.get("county")) else ""

        records.append({
            "id": rep_id,
            "report_id": rep_id,
            "title": title_val,
            "summary": observed_val,
            "latitude": float(lat_val),
            "longitude": float(lon_val),
            "event_date": date_val,
            "class_rating": class_val,
            "county": county_val,
            "source": "BFRO (Geocoded Archive)",
            "real_terrain_roughness": None,
        })
    return records

raw_sightings_bfro = load_bfro_clean_data()
existing_bfro_ids = {str(r.get("report_id", "")).strip() for r in raw_sightings_bfro if str(r.get("report_id", "")).strip()}
for r_rec in load_bfro_renner_data():
    r_id = str(r_rec.get("report_id", "")).strip()
    if r_id and r_id not in existing_bfro_ids:
        raw_sightings_bfro.append(r_rec)
        existing_bfro_ids.add(r_id)

def load_junk_drawer_data(path="data/junk_drawer.csv"):
    import pandas as pd
    import os
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
    import pandas as pd
    import os
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
    import pandas as pd
    import os
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

raw_sightings_john_green = load_john_green_data()
raw_sightings = raw_sightings_bfro + raw_sightings_john_green
raw_lore = load_and_standardize_dataset("data/indigenous_lore.csv")
raw_news = load_and_standardize_dataset("data/press_archives.csv")
raw_camps = load_and_standardize_dataset("data/campsites.csv")
raw_junk = load_junk_drawer_data()
all_junk_records = raw_junk

raw_researcher_archives = load_researcher_archives_data()
KRANTZ_PDF_URL = "https://raw.githubusercontent.com/MaxPowersPro/Bigfoot-gis-map/main/data/krantz_finding_aid.pdf"

# Live USGS Wind Turbine API Loader
@st.cache_data(ttl=3600)
def fetch_usgs_wind_turbines(target_lat, target_lon, radius_miles):
    deg_r = radius_miles / 69.0
    min_lat, max_lat = target_lat - deg_r, target_lat + deg_r
    min_lon, max_lon = target_lon - deg_r, target_lon + deg_r

    url = f"https://energy.usgs.gov/api/uswtdb/v1/turbines?ylat=gte.{min_lat}&ylat=lte.{max_lat}&xlong=gte.{min_lon}&xlong=lte.{max_lon}"
    turbines = []
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            for row in data:
                t_lat, t_lon = float(row.get("ylat", 0)), float(row.get("xlong", 0))
                dist = haversine_miles(target_lat, target_lon, t_lat, t_lon)
                if dist <= radius_miles:
                    turbines.append({
                        "id": row.get("case_id"),
                        "project_name": row.get("p_name", "Wind Energy Project"),
                        "latitude": t_lat,
                        "longitude": t_lon,
                        "hub_height_m": row.get("t_hh", "N/A"),
                        "capacity_kw": row.get("t_cap", "N/A"),
                        "dist_miles": dist,
                        "felt_radius_miles": 1.5,
                        "prop_radius_miles": 6.0
                    })
    except Exception:
        pass
    return turbines

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
    st.session_state.user_lat = _pending["latitude"]
    st.session_state.user_lon = _pending["longitude"]
    st.session_state.location_name = _pending.get("location_name", "Loaded Search")
    st.session_state.radius_miles_key = _pending.get("radius_miles", 100)
    for _layer_key, _layer_val in (_pending.get("layers") or {}).items():
        st.session_state[_layer_key] = _layer_val

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
        * **👣📜 Footprint Markers:** Blue = standard biological sighting. Purple = report text includes documented high-strangeness or sensory dread language.
        * **🚨 Red Dotted Rings:** Hot Zones where reports, wildlife density, cover, and water overlap.
        * **🪹 Orange Dotted Rings:** Predictive Refuges — high inferred habitat quality with no direct sightings nearby.
        * **🌲 Green Channels:** Larson transit corridors (≤20 miles) connecting nearby Hot Zones along natural terrain gaps.
        * **🔊 Purple Marker:** Infrasound source (waterfalls, dams, wind gaps). Bold inner ring = felt zone; light outer ring = detectable zone.
        * **💨 Wind Turbines (Merged Purple Zone):** Single unified felt-zone footprint (~1.5 mi) with one central turbine marker and USGS data link.
        * **🏕️ Green Campgrounds:** Dispersed campsites and backcountry staging points.
        """)
    with col_g3:
        st.markdown("#### ⚙️ Sidebar Layer Toggles")
        st.write("""
        * Use the **8 Active Map Layers** in the sidebar to show or hide specific data overlays in real time.
        * Turn layers off to declutter dense search sectors when analyzing high-density sighting clusters.
        * **📝 Submitting a Field Log:** Enter a Researcher Name in the sidebar first, then use the "Submit Investigator Field Log" drawer.
        """)
    st.markdown("---")

# --- Local SQLite Database ---
@st.cache_resource
def init_local_db():
    conn = sqlite3.connect("field_database.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_name TEXT,
            label TEXT,
            location_name TEXT,
            latitude REAL,
            longitude REAL,
            radius_miles INTEGER,
            layers TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS acoustic_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            latitude REAL,
            longitude REAL,
            notes TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS investigator_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_by TEXT,
            is_public INTEGER,
            observation_type TEXT,
            event_date TEXT,
            latitude REAL,
            longitude REAL,
            physical_evidence_notes TEXT,
            field_narrative TEXT,
            ethics_agreed INTEGER
        )
    """)
    conn.commit()
    return conn

db_conn = init_local_db()

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

def generate_tone_wav(freq_hz=12.0, duration_sec=3.0, sample_rate=8000):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Audible carrier modulated at the low infrasound pulse frequency so standard speakers can demonstrate the rhythm
    carrier = np.sin(2 * np.pi * 110.0 * t)
    modulator = 0.5 * (1.0 + np.sin(2 * np.pi * freq_hz * t))
    signal = (carrier * modulator * 0.6 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())
    buf.seek(0)
    return buf.read()

# --- Infrasound Master Categories ---
INFRASOUND_TYPES = {
    "waterfall": {"label": "🌊 Waterfall / Hydrological", "cause": "High-volume water impact at waterfalls, rapids, or river spillways.", "freq_range": "3 – 15 Hz", "character": "Continuous rumble.", "felt_miles": 12, "travel_miles": 80, "effect": "Falls mostly in the vestibular resonance band (1-7 Hz) — dizziness, pressure headaches, and loss of balance are the most commonly reported effects within the felt zone."},
    "wind": {"label": "🌬️ Wind / Mountain Pass", "cause": "High-velocity wind funneling through narrow ridge gaps, saddles, or mountain passes.", "freq_range": "0.5 – 5 Hz", "character": "Continuous, weather- and pressure-system dependent.", "felt_miles": 8, "travel_miles": 55, "effect": "Sits at the low end of the vestibular band — dizziness and disorientation are most likely, especially during active frontal weather passing through the notch."},
    "manmade": {"label": "⚙️ Man-Made (Dam / Mine)", "cause": "Dam spillway turbulence (continuous hum) or mine/quarry blasting (short, high-amplitude pulses).", "freq_range": "0.5 – 15 Hz", "character": "Dams: steady hum. Mines/quarries: sharp, punctuated blast pulses, documented detectable 10+ miles out.", "felt_miles": 8, "travel_miles": 60, "effect": "Mine blasts can produce a sudden felt pressure jolt rather than a gradual sensation; dam hum tends to produce the same dizziness/dread pattern as natural hydrological sources."},
    "biological": {"label": "🦍 Biological", "cause": "Large thoracic resonance structures and specialized laryngeal folds in large-bodied animals producing sub-audible vocal bursts.", "freq_range": "8 – 18 Hz", "character": "Vocal burst, not continuous.", "felt_miles": 3, "travel_miles": 8, "effect": "Overlaps the human alpha-wave / 'being watched' band (7-12 Hz) most directly, but the felt zone is small since source amplitude is far below geological or man-made sources."},
    "turbine": {"label": "💨 Wind Turbine (Industrial)", "cause": "Blade passage past vertical support towers, generating continuous low-frequency pressure pulses.", "freq_range": "0.5 – 20 Hz", "character": "Periodic pulse at blade-pass frequency (~0.5–2.5 Hz).", "felt_miles": 1.5, "travel_miles": 6.0, "effect": "Vestibular disturbance: lightheadedness, nausea, sensation of ear pressure, and low-level autonomic arousal (hyper-vigilance/restlessness/ghost-frequency overlap) in sensitive individuals within the felt zone (0.5–1.5 mi)."},
}

RANGE_BASED_BIOLOGICAL_SPECIES = {
    "American Alligator": {
        "states": {
            "Texas", "Oklahoma", "Louisiana", "Arkansas", "Mississippi",
            "Alabama", "Georgia", "Florida", "South Carolina", "North Carolina",
        },
        "frequency": "16–25 Hz",
        "description": "Male American alligators produce infrasound (16-25 Hz) during breeding-season bellows; felt from 50+ ft away.",
        "citation": "Range: USFWS Federal Register (2021). Frequency: Vliet, Univ. of Florida.",
    },
    "Ruffed Grouse": {
        "states": {
            "Maine", "New Hampshire", "Vermont", "Massachusetts", "Connecticut",
            "New York", "Pennsylvania", "New Jersey", "Virginia", "West Virginia",
            "Ohio", "Maryland", "North Carolina", "Tennessee", "Kentucky", "Georgia",
            "Michigan", "Wisconsin", "Minnesota", "Illinois", "Indiana", "Iowa",
            "South Dakota", "North Dakota", "Montana", "Idaho", "Wyoming", "Utah",
            "Washington", "Oregon", "Alaska",
        },
        "frequency": "~45 Hz peak (partial true infrasound per Freeman 2012)",
        "description": "Male ruffed grouse perform a wing-beat 'drumming' display fast enough to create a small sonic boom; often felt as much as heard.",
        "citation": "Range: Ruffed Grouse Society. Frequency: Garcia et al. 2012; Freeman 2012.",
    },
}

BIO_SPECIES_LISTEN_LINKS = {
    "Ruffed Grouse": "https://search.macaulaylibrary.org/catalog?taxonCode=rufgro",
    "American Alligator": "https://search.macaulaylibrary.org/catalog?taxonCode=ameall",
}

def classify_infrasound_type(event_type_str):
    s = str(event_type_str).lower()
    if any(k in s for k in ["waterfall", "falls", "rapid", "hydro", "niagara", "snoqualmie"]):
        return "waterfall"
    if any(k in s for k in ["wind", "pass", "ridge", "saddle", "gap", "aeolian"]):
        return "wind"
    if any(k in s for k in ["dam", "mine", "quarry", "blast"]):
        return "manmade"
    if any(k in s for k in ["biotic", "biological", "animal", "vocal", "call"]):
        return "biological"
    if any(k in s for k in ["turbine", "windmill", "wind farm", "wind energy"]):
        return "turbine"
    return None

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
    "being watched", "watched me", "sudden dread", "overwhelming dread", "paralyzed with fear",
    "frozen in fear", "unexplained panic", "shadow figure", "felt a presence", "sense of dread",
]

def scan_paranormal_content(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in PARANORMAL_KEYWORDS)

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
        ET.SubElement(wpt, "desc").text = a.get('notes', '')
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
        help="Type a name once per visit. It gets attached to anything you save below."
    )
    researcher_name = st.session_state.researcher_name.strip()
    if not researcher_name:
        st.caption("⚠️ Enter a name above to save or view your own Field Log entries and Saved Searches.")

    if not st.secrets.get("MAPBOX_TOKEN", ""):
        st.caption("⚠️ No MAPBOX_TOKEN set in Secrets — area search won't work until this is added.")

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
            st.warning("Search didn't return a result — check MAPBOX_TOKEN in Secrets.")

    if col_s2.button("📲 Device GPS", use_container_width=True):
        with st.spinner("Getting your location..."):
            loc_data = get_geolocation()
        if loc_data and "coords" in loc_data:
            st.session_state.user_lat = loc_data["coords"]["latitude"]
            st.session_state.user_lon = loc_data["coords"]["longitude"]
            st.session_state.location_name = "Device GPS"
            st.rerun()

    with st.expander("💾 Saved Searches"):
        st.caption("Save this exact location, radius, and layer setup under your name.")
        current_layers = {
            "layer_bfro": st.session_state.get("layer_bfro", True),
            "layer_lore": st.session_state.get("layer_lore", True),
            "layer_news": st.session_state.get("layer_news", True),
            "layer_hotspots": st.session_state.get("layer_hotspots", True),
            "layer_audio": st.session_state.get("layer_audio", True),
            "layer_user_logs": st.session_state.get("layer_user_logs", True),
            "layer_camps": st.session_state.get("layer_camps", True),
            "layer_turbines": st.session_state.get("layer_turbines", True),
        }

        if not researcher_name:
            st.info("Enter a Researcher Name above to save or load searches.")
        else:
            save_label = st.text_input("Label for this search", value=loc_name, key="save_search_label")
            if st.button("Save This Search", use_container_width=True):
                with st.spinner("Saving..."):
                    try:
                        cursor = db_conn.cursor()
                        cursor.execute("""
                            INSERT INTO saved_searches 
                            (researcher_name, label, location_name, latitude, longitude, radius_miles, layers)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (researcher_name, save_label, loc_name, lat, lon, radius_miles, json.dumps(current_layers)))
                        db_conn.commit()
                        st.success(f"Saved: {save_label}")
                    except Exception as e:
                        st.error(f"Couldn't save: {e}")

            st.markdown("---")
            try:
                cursor = db_conn.cursor()
                cursor.execute("SELECT * FROM saved_searches WHERE researcher_name = ?", (researcher_name,))
                rows = cursor.fetchall()
                my_searches_data = []
                for row in rows:
                    r_dict = dict(row)
                    if isinstance(r_dict.get("layers"), str):
                        r_dict["layers"] = json.loads(r_dict["layers"])
                    my_searches_data.append(r_dict)
            except Exception as e:
                my_searches_data = []
                st.caption(f"Couldn't load saved searches: {e}")

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
                        cursor = db_conn.cursor()
                        cursor.execute("DELETE FROM saved_searches WHERE id = ?", (saved["id"],))
                        db_conn.commit()
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
    show_user_logs = st.checkbox("6. ⚠️ Community Field Logs", value=True, key="layer_user_logs")
    show_camps = st.checkbox("7. 🏕️ Camping & Access Points", value=True, key="layer_camps")
    show_turbines = st.checkbox("8. 💨 Wind Turbines (USGS Database)", value=True, key="layer_turbines")

PARAM_DEFAULTS = {
    "param_hotzone_radius_mi": 15.0,
    "param_report_gap_mi": 10.0,
    "param_esi_threshold": 0.55,
    "param_corridor_max_mi": 20.0,
    "param_effort_k": 0.5,
}
for _pk, _pv in PARAM_DEFAULTS.items():
    if _pk not in st.session_state:
        st.session_state[_pk] = _pv

sightings_data, camps_data, audio_data, media_data, lore_data, user_logs_data = [], [], [], [], [], []
seasonal_breakdown = {}

if raw_sightings:
    for s in raw_sightings:
        s_lat, s_lon = s.get("latitude"), s.get("longitude")
        if s_lat is not None and s_lon is not None:
            if haversine_miles(lat, lon, float(s_lat), float(s_lon)) <= radius_miles:
                event_d_str = s.get('event_date', 'N/A')
                season = get_season(event_d_str)
                seasonal_breakdown[season] = seasonal_breakdown.get(season, 0) + 1
                try: ev_month = int(str(event_d_str).split('-')[1])
                except Exception: ev_month = 6

                s_dist_road = float(s.get("dist_to_road_miles", 0.4))
                s_pop_density = float(s.get("pop_density_sq_mi", 45.0))
                eff_factor = calculate_human_effort_factor(s_dist_road, s_pop_density)
                has_physical = bool(s.get("has_tracks") or s.get("has_hair") or s.get("has_physical_evidence"))
                class_rat = s.get("class_rating", "Class A")
                weight_dict = calculate_adjusted_evidence_weight(class_rat, has_physical, eff_factor, k=st.session_state.param_effort_k)
                s["effort_factor"] = eff_factor
                s["evidence_weight"] = weight_dict["final_weight"]
                s["base_weight"] = weight_dict["base_weight"]
                s["audit_explanation"] = weight_dict["audit_explanation"]

                sc_index = calculate_seasonal_cover_index(ev_month, 0.4, 0.5, True)
                if s.get("real_terrain_roughness") is not None:
                    terrain_input = s["real_terrain_roughness"]
                    s["terrain_data_source"] = "real (USGS elevation-derived)"
                else:
                    terrain_input = 0.6
                    s["terrain_data_source"] = "placeholder (not yet farmed for this report)"
                s["esi_score"] = calculate_environmental_suitability_index(sc_index, 0.3, terrain_input, 0.7)
                sightings_data.append(s)

all_lore_records = [item.get("metadata", item) for item in raw_lore] if raw_lore else []
all_press_records = [item.get("metadata", item) for item in raw_news] if raw_news else []

if raw_lore:
    for item in raw_lore:
        record = item.get("metadata", item)
        l_lat = float(record.get("latitude", lat))
        l_lon = float(record.get("longitude", lon))
        entry_radius = float(record.get("relevance_radius_miles", 100))
        if haversine_miles(lat, lon, l_lat, l_lon) <= entry_radius:
            lore_data.append(record)

if raw_news:
    for item in raw_news:
        record = item.get("metadata", item)
        m_lat = float(record.get("latitude", lat))
        m_lon = float(record.get("longitude", lon))
        entry_radius = float(record.get("relevance_radius_miles", 30))
        if haversine_miles(lat, lon, m_lat, m_lon) <= entry_radius:
            media_data.append(record)

if raw_camps:
    for item in raw_camps:
        record = item.get("metadata", item)
        c_lat = float(record.get("latitude", lat))
        c_lon = float(record.get("longitude", lon))
        if haversine_miles(lat, lon, c_lat, c_lon) <= radius_miles:
            camps_data.append(record)

try:
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM acoustic_reports")
    for a in [dict(r) for r in cursor.fetchall()]:
        a_lat, a_lon = float(a["latitude"]), float(a["longitude"])
        e_type = a.get("event_type", "")
        itype = classify_infrasound_type(e_type)
        type_info = INFRASOUND_TYPES.get(itype)
        prop_radius = type_info["travel_miles"] if type_info else 45
        felt_radius = type_info["felt_miles"] if type_info else 8
        a["infrasound_type"] = itype
        a["prop_radius_miles"] = prop_radius
        a["felt_radius_miles"] = felt_radius
        dist_to_target = haversine_miles(lat, lon, a_lat, a_lon)
        a["dist_to_target"] = dist_to_target
        if dist_to_target <= (radius_miles + prop_radius):
            overlap_dist = (radius_miles + prop_radius) - dist_to_target
            a["coverage_pct"] = max(10, min(100, int((overlap_dist / (radius_miles * 2)) * 100)))
            a["is_offscreen"] = dist_to_target > radius_miles
            audio_data.append(a)
except Exception:
    pass

try:
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM investigator_logs")
    for log in [dict(r) for r in cursor.fetchall()]:
        is_public = bool(log.get("is_public"))
        is_mine = bool(researcher_name) and log.get("submitted_by") == researcher_name
        if not (is_public or is_mine):
            continue
        if haversine_miles(lat, lon, float(log["latitude"]), float(log["longitude"])) <= radius_miles:
            user_logs_data.append(log)
except Exception:
    pass

turbines_in_sector = fetch_usgs_wind_turbines(lat, lon, radius_miles) if show_turbines else []

# Cross-reference Sightings & High-Strangeness Reports against ALL Infrasound Felt Zones (including Wind Turbines)
woo_total_in_sector = 0
woo_infrasound_overlap = 0

for s in sightings_data:
    s["nearby_infrasound"] = None
    s_lat, s_lon = float(s["latitude"]), float(s["longitude"])

    for a in audio_data:
        dist = haversine_miles(s_lat, s_lon, float(a["latitude"]), float(a["longitude"]))
        if dist <= a.get("felt_radius_miles", 8):
            itype_key = a.get("infrasound_type")
            t_meta = INFRASOUND_TYPES.get(itype_key, {})
            s["nearby_infrasound"] = {
                "label": t_meta.get("label", "an unclassified infrasound source"),
                "freq_range": t_meta.get("freq_range", "0.5 – 20 Hz"),
                "event_type": a.get("event_type", ""),
                "dist_miles": dist,
            }
            break

    if not s["nearby_infrasound"] and turbines_in_sector:
        for t in turbines_in_sector:
            dist = haversine_miles(s_lat, s_lon, t["latitude"], t["longitude"])
            if dist <= t["felt_radius_miles"]:
                t_meta = INFRASOUND_TYPES["turbine"]
                s["nearby_infrasound"] = {
                    "label": f"{t_meta['label']} ({t['project_name']})",
                    "freq_range": t_meta["freq_range"],
                    "event_type": t["project_name"],
                    "dist_miles": dist,
                }
                break

    s["is_anomalous"] = scan_paranormal_content(f"{s.get('title', '')} {s.get('summary', '')}")
    if s["is_anomalous"]:
        woo_total_in_sector += 1
        if s["nearby_infrasound"]:
            woo_infrasound_overlap += 1

applicable_bio_species = [
    (name, info) for name, info in RANGE_BASED_BIOLOGICAL_SPECIES.items()
    if active_state in info["states"]
]

st.markdown(f"""
<div style="background-color:#1e272c; color:white; padding:10px 14px; border-radius:6px; margin-bottom:12px; font-size:14px; border-left:4px solid #e74c3c;">
<b>📍 Active Sector Records ({loc_name} • {active_state}):</b> &nbsp;
👣 Sightings: <b><code>{len(sightings_data)}</code></b> &nbsp;|&nbsp;
🪶 Regional Lore: <b><code>{len(lore_data)}</code></b> &nbsp;|&nbsp;
📰 Press Archives: <b><code>{len(media_data)}</code></b> &nbsp;|&nbsp;
🔊 Infrasound Waves: <b><code>{len(audio_data)}</code></b> &nbsp;|&nbsp;
💨 Turbines: <b><code>{len(turbines_in_sector)}</code></b> &nbsp;|&nbsp;
🏕️ Campsites: <b><code>{len(camps_data)}</code></b>
</div>
""", unsafe_allow_html=True)

# --- Folium Map ---
m = folium.Map(location=[lat, lon], zoom_start=8, tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap")
folium.Circle(radius=radius_miles * 1609.34, location=[lat, lon], color="#e74c3c", weight=2, fill=True, fill_opacity=0.02).add_to(m)
folium.Marker([lat, lon], popup=f"<b>📍 TARGET CENTER: {loc_name}</b>", icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(m)

if show_bfro and sightings_data:
    for s in sightings_data:
        j_lat, j_lon = apply_jitter(s["latitude"], s["longitude"], offset_seed=1)
        raw_summary = s.get("summary", "No transcript summary provided.")
        raw_id = str(s.get('report_id', s.get('id', ''))).strip()
        source_label = s.get("source", "BFRO")
        link_html = f'<br><a href="https://www.bfro.net/GDB/show_report.asp?id={raw_id}" target="_blank" style="display:inline-block; margin-top:4px; padding:3px 6px; background:#007bff; color:white; border-radius:3px; text-decoration:none; font-size:10px;">📄 Direct BFRO Report #{raw_id}</a>' if raw_id.isdigit() else ''

        is_anomalous = s.get("is_anomalous", False)
        pin_color = "#8e44ad" if is_anomalous else "#2b78e4"
        source_icon = "📜" if source_label != "BFRO" else "👣"

        infrasound_note = ""
        if s.get("nearby_infrasound"):
            ni = s["nearby_infrasound"]
            if is_anomalous:
                infrasound_note = f"""<div style="margin-top:4px; padding:4px; background:#f4f0fa; border-left:3px solid #8e44ad; font-size:9px;">👻/🔊 <b>Infrasound Cross-Reference:</b> sits <b>{ni['dist_miles']:.1f} mi</b> inside the felt zone of {ni['label']} ({ni['freq_range']}, overlapping the 7–19 Hz 'ghost / being-watched' band).</div>"""
            else:
                infrasound_note = f"""<div style="margin-top:4px; padding:4px; background:#f4f0fa; border-left:3px solid #8e44ad; font-size:9px;">🔊 <b>Natural explanation check:</b> within the felt zone of {ni['label']} ({ni['dist_miles']:.1f} mi) — see Local Intel.</div>"""

        anomalous_note = ""
        if is_anomalous:
            anomalous_note = """<div style="margin-top:4px; padding:4px; background:#f4f0fa; border-left:3px solid #8e44ad; font-size:9px;">👻 <b>Content flag (conjecture, not fact):</b> report text includes documented high-strangeness or sensory dread language.</div>"""

        terrain_source = s.get("terrain_data_source", "placeholder (not yet farmed for this report)")
        terrain_icon = "🟢" if "real" in terrain_source else "⚪"
        terrain_note = f"""<div style="margin-top:2px; font-size:8px; color:#888;">{terrain_icon} ESI terrain input: {terrain_source}</div>"""

        popup_html = f"""
        <div style="font-family:sans-serif; width:260px;">
        <b style="color:{pin_color};">{source_icon} {s.get('title', 'Sighting Report')}</b><br>
        <small><b>Source:</b> {source_label} | <b>Class:</b> {s.get('class_rating', 'Class A')} | <b>Weight:</b> {s.get('evidence_weight', 1.0)}x</small><br>
        <hr style="margin:4px 0;">
        <p style="font-size:10px; margin:2px 0; background:#f8f9fa; padding:4px;">{raw_summary[:160]}...</p>
        {anomalous_note}
        {infrasound_note}
        {terrain_note}
        {link_html}
        </div>
        """
        footprint_svg = f"""<svg width="18" height="18" viewBox="0 0 24 24" style="filter: drop-shadow(0 0 1px white);"><ellipse cx="12" cy="15" rx="4.2" ry="8.5" fill="{pin_color}"/><ellipse cx="7.5" cy="5" rx="1.3" ry="2.1" fill="{pin_color}"/><ellipse cx="10.5" cy="3.2" rx="1.4" ry="2.3" fill="{pin_color}"/><ellipse cx="13.7" cy="3.5" rx="1.3" ry="2.2" fill="{pin_color}"/><ellipse cx="16.5" cy="5" rx="1.2" ry="1.9" fill="{pin_color}"/><ellipse cx="18.7" cy="7.5" rx="1.0" ry="1.5" fill="{pin_color}"/></svg>"""
        folium.Marker([j_lat, j_lon], popup=folium.Popup(popup_html, max_width=280), icon=folium.DivIcon(html=footprint_svg, icon_size=(20, 20), icon_anchor=(10, 16))).add_to(m)

if show_lore and lore_data:
    for l_rec in lore_data:
        l_lat, l_lon = float(l_rec.get("latitude", lat)), float(l_rec.get("longitude", lon))
        l_popup = f"""
        <div style="font-family:sans-serif; width:240px;">
        <b>🪶 {l_rec.get('name', l_rec.get('title', 'Indigenous Lore'))}</b><br>
        <small><b>Nation/Region:</b> {l_rec.get('tribe', l_rec.get('region', 'Regional'))}</small><br>
        <p style="font-size:10px; margin:4px 0;">{str(l_rec.get('summary', l_rec.get('description', '')))[:180]}...</p>
        </div>
        """
        folium.Marker([l_lat, l_lon], popup=folium.Popup(l_popup, max_width=260), icon=folium.Icon(color="darkred", icon="feather", prefix="fa")).add_to(m)

if show_news and media_data:
    for m_rec in media_data:
        m_lat, m_lon = float(m_rec.get("latitude", lat)), float(m_rec.get("longitude", lon))
        m_popup = f"""
        <div style="font-family:sans-serif; width:240px;">
        <b>📰 {m_rec.get('title', 'Historical Press Clipping')}</b><br>
        <small><b>Date:</b> {m_rec.get('date', 'N/A')} | <b>Publication:</b> {m_rec.get('publication', 'Archive')}</small><br>
        <p style="font-size:10px; margin:4px 0;">{str(m_rec.get('summary', ''))[:180]}...</p>
        </div>
        """
        folium.Marker([m_lat, m_lon], popup=folium.Popup(m_popup, max_width=260), icon=folium.Icon(color="lightgray", icon="newspaper", prefix="fa")).add_to(m)

if show_camps and camps_data:
    for c in camps_data:
        c_popup = f"<b>🏕️ {c.get('name', 'Campsite')}</b><br><small>Type: {c.get('type', 'Primitive / Dispersed')}</small>"
        folium.Marker([c["latitude"], c["longitude"]], popup=c_popup, icon=folium.Icon(color="green", icon="campground", prefix="fa")).add_to(m)

if show_audio and audio_data:
    for a in audio_data:
        type_label = INFRASOUND_TYPES.get(a.get("infrasound_type"), {}).get("label", "Unclassified Source")
        felt_m = a.get("felt_radius_miles", 8) * 1609.34
        travel_m = a["prop_radius_miles"] * 1609.34
        off_str = f"<br><b style='color:#d35400;'>⚠️ Trans-Boundary Source: {int(a['dist_to_target'])} miles away ({a['coverage_pct']}% local sector coverage)</b>" if a.get("is_offscreen") else ""
        a_popup = f"""<b>🔊 {type_label}</b><br><b>{a.get('event_type')}</b><br><small>Felt zone: ~{a.get('felt_radius_miles', 8)} mi | Detectable to: ~{a['prop_radius_miles']} mi</small>{off_str}"""
        if not a.get("is_offscreen"):
            folium.Marker([a["latitude"], a["longitude"]], popup=a_popup, icon=folium.Icon(color="purple", icon="volume-up", prefix="fa")).add_to(m)
        folium.Circle(radius=travel_m, location=[a["latitude"], a["longitude"]], color="#8e44ad", weight=1, dash_array="4, 6", fill=False, popup=f"Detectable footprint (~{a['prop_radius_miles']} mi)").add_to(m)
        folium.Circle(radius=felt_m, location=[a["latitude"], a["longitude"]], color="#6c3483", weight=1.5, fill=True, fill_color="#8e44ad", fill_opacity=0.08, popup=f"Felt zone (~{a.get('felt_radius_miles', 8)} mi)").add_to(m)

# Layer 8: USGS Wind Turbines — Merged Amorphous Purple Shape + Single Icon per Cluster
if show_turbines and turbines_in_sector:
    felt_deg = 1.5 / 69.0
    turbine_buffers = []
    for t in turbines_in_sector:
        # Approximate metric buffer in degrees accounting for latitude cosine scaling
        lat_rad = np.radians(t["latitude"])
        pt = Point(t["longitude"] * np.cos(lat_rad), t["latitude"])
        turbine_buffers.append((pt.buffer(felt_deg), t))

    merged_geom = unary_union([b[0] for b in turbine_buffers])
    if isinstance(merged_geom, Polygon):
        cluster_polys = [merged_geom]
    elif isinstance(merged_geom, MultiPolygon):
        cluster_polys = list(merged_geom.geoms)
    else:
        cluster_polys = []

    for poly in cluster_polys:
        # Match turbines inside this merged amorphous polygon
        cluster_turbines = [t for buf, t in turbine_buffers if poly.intersects(buf)]
        project_names = sorted({t["project_name"] for t in cluster_turbines if t.get("project_name")})
        proj_title = ", ".join(project_names[:2]) + (f" (+{len(project_names)-2} more)" if len(project_names) > 2 else "")
        if not proj_title:
            proj_title = "Regional Wind Energy Cluster"

        centroid = poly.centroid
        c_lat = centroid.y
        c_lon = centroid.x / max(0.1, np.cos(np.radians(c_lat)))

        # Unscale exterior coordinates back to standard (lat, lon) pairs for Folium
        exterior_coords = [
            [y, x / max(0.1, np.cos(np.radians(y)))]
            for x, y in poly.exterior.coords
        ]

        cluster_popup_html = f"""
        <div style="font-family:sans-serif; width:270px;">
        <b style="color:#6c3483;">💨 Industrial Wind Turbine Infrasound Zone</b><br>
        <small><b>Facility:</b> {proj_title} ({len(cluster_turbines)} turbines)</small>
        <hr style="margin:5px 0;">
        <p style="font-size:10px; margin:3px 0;">
        <b>What & Why:</b> Rotating turbine blades generate periodic low-frequency pressure pulses (<b>0.5–20 Hz</b>) every time a blade passes the tower. Within this <b>~1.5-mile felt zone</b>, overlapping waves can trigger inner-ear pressure, mild dizziness, restlessness, or a sense of unease ("ghost frequency" 7–19 Hz overlap) that may be mistaken for an unexplained presence.
        </p>
        <a href="https://energy.usgs.gov/uswtdb/" target="_blank" style="display:inline-block; margin-top:4px; padding:4px 7px; background:#6c3483; color:white; border-radius:3px; text-decoration:none; font-size:10px;">
        🌐 Verify on USGS Wind Turbine Database
        </a>
        </div>
        """

        # Single smooth, amorphous purple polygon (zero overlapping fill darkening)
        folium.Polygon(
            locations=exterior_coords,
            color="#6c3483",
            weight=1.5,
            fill=True,
            fill_color="#8e44ad",
            fill_opacity=0.14,
            popup=folium.Popup(cluster_popup_html, max_width=290)
        ).add_to(m)

        # Single clean turbine icon at the center of the merged purple zone
        folium.Marker(
            location=[c_lat, c_lon],
            popup=folium.Popup(cluster_popup_html, max_width=290),
            icon=folium.Icon(color="cadetblue", icon="fan", prefix="fa")
        ).add_to(m)

if show_user_logs and user_logs_data:
    for ulog in user_logs_data:
        has_facts = bool(ulog.get('physical_evidence_notes'))
        facts_html = f"<p style='font-size:10px; margin:4px 0;'><b>Hard Facts:</b> {ulog.get('physical_evidence_notes')}</p>" if ulog.get('physical_evidence_notes') else ""
        narrative_html = f"<p style='font-size:10px; margin:4px 0;'><b>Narrative:</b> {ulog.get('field_narrative')}</p>" if ulog.get('field_narrative') else ""
        log_popup = f"""
        <div style="font-family:sans-serif; width:240px;">
        <b>📝 FIELD LOG</b><br><small>Type: {ulog.get('observation_type')}</small>
        {facts_html}
        {narrative_html}
        </div>
        """
        folium.Marker([ulog["latitude"], ulog["longitude"]], popup=folium.Popup(log_popup, max_width=280), icon=folium.Icon(color="green" if has_facts else "orange", icon="clipboard", prefix="fa")).add_to(m)

ground_truth_hubs, predictive_refuges = [], []
larson_corridor_count = 0
combined_evidence_points = [
    {"lat": float(s["latitude"]), "lon": float(s["longitude"]), "weight": float(s.get("evidence_weight", 1.0)), "esi": float(s.get("esi_score", 0.5))}
    for s in sightings_data if not filter_urban(float(s["latitude"]), float(s["longitude"]))
]

if show_hotspots and combined_evidence_points:
    coords_arr = np.array([[pt["lat"], pt["lon"]] for pt in combined_evidence_points])
    weights_arr = np.array([pt["weight"] for pt in combined_evidence_points])
    dist_matrix = np.sqrt(((coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]) ** 2).sum(axis=-1))
    visited = set()
    RADIUS_DEG = st.session_state.param_hotzone_radius_mi / 69.0

    for i, pt in enumerate(coords_arr):
        if i in visited: continue
        neighbors = np.where(dist_matrix[i] < RADIUS_DEG)[0]
        if len(neighbors) >= 1:
            ground_truth_hubs.append({
                "lat": np.average(coords_arr[neighbors, 0], weights=weights_arr[neighbors]),
                "lon": np.average(coords_arr[neighbors, 1], weights=weights_arr[neighbors]),
                "weight": float(np.sum(weights_arr[neighbors])),
                "count": len(neighbors)
            })
            visited.update(neighbors)

    for hub in ground_truth_hubs:
        folium.Circle(radius=8000 + (hub['weight'] * 1500), location=[hub['lat'], hub['lon']], color="#e74c3c", weight=2, dash_array="5, 8", fill=True, fill_color="#e74c3c", fill_opacity=0.15, popup=f"🚨 Hot Zone ({hub['count']} evidence points)").add_to(m)

    predictive_refuges = []
    REPORT_GAP_DEG = st.session_state.param_report_gap_mi / 69.0
    ESI_PROXY_RADIUS_DEG = 0.6
    ESI_THRESHOLD = st.session_state.param_esi_threshold
    MIN_REFUGE_SEPARATION_DEG = 0.15

    if len(combined_evidence_points) >= 3:
        esi_arr = np.array([pt["esi"] for pt in combined_evidence_points])
        min_lat_g, max_lat_g = coords_arr[:, 0].min(), coords_arr[:, 0].max()
        min_lon_g, max_lon_g = coords_arr[:, 1].min(), coords_arr[:, 1].max()
        buffer_deg = 0.15
        grid_n = 16
        lat_grid = np.linspace(min_lat_g - buffer_deg, max_lat_g + buffer_deg, grid_n)
        lon_grid = np.linspace(min_lon_g - buffer_deg, max_lon_g + buffer_deg, grid_n)

        candidates = []
        for g_lat in lat_grid:
            for g_lon in lon_grid:
                if filter_urban(g_lat, g_lon):
                    continue
                dists_deg = np.sqrt((coords_arr[:, 0] - g_lat) ** 2 + (coords_arr[:, 1] - g_lon) ** 2)
                nearest_report_deg = float(np.min(dists_deg))
                if nearest_report_deg < REPORT_GAP_DEG:
                    continue
                in_radius = dists_deg <= ESI_PROXY_RADIUS_DEG
                if not np.any(in_radius):
                    continue
                idw_weights = 1.0 / (dists_deg[in_radius] + 0.02) ** 2
                esi_proxy = float(np.sum(idw_weights * esi_arr[in_radius]) / np.sum(idw_weights))
                if esi_proxy >= ESI_THRESHOLD:
                    candidates.append({"lat": g_lat, "lon": g_lon, "esi_proxy": esi_proxy, "gap_miles": nearest_report_deg * 69.0})

        candidates.sort(key=lambda c: c["esi_proxy"], reverse=True)
        for cand in candidates:
            too_close = any(
                np.sqrt((cand["lat"] - kept["lat"]) ** 2 + (cand["lon"] - kept["lon"]) ** 2) < MIN_REFUGE_SEPARATION_DEG
                for kept in predictive_refuges
            )
            if not too_close:
                predictive_refuges.append(cand)
            if len(predictive_refuges) >= 5:
                break

    for ref in predictive_refuges:
        ref_popup = f"🪹 Predictive Refuge Zone<br><b>ESI proxy:</b> {ref['esi_proxy']:.2f}"
        folium.Circle(radius=12000, location=[ref['lat'], ref['lon']], color="#d35400", weight=2, dash_array="8, 8", fill=True, fill_color="#e67e22", fill_opacity=0.18, popup=ref_popup).add_to(m)

    LARSON_MAX_DEG = st.session_state.param_corridor_max_mi / 69.0
    if len(ground_truth_hubs) > 1:
        connected_pairs = set()
        for i in range(len(ground_truth_hubs)):
            h1 = ground_truth_hubs[i]
            dists = [(np.sqrt((h1["lat"] - ground_truth_hubs[j]["lat"]) ** 2 + (h1["lon"] - ground_truth_hubs[j]["lon"]) ** 2), j) for j in range(len(ground_truth_hubs)) if i != j]
            dists.sort()
            if dists and dists[0][0] < LARSON_MAX_DEG:
                j_near = dists[0][1]
                pair_key = tuple(sorted([i, j_near]))
                if pair_key not in connected_pairs:
                    connected_pairs.add(pair_key)
                    h2 = ground_truth_hubs[j_near]
                    vec = np.array([h2["lon"] - h1["lon"], h2["lat"] - h1["lat"]])
                    perp = np.array([-vec[1], vec[0]]) / (np.linalg.norm(vec) + 1e-6) * 0.025
                    p1 = [h1["lat"] + perp[1], h1["lon"] + perp[0]]
                    p2 = [h2["lat"] + perp[1], h2["lon"] + perp[0]]
                    p3 = [h2["lat"] - perp[1], h2["lon"] - perp[0]]
                    p4 = [h1["lat"] - perp[1], h1["lon"] - perp[0]]
                    folium.Polygon(locations=[p1, p2, p3, p4], color="#27ae60", weight=1.5, fill=True, fill_color="#27ae60", fill_opacity=0.15, popup="🌲 Larson Corridor").add_to(m)
                    larson_corridor_count += 1

st_folium(m, width="100%", height=500, key=f"map_{lat:.2f}_{lon:.2f}")

# --- Integrated Regional Intelligence Drawer (All Original Tabs & Features Intact) ---
st.markdown("---")
with st.expander(f"📊 Integrated Regional Intelligence — Active Sector: {loc_name} ({active_state})", expanded=True):
    panel_tab1, panel_tab2, panel_tab3, panel_tab4 = st.tabs([
        "🚨 Hot Zones & Larson Hypothesis",
        "🔊 Infrasound Physics & Formula",
        "🦉 Bioacoustics",
        "🗂️ Regional Intel"
    ])

    with panel_tab1:
        st.markdown("### 📊 Live Sector Calculations")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label="🚨 Hot Zones Found", value=len(ground_truth_hubs))
        col_m2.metric(label="🪹 Predictive Refuges Found", value=len(predictive_refuges))
        col_m3.metric(label="🌲 Larson Corridors Drawn", value=larson_corridor_count if show_hotspots else 0)

        if seasonal_breakdown:
            st.markdown("#### 📅 Seasonal Breakdown in Sector")
            s_cols = st.columns(len(seasonal_breakdown))
            for idx, (s_name, s_count) in enumerate(seasonal_breakdown.items()):
                s_cols[idx].metric(s_name, s_count)

        evidence_scan = scan_evidence_patterns(sightings_data)
        if any(evidence_scan.values()):
            st.markdown("#### 🔍 Behavioral & Physical Evidence Breakdown")
            ev_cols = st.columns(len(EVIDENCE_CATEGORIES))
            for idx, (cat_name, matches) in enumerate(evidence_scan.items()):
                ev_cols[idx].metric(cat_name, len(matches))

        with st.expander("🎛️ Model Parameter Tuning (Hot Zones, Refuges & Corridors)", expanded=False):
            c_p1, c_p2 = st.columns(2)
            st.session_state.param_hotzone_radius_mi = c_p1.slider("Hot Zone Clustering Radius (mi)", 5.0, 35.0, float(st.session_state.param_hotzone_radius_mi), 1.0)
            st.session_state.param_report_gap_mi = c_p2.slider("Predictive Refuge Min Gap from Reports (mi)", 5.0, 30.0, float(st.session_state.param_report_gap_mi), 1.0)
            st.session_state.param_esi_threshold = c_p1.slider("Environmental Suitability Index (ESI) Threshold", 0.30, 0.85, float(st.session_state.param_esi_threshold), 0.05)
            st.session_state.param_corridor_max_mi = c_p2.slider("Larson Corridor Max Span (mi)", 5.0, 50.0, float(st.session_state.param_corridor_max_mi), 1.0)
            st.session_state.param_effort_k = st.slider("Human Effort Bias Penalty (k)", 0.0, 2.0, float(st.session_state.param_effort_k), 0.1)

    with panel_tab2:
        st.markdown("### 🔊 Infrasound Physics & Attenuation Profiles")
        st.caption("Covers natural (hydrological/wind), industrial (dams/mines/turbines), and biological acoustic sources.")

        st.info(
            f"**👻/🔊 High-Strangeness & 'Ghost Frequency' (7–19 Hz) Sector Check:** "
            f"**{woo_infrasound_overlap} of {woo_total_in_sector}** high-strangeness or sensory-dread reports in this sector fall inside an active infrasound felt zone. "
            f"Standing waves near **18.9 Hz** (Vic Tandy's 'ghost frequency') vibrate the human ocular globe and inner ear, triggering peripheral shadow apparitions, cold sweats, sudden dread, and the intense sensation of **being watched**."
        )

        st.markdown("#### 📐 Attenuation & Effort-Adjusted Weighting Formulas")
        st.latex(r"W_{\text{adj}} = \frac{W_{\text{base}}}{1 + k \cdot E_{\text{human}}}, \quad E_{\text{human}} = \left(\frac{\rho_{\text{pop}}}{50}\right) \cdot \left(\frac{1}{d_{\text{road}} + 0.1}\right)")
        st.latex(r"\text{ESI} = 0.35\,\text{SC}_m + 0.25\,S_{\text{water}} + 0.20\,R_{\text{terrain}} + 0.20\,B_{\text{ungulate}}")

        for key, info in INFRASOUND_TYPES.items():
            with st.expander(f"{info['label']} — {info['freq_range']}"):
                st.write(f"**Cause:** {info['cause']}")
                st.write(f"**Character:** {info['character']}")
                st.write(f"**Felt Zone:** ~{info['felt_miles']} miles | **Detectable Footprint:** ~{info['travel_miles']} miles")
                st.caption(info["effect"])
                if key == "turbine":
                    st.markdown("[🌐 Official USGS Wind Turbine Database (USWTDB)](https://energy.usgs.gov/uswtdb/)")

        st.markdown("#### 🎧 Infrasound Pulse Rhythm Simulator")
        demo_hz = st.slider("Select Low-Frequency Pulse Rate (Hz) to Audition Rhythm:", 1.0, 19.0, 12.0, 0.5)
        st.audio(generate_tone_wav(freq_hz=demo_hz), format="audio/wav")
        st.caption("Modulated 110 Hz carrier demonstrating the physical cadence of low-frequency infrasound pulses on standard speakers.")

    with panel_tab3:
        st.markdown("### 🦉 Bioacoustics & Misidentification")
        st.write("Common wildlife species whose calls, wing-drumming, or territorial displays produce sub-audible pulses or mimic unexplained vocalizations.")

        if applicable_bio_species:
            st.markdown(f"#### 📍 Range-Verified Infrasonic Wildlife in {active_state}")
            for sp_name, sp_info in applicable_bio_species:
                with st.expander(f"🔊 {sp_name} ({sp_info['frequency']})"):
                    st.write(sp_info["description"])
                    st.caption(sp_info["citation"])
                    if sp_name in BIO_SPECIES_LISTEN_LINKS:
                        st.markdown(f"[🎧 Listen to Verified Field Recordings (Cornell Macaulay Library)]({BIO_SPECIES_LISTEN_LINKS[sp_name]})")
        else:
            st.info(f"No specialized range-restricted infrasonic species flagged specifically for {active_state}; general regional mimics apply.")

        st.markdown("#### 🌲 General North American Acoustic Mimics")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.write("* **Barred Owl:** Duetting territorial calls often mistaken for primate whoops or distant conversation.")
            st.write("* **Eastern / Western Coyote:** Pack yip-howls echoing through valleys mimic prolonged vocalizations.")
        with col_b2:
            st.write("* **White-Tailed Deer / Elk:** Alarm snorts, bugles, and heavy brush crashes at night.")
            st.write("* **American Black Bear:** Jaw-popping, huffing, and bipedal curiosity stands.")

    with panel_tab4:
        st.markdown("### 🗂️ Regional Lore, Press Archives & Researcher Findings")
        st.write(f"Showing indexed historical and cultural records near **{loc_name}** ({active_state}).")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"#### 🪶 Indigenous Lore ({len(lore_data)} local / {len(all_lore_records)} total)")
            if lore_data:
                for lr in lore_data[:10]:
                    with st.expander(f"{lr.get('name', lr.get('title', 'Lore Record'))} ({lr.get('tribe', 'Regional')})"):
                        st.write(lr.get("summary", lr.get("description", "No summary available.")))
                        if lr.get("source"):
                            st.caption(f"Source: {lr.get('source')}")
            else:
                st.caption("No localized indigenous lore entries within relevance radius of this center.")

        with col_r2:
            st.markdown(f"#### 📰 Historical Press Archives ({len(media_data)} local / {len(all_press_records)} total)")
            if media_data:
                for pr in media_data[:10]:
                    with st.expander(f"{pr.get('title', 'Press Clipping')} ({pr.get('date', 'N/A')})"):
                        st.write(pr.get("summary", "No summary available."))
                        if pr.get("publication"):
                            st.caption(f"Publication: {pr.get('publication')}")
            else:
                st.caption("No localized press clippings within relevance radius of this center.")

        st.markdown("---")
        st.markdown("#### 📚 Researcher Archives & Grover Krantz Finding Aid")
        st.markdown(f"[📄 Open Grover Krantz Finding Aid PDF (GitHub Archive)]({KRANTZ_PDF_URL})")
        if raw_researcher_archives:
            with st.expander(f"Browse Researcher Archive Index ({len(raw_researcher_archives)} entries)"):
                st.dataframe(raw_researcher_archives, use_container_width=True)
        if all_junk_records:
            with st.expander(f"🗃️ Unclassified / Supplemental Field Notes ({len(all_junk_records)} records)"):
                st.dataframe(all_junk_records, use_container_width=True)

# --- Field Log Submission ---
with st.expander("📝 Submit Investigator Field Log", expanded=False):
    with st.form("investigator_log_form"):
        visibility = st.radio("Storage Mode:", ["🔒 Private Vault", "🌐 Public Community Layer"], horizontal=True)
        obs_type = st.selectbox("Type", ["Visual Observation", "Suspect Impression", "Potential Nesting Site", "Vegetation Disturbance", "Acoustic Event"])
        physical_notes = st.text_area("Hard Physical Facts", placeholder="Measurements, trackway depth...")
        field_narrative = st.text_area("Observer Conjecture & Narrative", placeholder="Hypothesis...")
        submitted = st.form_submit_button("💾 Save Field Log", use_container_width=True)
        if submitted:
            if not researcher_name:
                st.warning("Enter a Researcher Name in the sidebar first.")
            else:
                try:
                    cursor = db_conn.cursor()
                    cursor.execute("""
                        INSERT INTO investigator_logs
                        (submitted_by, is_public, observation_type, event_date, latitude, longitude, physical_evidence_notes, field_narrative, ethics_agreed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (researcher_name, 1 if "Public" in visibility else 0, obs_type, str(datetime.now().date()), lat, lon, physical_notes, field_narrative, 1))
                    db_conn.commit()
                    st.success("Log saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving log: {e}")

# --- GPX Download ---
with st.expander("📡 Offline Field Export & GPX Package", expanded=False):
    gpx_data = generate_gpx(lat, lon, loc_name, sightings_data, camps_data, audio_data, user_logs_data)
    st.download_button(label="📥 Download Active Area GPX Package", data=gpx_data, file_name="bigfoot_field_zone.gpx", mime="application/gpx+xml")
