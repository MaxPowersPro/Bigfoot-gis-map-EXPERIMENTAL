import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
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

raw_sightings_bfro = load_bfro_clean_data()

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
        * **👣📜 Footprint Markers:** Blue = standard biological sighting. Purple = report text includes documented high-strangeness language (UFO, mystery lights, mind-speak, etc.) — a content flag, not a claim that anything paranormal occurred. Footprint vs. scroll icon shows BFRO vs. John Green historical source.
        * **🚨 Red Dotted Rings:** Hot Zones where reports, wildlife density, cover, and water overlap.
        * **🪹 Orange Dotted Rings:** Predictive Refuges — high inferred habitat quality with no direct sightings nearby (could mean it's genuinely quiet, or that reports aren't reaching outside databases).
        * **🌲 Green Channels:** Larson transit corridors (≤20 miles) connecting nearby Hot Zones along natural terrain gaps.
        * **🔊 Purple Marker:** Infrasound source. Bold inner ring = where it can actually be *felt*; light outer ring = where it's still *detectable*. Full source details are in the Local Intel drawer below the map.
        * **🏕️ Green Campgrounds:** Dispersed campsites and backcountry staging points.
        """)
    with col_g3:
        st.markdown("#### ⚙️ Sidebar Layer Toggles")
        st.write("""
        * Use the **7 Active Map Layers** in the sidebar to show or hide specific data overlays in real time.
        * Turn layers off to declutter dense search sectors when analyzing high-density sighting clusters.
        * **📝 Submitting a Field Log:** Enter a Researcher Name in the sidebar first, then use the "Submit Investigator Field Log" drawer near the bottom of the page. Full instructions are inside that drawer.
        """)
    st.markdown("---")

# --- Local SQLite Database Initialization ---
@st.cache_resource
def init_local_db():
    conn = sqlite3.connect("saved_searches.db", check_same_thread=False)
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

INFRASOUND_TYPES = {
    "waterfall": {"label": "🌊 Waterfall / Hydrological", "cause": "High-volume water impact at waterfalls, rapids, or river spillways.", "freq_range": "3 – 15 Hz", "character": "Continuous rumble.", "felt_miles": 12, "travel_miles": 80, "effect": "Falls mostly in the vestibular resonance band (1-7 Hz) — dizziness, pressure headaches, and loss of balance are the most commonly reported effects within the felt zone."},
    "wind": {"label": "🌬️ Wind / Mountain Pass", "cause": "High-velocity wind funneling through narrow ridge gaps, saddles, or mountain passes.", "freq_range": "0.5 – 5 Hz", "character": "Continuous, weather- and pressure-system dependent.", "felt_miles": 8, "travel_miles": 55, "effect": "Sits at the low end of the vestibular band — dizziness and disorientation are most likely, especially during active frontal weather passing through the notch."},
    "manmade": {"label": "⚙️ Man-Made (Dam / Mine)", "cause": "Dam spillway turbulence (continuous hum) or mine/quarry blasting (short, high-amplitude pulses).", "freq_range": "0.5 – 15 Hz", "character": "Dams: steady hum. Mines/quarries: sharp, punctuated blast pulses, documented detectable 10+ miles out.", "felt_miles": 8, "travel_miles": 60, "effect": "Mine blasts can produce a sudden felt pressure jolt rather than a gradual sensation; dam hum tends to produce the same dizziness/dread pattern as natural hydrological sources."},
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
    if any(k in s for k in ["wind", "pass", "ridge", "saddle", "gap", "aeolian"]):
        return "wind"
    if any(k in s for k in ["dam", "mine", "quarry", "blast"]):
        return "manmade"
    if any(k in s for k in ["biotic", "biological", "animal", "vocal", "call"]):
        return "biological"
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
    return ET.tostring(
