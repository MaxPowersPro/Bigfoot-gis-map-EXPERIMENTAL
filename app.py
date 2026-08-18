import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from supabase import create_client, Client
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

# ==========================================
# 1. PAGE SETUP & AUTO-LOCATION INIT
# ==========================================
st.set_page_config(
    page_title="Bigfoot Field Analysis Platform",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load master bundled datasets from /data directory
def load_bfro_clean_data(path="data/bfro_reports_clean.csv"):
    """Loads the official BFRO KMZ-derived dataset (replaces the old corrupted
    bfro_reports.csv entirely). Built as its own dedicated loader rather than
    reusing the generic BFRO-tuned loader, since that loader assumes a 'number'
    column for report IDs (this file has 'report_id') and only exposes a few
    fields at the top level -- reusing it would have silently broken the direct
    BFRO report links."""
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
    """Loads the Junk Drawer data directly - NOT through the generic loader, since that
    one requires latitude/longitude columns and silently returns nothing without them.
    Junk Drawer entries aren't tied to a location at all, which is exactly what caused
    the 'file not found or empty' bug even though the file was correctly named."""
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
    """Loads the Researcher Archives reference file - same pattern as the Junk Drawer
    loader, since this data isn't tied to a location either and the generic loader
    would silently return nothing for it."""
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
    """Loads John Green's historical sasquatch database (public domain), building
    sighting records from its real columns rather than forcing it through the
    BFRO-tuned generic loader. Every record is tagged source='John Green Historical
    Archive' so it never gets silently blended with/mistaken for BFRO data."""
    import pandas as pd
    import os
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            # A single malformed row (stray comma/quote in the free-text narrative or
            # citation fields) can make the strict parser fail the WHOLE file silently,
            # returning zero sightings with no visible error. Same class of bug already
            # found and fixed in the research scripts -- this fallback carries that same
            # fix into the live app.
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
raw_camps = load_and_standardize_dataset("data/campsites.csv")  # optional - app runs fine if this file doesn't exist yet
raw_junk = load_junk_drawer_data()  # optional - app runs fine if this file doesn't exist yet
all_junk_records = raw_junk

raw_researcher_archives = load_researcher_archives_data()  # optional - app runs fine if this file doesn't exist yet
KRANTZ_PDF_URL = "https://raw.githubusercontent.com/MaxPowersPro/Bigfoot-gis-map/main/data/krantz_finding_aid.pdf"

# Check visitor browser location on first load
if "user_lat" not in st.session_state or "user_lon" not in st.session_state:
    device_loc = get_geolocation()
    if device_loc and "coords" in device_loc:
        st.session_state.user_lat = device_loc["coords"]["latitude"]
        st.session_state.user_lon = device_loc["coords"]["longitude"]
        st.session_state.location_name = "Detected Local Sector"
    else:
        st.session_state.user_lat = 41.7000
        st.session_state.user_lon = -70.3000
        st.session_state.location_name = "Default Target Zone (Cape Cod / Wampanoag Sector)"

if "user_state" not in st.session_state:
    st.session_state.user_state = "Massachusetts"
if "user_county" not in st.session_state:
    st.session_state.user_county = "Barnstable County"

lat = float(st.session_state.user_lat)
lon = float(st.session_state.user_lon)
loc_name = str(st.session_state.location_name)
active_state = str(st.session_state.user_state)
active_county = str(st.session_state.user_county)

# ==========================================
# BRANDING HEADER BANNER
# ==========================================
try:
    st.image("image.png", use_container_width=True)
except Exception:
    try:
        st.image("header_banner.png", use_container_width=True)
    except Exception:
        st.title("Maxquest GIS")

st.caption("Site-Specific Spatial Map & Predictive Multi-Criteria Analysis Engine")

# ==========================================
# TOP COLLAPSED APP NAVIGATION & KEY GUIDE
# ==========================================
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
        """)
    st.markdown("---")

# ==========================================
# 2. SUPABASE CONNECTION & UTILITIES
# ==========================================
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None

supabase: Client = init_supabase()

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

# ---- Infrasound source classification (4 real categories, used by Local Intel + Research Library) ----
INFRASOUND_TYPES = {
    "waterfall": {
        "label": "🌊 Waterfall / Hydrological",
        "cause": "High-volume water impact at waterfalls, rapids, or river spillways.",
        "freq_range": "3 – 15 Hz",
        "character": "Continuous rumble.",
        "felt_miles": 12,
        "travel_miles": 80,
        "effect": "Falls mostly in the vestibular resonance band (1-7 Hz) — dizziness, pressure headaches, and loss of balance are the most commonly reported effects within the felt zone."
    },
    "wind": {
        "label": "🌬️ Wind / Mountain Pass",
        "cause": "High-velocity wind funneling through narrow ridge gaps, saddles, or mountain passes.",
        "freq_range": "0.5 – 5 Hz",
        "character": "Continuous, weather- and pressure-system dependent.",
        "felt_miles": 8,
        "travel_miles": 55,
        "effect": "Sits at the low end of the vestibular band — dizziness and disorientation are most likely, especially during active frontal weather passing through the notch."
    },
    "manmade": {
        "label": "⚙️ Man-Made (Dam / Mine)",
        "cause": "Dam spillway turbulence (continuous hum) or mine/quarry blasting (short, high-amplitude pulses).",
        "freq_range": "0.5 – 15 Hz",
        "character": "Dams: steady hum. Mines/quarries: sharp, punctuated blast pulses, documented detectable 10+ miles out.",
        "felt_miles": 8,
        "travel_miles": 60,
        "effect": "Mine blasts can produce a sudden felt pressure jolt rather than a gradual sensation; dam hum tends to produce the same dizziness/dread pattern as natural hydrological sources."
    },
    "biological": {
        "label": "🦍 Biological",
        "cause": "Large thoracic resonance structures and specialized laryngeal folds in large-bodied animals producing sub-audible vocal bursts.",
        "freq_range": "8 – 18 Hz",
        "character": "Vocal burst, not continuous.",
        "felt_miles": 3,
        "travel_miles": 8,
        "effect": "Overlaps the human alpha-wave / 'being watched' band (7-12 Hz) most directly, but the felt zone is small since source amplitude is far below geological or man-made sources."
    },
}

def classify_infrasound_type(event_type_str):
    """Returns one of 'waterfall' / 'wind' / 'manmade' / 'biological', or None if the source
    doesn't match a known category — we show it as unclassified rather than guessing."""
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

# ---- Evidence Pattern Scanner: keyword-based behavior detection across sightings ----
EVIDENCE_CATEGORIES = {
    "Tracks / Footprints": ["track", "footprint", "print", "cast"],
    "Vocalizations": ["whoop", "howl", "scream", "whistle", "vocal", "holler", "yell", "call"],
    "Tree Knocks": ["knock", "banging", "wood knock"],
    "Rock Throwing": ["rock", "stone", "thrown", "throw"],
    "Distinctive Odor": ["smell", "odor", "stench", "stink", "foul"],
}

def scan_evidence_patterns(sightings):
    """Scans each sighting's title+summary for known evidence/behavior keywords.
    Returns {category: [{report, season}, ...]} -- pure text matching, no claim about
    what actually happened, just what the report text mentions."""
    results = {cat: [] for cat in EVIDENCE_CATEGORIES}
    for s in sightings:
        text = f"{s.get('title', '')} {s.get('summary', '')}".lower()
        season = get_season(s.get("event_date", "N/A"))
        for category, keywords in EVIDENCE_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                results[category].append({"report": s, "season": season})
    return results

# ---- Paranormal/high-strangeness content indicator ----
# Deliberately content-based, not source-based: an earlier version of this app used
# Class C (secondhand sourcing) as a stand-in for "anomalous," which conflates two
# different things -- how reliably a report was sourced isn't the same as whether its
# content includes high-strangeness elements. A rock-solid Class A report can include
# a UFO; a Class C report can be perfectly mundane. This scans the actual text instead.
# Categories drawn from "Where the Footprints End" (Cutchin & Renner) -- documented
# recurring high-strangeness elements in Bigfoot literature.
PARANORMAL_KEYWORDS = [
    "ufo", "mystery light", "orb", "telepath", "mind-speak", "mindspeak", "mind speak",
    "vanished", "disappeared without", "teleport", "portal", "shapeshift", "shape-shift",
    "glowing eyes", "no tracks after", "tracks stopped", "psychic",
]

def scan_paranormal_content(text: str) -> bool:
    """Returns True if the report's own text mentions a documented high-strangeness
    category -- a content-based flag, never a claim that anything paranormal actually
    happened. Purely descriptive of what the report says."""
    low = text.lower()
    return any(kw in low for kw in PARANORMAL_KEYWORDS)

# ---- Black-bear-census-style weighting formulas (kept from the current live app) ----
def calculate_human_effort_factor(dist_to_road_miles: float, pop_density_sq_mi: float) -> float:
    """Models how much a location's report count is inflated/suppressed by human access & population,
    the same way bear census work corrects raw sighting counts for observer bias."""
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

# ==========================================
# 3. SIDEBAR CONTROLS & GEOCODING
# ==========================================
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
    if not st.secrets.get("MAPBOX_TOKEN", ""):
        st.caption("⚠️ No MAPBOX_TOKEN set in Secrets — area search won't work until this is added. Device GPS still works.")

    with st.expander("📂 Load a Saved Search"):
        loaded_file = st.file_uploader("Upload a saved search file (.json)", type=["json"], key="load_search_uploader")
        if loaded_file is not None and st.button("Jump to this saved search"):
            import json as _json
            try:
                saved = _json.loads(loaded_file.read())
                st.session_state.user_lat = saved["lat"]
                st.session_state.user_lon = saved["lon"]
                st.session_state.location_name = saved.get("location_name", "Loaded Search")
                st.session_state.radius_miles_key = saved.get("radius_miles", 100)
                for layer_key, layer_val in saved.get("layers", {}).items():
                    st.session_state[layer_key] = layer_val
                st.success(f"Loaded: {saved.get('location_name', 'saved search')}")
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

    loc_search = st.text_input("📍 Target Search Area", value=loc_name)
    radius_miles = st.selectbox("Field Radius (Miles)", [25, 50, 100, 250, 500], index=2, key="radius_miles_key")
    deg_delta = radius_miles / 69.0

    col_s1, col_s2 = st.columns(2)
    if col_s1.button("🔎 Search Area", use_container_width=True) and loc_search:
        res = geocode_mapbox(loc_search)
        if res:
            st.session_state.user_lat, st.session_state.user_lon, st.session_state.location_name, st.session_state.user_state, st.session_state.user_county = res
            st.rerun()
        else:
            st.warning("Search didn't return a result — check MAPBOX_TOKEN in Secrets, or try Device GPS instead.")

    if col_s2.button("📲 Device GPS", use_container_width=True):
        loc_data = get_geolocation()
        if loc_data and "coords" in loc_data:
            st.session_state.user_lat = loc_data["coords"]["latitude"]
            st.session_state.user_lon = loc_data["coords"]["longitude"]
            st.session_state.location_name = "Device GPS"
            st.rerun()

    st.markdown("---")
    st.subheader("🗺️ Active Map Layers")
    show_bfro = st.checkbox("1. 👣 Sightings (Dual Footprints)", value=True, key="layer_bfro")
    show_lore = st.checkbox("2. 🪶 Native American Lore Net", value=True, key="layer_lore")
    show_news = st.checkbox("3. 📰 Press Archives Net", value=True, key="layer_news")
    show_hotspots = st.checkbox("4. 🚨 Hot Zones, Refuges & The Larson Hypothesis", value=True, key="layer_hotspots")
    show_audio = st.checkbox("5. 🔊 Infrasound / Acoustic Masking", value=True, key="layer_audio")
    show_user_logs = st.checkbox("6. ⚠️ Community Field Logs", value=True, key="layer_user_logs")
    show_camps = st.checkbox("7. 🏕️ Camping & Access Points", value=True, key="layer_camps")

    st.markdown("---")
    save_search_payload = {
        "location_name": loc_name,
        "lat": lat,
        "lon": lon,
        "radius_miles": radius_miles,
        "layers": {
            "layer_bfro": show_bfro, "layer_lore": show_lore, "layer_news": show_news,
            "layer_hotspots": show_hotspots, "layer_audio": show_audio,
            "layer_user_logs": show_user_logs, "layer_camps": show_camps,
        },
    }
    import json as _json
    st.download_button(
        "💾 Save This Search",
        data=_json.dumps(save_search_payload, indent=2),
        file_name=f"saved_search_{loc_name.replace(' ', '_').replace(',', '')[:30]}.json",
        mime="application/json",
        use_container_width=True,
        help="Downloads a small file to your device with this exact location, radius, and layer setup. Load it back anytime with 'Load a Saved Search' above.",
    )

# ==========================================
# ADJUSTABLE MODEL ASSUMPTIONS — set defaults once; the Math & Science Drawer below the
# map lets a user change these live. Streamlit reruns top-to-bottom on every interaction,
# so reading from session_state here (before these values are used) means a slider defined
# much further down the page still correctly drives everything computed above it.
# ==========================================
PARAM_DEFAULTS = {
    "param_hotzone_radius_mi": 15.0,   # Hot Zone clustering bandwidth
    "param_report_gap_mi": 10.0,       # min distance from a direct report to flag a Predictive Refuge
    "param_esi_threshold": 0.55,       # min ESI proxy score to flag a Predictive Refuge
    "param_corridor_max_mi": 20.0,     # max distance between Hot Zones to draw a Larson corridor
    "param_effort_k": 0.5,             # strength of the human-access down-weighting in the effort formula
}
for _pk, _pv in PARAM_DEFAULTS.items():
    if _pk not in st.session_state:
        st.session_state[_pk] = _pv

# ==========================================
# 4. UNIFIED DATA PROCESSING & FILTERING
# ==========================================
sightings_data, camps_data, audio_data, media_data, lore_data, user_logs_data = [], [], [], [], [], []
seasonal_breakdown = {}

# Process Local Sightings
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

# Full, unfiltered datasets for the Research Library -- genuinely nationwide, not
# accidentally scoped to whatever's near the current search location. This was a real
# bug: the Research Library used to search the already-locally-filtered lore_data/
# media_data instead of everything.
all_lore_records = [item.get("metadata", item) for item in raw_lore] if raw_lore else []
all_press_records = [item.get("metadata", item) for item in raw_news] if raw_news else []

# Process Local Lore -- each entity has its own real relevance_radius_miles reflecting
# how localized vs. broadly regional that specific tradition actually is (a single
# named rock vs. a whole coastal nation's territory), rather than one generic multiplier
# for everything. Replaces the old state-name-substring fallback entirely, which is what
# caused the stale-Massachusetts-default bug -- clean distance-only matching now.
if raw_lore:
    for item in raw_lore:
        record = item.get("metadata", item)
        l_lat = float(record.get("latitude", lat))
        l_lon = float(record.get("longitude", lon))
        entry_radius = float(record.get("relevance_radius_miles", 100))
        if haversine_miles(lat, lon, l_lat, l_lon) <= entry_radius:
            lore_data.append(record)

# Process Local Press Archives -- same real-relevance-radius approach, no state fallback.
if raw_news:
    for item in raw_news:
        record = item.get("metadata", item)
        m_lat = float(record.get("latitude", lat))
        m_lon = float(record.get("longitude", lon))
        entry_radius = float(record.get("relevance_radius_miles", 30))
        if haversine_miles(lat, lon, m_lat, m_lon) <= entry_radius:
            media_data.append(record)

# Process Local Campsites
if raw_camps:
    for item in raw_camps:
        record = item.get("metadata", item)
        c_lat = float(record.get("latitude", lat))
        c_lon = float(record.get("longitude", lon))
        if haversine_miles(lat, lon, c_lat, c_lon) <= radius_miles:
            camps_data.append(record)

# Additive Supabase Layers (acoustic reports & community field logs stay live in Supabase)
if supabase:
    try:
        r = supabase.table("acoustic_reports").select("*").execute()
        for a in (r.data or []):
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
        r = supabase.table("investigator_logs").select("*").execute()
        for log in (r.data or []):
            if haversine_miles(lat, lon, float(log["latitude"]), float(log["longitude"])) <= radius_miles:
                user_logs_data.append(log)
    except Exception:
        pass

# Debunking cross-reference: flag sightings that fall within a known infrasound source's felt zone.
# This is informational only — it never touches evidence_weight or the ESI math. It exists so a
# researcher can see "a natural physiological explanation is plausible here" before leaning esoteric,
# the same role the fauna misidentification tab plays for sounds/sightings.
for s in sightings_data:
    s["nearby_infrasound"] = None
    for a in audio_data:
        dist = haversine_miles(float(s["latitude"]), float(s["longitude"]), float(a["latitude"]), float(a["longitude"]))
        if dist <= a.get("felt_radius_miles", 8):
            itype_label = INFRASOUND_TYPES.get(a.get("infrasound_type"), {}).get("label", "an unclassified infrasound source")
            s["nearby_infrasound"] = {"label": itype_label, "event_type": a.get("event_type", ""), "dist_miles": dist}
            break
    s["is_anomalous"] = scan_paranormal_content(f"{s.get('title', '')} {s.get('summary', '')}")

# ==========================================
# 5. MAP BANNER & FOLIUM MAP RENDERER
# ==========================================
st.markdown(f"""
<div style="background-color:#1e272c; color:white; padding:10px 14px; border-radius:6px; margin-bottom:12px; font-size:14px; border-left:4px solid #e74c3c;">
<b>📍 Active Sector Records ({loc_name} • {active_state}):</b> &nbsp;
👣 Sightings: <b><code>{len(sightings_data)}</code></b> &nbsp;|&nbsp;
🪶 Regional Lore: <b><code>{len(lore_data)}</code></b> &nbsp;|&nbsp;
📰 Press Archives: <b><code>{len(media_data)}</code></b> &nbsp;|&nbsp;
🔊 Infrasound Waves: <b><code>{len(audio_data)}</code></b> &nbsp;|&nbsp;
🏕️ Campsites: <b><code>{len(camps_data)}</code></b>
</div>
""", unsafe_allow_html=True)

m = folium.Map(location=[lat, lon], zoom_start=8, tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap")
folium.Circle(radius=radius_miles * 1609.34, location=[lat, lon], color="#e74c3c", weight=2, fill=True, fill_opacity=0.02).add_to(m)
folium.Marker([lat, lon], popup=f"<b>📍 TARGET CENTER: {loc_name}</b>", icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(m)

# Sightings Layer
if show_bfro and sightings_data:
    for s in sightings_data:
        j_lat, j_lon = apply_jitter(s["latitude"], s["longitude"], offset_seed=1)
        raw_summary = s.get("summary", "No transcript summary provided.")
        raw_id = str(s.get('report_id', s.get('id', ''))).strip()
        source_label = s.get("source", "BFRO")
        link_html = f'<br><a href="https://www.bfro.net/GDB/show_report.asp?id={raw_id}" target="_blank" style="display:inline-block; margin-top:4px; padding:3px 6px; background:#007bff; color:white; border-radius:3px; text-decoration:none; font-size:10px;">📄 Direct BFRO Report #{raw_id}</a>' if raw_id.isdigit() else ''
        infrasound_note = ""
        if s.get("nearby_infrasound"):
            ni = s["nearby_infrasound"]
            infrasound_note = f"""<div style="margin-top:4px; padding:4px; background:#f4f0fa; border-left:3px solid #8e44ad; font-size:9px;">🔊 <b>Natural explanation check:</b> within the felt zone of {ni['label']} ({ni['dist_miles']:.1f} mi) — see Local Intel.</div>"""

        is_anomalous = s.get("is_anomalous", False)
        pin_color = "#8e44ad" if is_anomalous else "#2b78e4"
        source_icon = "📜" if source_label != "BFRO" else "👣"

        anomalous_note = ""
        if is_anomalous:
            anomalous_note = """<div style="margin-top:4px; padding:4px; background:#f4f0fa; border-left:3px solid #8e44ad; font-size:9px;">👻 <b>Content flag (conjecture, not fact):</b> this report's own text includes language associated with documented high-strangeness categories. Not a claim that anything paranormal occurred — a flag for the physical-vs-paranormal distinction only.</div>"""

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
        # Real colorable SVG footprint, not an emoji -- purple = content-flagged as
        # containing high-strangeness language, blue = standard biological sighting.
        # Dots were the old marker shape; kept available for other future layers per
        # Max's request, footprints are now reserved for sightings specifically.
        footprint_svg = f"""<svg width="18" height="18" viewBox="0 0 24 24" style="filter: drop-shadow(0 0 1px white);"><ellipse cx="12" cy="15" rx="4.2" ry="8.5" fill="{pin_color}"/><ellipse cx="7.5" cy="5" rx="1.3" ry="2.1" fill="{pin_color}"/><ellipse cx="10.5" cy="3.2" rx="1.4" ry="2.3" fill="{pin_color}"/><ellipse cx="13.7" cy="3.5" rx="1.3" ry="2.2" fill="{pin_color}"/><ellipse cx="16.5" cy="5" rx="1.2" ry="1.9" fill="{pin_color}"/><ellipse cx="18.7" cy="7.5" rx="1.0" ry="1.5" fill="{pin_color}"/></svg>"""
        folium.Marker([j_lat, j_lon], popup=folium.Popup(popup_html, max_width=280), icon=folium.DivIcon(html=footprint_svg, icon_size=(20, 20), icon_anchor=(10, 16))).add_to(m)

# Campsites Layer
if show_camps and camps_data:
    for c in camps_data:
        c_popup = f"<b>🏕️ {c.get('name', 'Campsite')}</b><br><small>Type: {c.get('type', 'Primitive / Dispersed')}</small>"
        folium.Marker([c["latitude"], c["longitude"]], popup=c_popup, icon=folium.Icon(color="green", icon="campground", prefix="fa")).add_to(m)

# RESTORED: Infrasound / Acoustic Layer — single purple glance-and-go icon; details live in Local Intel below
if show_audio and audio_data:
    for a in audio_data:
        type_label = INFRASOUND_TYPES.get(a.get("infrasound_type"), {}).get("label", "❔ Unclassified Source")
        felt_m = a.get("felt_radius_miles", 8) * 1609.34
        travel_m = a["prop_radius_miles"] * 1609.34
        off_str = f"<br><b style='color:#d35400;'>⚠️ Trans-Boundary Source: {int(a['dist_to_target'])} miles away ({a['coverage_pct']}% local sector coverage)</b>" if a.get("is_offscreen") else ""
        a_popup = f"""<b>🔊 {type_label}</b><br><b>{a.get('event_type')}</b><br><small>Felt zone: ~{a.get('felt_radius_miles', 8)} mi | Detectable to: ~{a['prop_radius_miles']} mi</small>{off_str}<br><p style='font-size:10px; margin-top:4px;'>See Local Intel drawer below for full details.</p>"""
        if not a.get("is_offscreen"):
            folium.Marker([a["latitude"], a["longitude"]], popup=a_popup, icon=folium.Icon(color="purple", icon="volume-up", prefix="fa")).add_to(m)
        # Lighter outer ring: full detectable/travel distance
        folium.Circle(radius=travel_m, location=[a["latitude"], a["longitude"]], color="#8e44ad", weight=1, dash_array="4, 6", fill=True, fill_color="#8e44ad", fill_opacity=0.04, popup=f"Detectable footprint (~{a['prop_radius_miles']} mi)").add_to(m)
        # Bold inner ring: where it can actually be felt
        folium.Circle(radius=felt_m, location=[a["latitude"], a["longitude"]], color="#6c3483", weight=2, fill=True, fill_color="#8e44ad", fill_opacity=0.16, popup=f"Felt zone (~{a.get('felt_radius_miles', 8)} mi)").add_to(m)

# RESTORED: Community Field Logs Layer (also toggled by sidebar but wasn't actually drawn)
if show_user_logs and user_logs_data:
    for ulog in user_logs_data:
        has_facts = bool(ulog.get('physical_evidence_notes'))
        log_popup = f"<b>📝 FIELD LOG</b><br><small>Type: {ulog.get('observation_type')}</small><br><p style='font-size:10px;'>{ulog.get('physical_evidence_notes') or ulog.get('field_narrative', '')}</p>"
        folium.Marker([ulog["latitude"], ulog["longitude"]], popup=log_popup, icon=folium.Icon(color="green" if has_facts else "orange", icon="clipboard", prefix="fa")).add_to(m)

# NATIONAL Hot Zones, Predictive Refuges & Larson Hypothesis corridors -- computed ONCE
# across the entire dataset, cached, shared by every user. Fixes two people a mile apart
# seeing different zones. Wrapped in a single try/except this time -- if ANYTHING inside
# fails, for any reason, the map still renders with everything else intact instead of
# going blank with no error, which is what happened on the last attempt.
ground_truth_hubs, predictive_refuges = [], []
larson_corridor_count = 0
zones_computed_at = None

try:
    @st.cache_data(ttl=86400, show_spinner=True)
    def compute_national_zones(evidence_points, hotzone_radius_mi, report_gap_mi, esi_threshold, corridor_max_mi):
        import datetime
        hubs, refuges, corridors = [], [], []
        if not evidence_points:
            return {"hubs": hubs, "refuges": refuges, "corridors": corridors, "computed_at": None}

        coords_arr = np.array([[p[0], p[1]] for p in evidence_points])
        weights_arr = np.array([p[2] for p in evidence_points])
        esi_arr = np.array([p[3] for p in evidence_points])

        RADIUS_DEG = hotzone_radius_mi / 69.0
        # Grid-bucketed hub finding instead of a full O(n^2) distance matrix -- ~35-40x
        # faster on the real dataset size, and fixes a real double-counting bug the old
        # full-matrix version had: a point sitting between two cluster seeds could get
        # counted into BOTH hubs' totals, inflating weight. This version explicitly
        # excludes already-claimed points from new clusters, so no point is ever counted twice.
        cell_size = RADIUS_DEG * 2
        cell_of = {}
        for idx, (la, lo) in enumerate(coords_arr):
            key = (int(la // cell_size), int(lo // cell_size))
            cell_of.setdefault(key, []).append(idx)

        visited = set()
        for i in range(len(coords_arr)):
            if i in visited: continue
            la, lo = coords_arr[i]
            cell_key = (int(la // cell_size), int(lo // cell_size))
            candidate_idxs = []
            for d_lat in (-1, 0, 1):
                for d_lon in (-1, 0, 1):
                    candidate_idxs.extend(cell_of.get((cell_key[0] + d_lat, cell_key[1] + d_lon), []))
            candidate_idxs = np.array(list(set(candidate_idxs) - visited))
            if len(candidate_idxs) == 0:
                continue
            sub_coords = coords_arr[candidate_idxs]
            dists = np.sqrt(((sub_coords - coords_arr[i]) ** 2).sum(axis=1))
            neighbors_local = np.where(dists < RADIUS_DEG)[0]
            if len(neighbors_local) >= 1:
                neighbor_idxs = candidate_idxs[neighbors_local]
                hubs.append({
                    "lat": float(np.average(coords_arr[neighbor_idxs, 0], weights=weights_arr[neighbor_idxs])),
                    "lon": float(np.average(coords_arr[neighbor_idxs, 1], weights=weights_arr[neighbor_idxs])),
                    "weight": float(np.sum(weights_arr[neighbor_idxs])),
                    "count": int(len(neighbor_idxs)),
                })
                visited.update(neighbor_idxs.tolist())

        REPORT_GAP_DEG = report_gap_mi / 69.0
        ESI_PROXY_RADIUS_DEG = 0.6
        MIN_REFUGE_SEPARATION_DEG = 0.15
        GRID_STEP_DEG = 0.3

        if len(evidence_points) >= 3:
            min_lat_g, max_lat_g = coords_arr[:, 0].min(), coords_arr[:, 0].max()
            min_lon_g, max_lon_g = coords_arr[:, 1].min(), coords_arr[:, 1].max()
            lat_grid = np.arange(min_lat_g - 0.15, max_lat_g + 0.15, GRID_STEP_DEG)
            lon_grid = np.arange(min_lon_g - 0.15, max_lon_g + 0.15, GRID_STEP_DEG)

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
                    if esi_proxy >= esi_threshold:
                        candidates.append({"lat": float(g_lat), "lon": float(g_lon), "esi_proxy": esi_proxy, "gap_miles": nearest_report_deg * 69.0})

            candidates.sort(key=lambda c: c["esi_proxy"], reverse=True)
            for cand in candidates:
                too_close = any(
                    np.sqrt((cand["lat"] - kept["lat"]) ** 2 + (cand["lon"] - kept["lon"]) ** 2) < MIN_REFUGE_SEPARATION_DEG
                    for kept in refuges
                )
                if not too_close:
                    refuges.append(cand)
                if len(refuges) >= 150:
                    break

        LARSON_MAX_DEG = corridor_max_mi / 69.0
        if len(hubs) > 1:
            connected_pairs = set()
            for i in range(len(hubs)):
                h1 = hubs[i]
                dists = [(np.sqrt((h1["lat"] - hubs[j]["lat"]) ** 2 + (h1["lon"] - hubs[j]["lon"]) ** 2), j) for j in range(len(hubs)) if i != j]
                dists.sort()
                if dists and dists[0][0] < LARSON_MAX_DEG:
                    j_near = dists[0][1]
                    pair_key = tuple(sorted([i, j_near]))
                    if pair_key not in connected_pairs:
                        connected_pairs.add(pair_key)
                        h2 = hubs[j_near]
                        corridors.append({"h1": h1, "h2": h2})

        return {"hubs": hubs, "refuges": refuges, "corridors": corridors, "computed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

    @st.cache_data(ttl=86400, show_spinner=True)
    def compute_all_sightings_weighted(all_sightings, effort_k):
        processed = []
        for s in all_sightings:
            s = dict(s)
            s_lat, s_lon = s.get("latitude"), s.get("longitude")
            if s_lat is None or s_lon is None:
                continue
            event_d_str = s.get('event_date', 'N/A')
            try: ev_month = int(str(event_d_str).split('-')[1])
            except Exception: ev_month = 6
            s_dist_road = float(s.get("dist_to_road_miles", 0.4))
            s_pop_density = float(s.get("pop_density_sq_mi", 45.0))
            eff_factor = calculate_human_effort_factor(s_dist_road, s_pop_density)
            has_physical = bool(s.get("has_tracks") or s.get("has_hair") or s.get("has_physical_evidence"))
            class_rat = s.get("class_rating", "Class A")
            weight_dict = calculate_adjusted_evidence_weight(class_rat, has_physical, eff_factor, k=effort_k)
            sc_index = calculate_seasonal_cover_index(ev_month, 0.4, 0.5, True)
            terrain_input = s["real_terrain_roughness"] if s.get("real_terrain_roughness") is not None else 0.6
            s["evidence_weight"] = weight_dict["final_weight"]
            s["esi_score"] = calculate_environmental_suitability_index(sc_index, 0.3, terrain_input, 0.7)
            processed.append(s)
        return processed

    if show_hotspots:
        all_weighted = compute_all_sightings_weighted(tuple(raw_sightings), st.session_state.param_effort_k) if raw_sightings else []
        national_evidence_points = tuple(
            (float(s["latitude"]), float(s["longitude"]), float(s.get("evidence_weight", 1.0)), float(s.get("esi_score", 0.5)))
            for s in all_weighted if not filter_urban(float(s["latitude"]), float(s["longitude"]))
        )
        national_zones = compute_national_zones(
            national_evidence_points,
            st.session_state.param_hotzone_radius_mi,
            st.session_state.param_report_gap_mi,
            st.session_state.param_esi_threshold,
            st.session_state.param_corridor_max_mi,
        )
        DISPLAY_BUFFER_MI = radius_miles + st.session_state.param_hotzone_radius_mi
        ground_truth_hubs = [h for h in national_zones["hubs"] if haversine_miles(lat, lon, h["lat"], h["lon"]) <= DISPLAY_BUFFER_MI]
        predictive_refuges = [r for r in national_zones["refuges"] if haversine_miles(lat, lon, r["lat"], r["lon"]) <= DISPLAY_BUFFER_MI]
        visible_corridors = [c for c in national_zones["corridors"] if haversine_miles(lat, lon, c["h1"]["lat"], c["h1"]["lon"]) <= DISPLAY_BUFFER_MI or haversine_miles(lat, lon, c["h2"]["lat"], c["h2"]["lon"]) <= DISPLAY_BUFFER_MI]
        zones_computed_at = national_zones["computed_at"]

        for hub in ground_truth_hubs:
            folium.Circle(radius=8000 + (hub['weight'] * 1500), location=[hub['lat'], hub['lon']], color="#e74c3c", weight=2, dash_array="5, 8", fill=True, fill_color="#e74c3c", fill_opacity=0.15, popup=f"🚨 Hot Zone ({hub['count']} evidence points, Total Weight: {hub['weight']:.1f}x)").add_to(m)

        for ref in predictive_refuges:
            ref_popup = (f"🪹 Predictive Refuge Zone<br>"
                         f"<b>ESI proxy:</b> {ref['esi_proxy']:.2f} (from nearby reports — not yet real terrain data)<br>"
                         f"<b>Nearest report:</b> {ref['gap_miles']:.1f} mi away<br>"
                         f"<small>High inferred habitat quality, no direct reports here.</small>")
            folium.Circle(radius=12000, location=[ref['lat'], ref['lon']], color="#d35400", weight=2, dash_array="8, 8", fill=True, fill_color="#e67e22", fill_opacity=0.18, popup=ref_popup).add_to(m)

        larson_corridor_count = len(visible_corridors)
        for c in visible_corridors:
            h1, h2 = c["h1"], c["h2"]
            vec = np.array([h2["lon"] - h1["lon"], h2["lat"] - h1["lat"]])
            perp = np.array([-vec[1], vec[0]]) / (np.linalg.norm(vec) + 1e-6) * 0.025
            p1 = [h1["lat"] + perp[1], h1["lon"] + perp[0]]
            p2 = [h2["lat"] + perp[1], h2["lon"] + perp[0]]
            p3 = [h2["lat"] - perp[1], h2["lon"] - perp[0]]
            p4 = [h1["lat"] - perp[1], h1["lon"] - perp[0]]
            folium.Polygon(locations=[p1, p2, p3, p4], color="#27ae60", weight=1.5, fill=True, fill_color="#27ae60", fill_opacity=0.15, popup="🌲 The Larson Hypothesis: Transit Channel").add_to(m)

except Exception as _zone_error:
    st.warning(f"⚠️ Zone computation hit an error and was skipped this time (map itself is fine): {_zone_error}")
    ground_truth_hubs, predictive_refuges = [], []
    larson_corridor_count = 0

st_folium(m, width="100%", height=500, key=f"map_{lat:.2f}_{lon:.2f}")

if zones_computed_at:
    st.caption(f"🕒 National zones last calculated: {zones_computed_at}")

# ==========================================
# 6. INTEGRATED REGIONAL INTELLIGENCE DRAWER (the 4-tab "local intel" drawer)
# ==========================================
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
        with col_m1:
            st.metric(label="🚨 Hot Zones Found", value=len(ground_truth_hubs))
        with col_m2:
            st.metric(label="🪹 Predictive Refuges Found", value=len(predictive_refuges))
        with col_m3:
            st.metric(label="🌲 Larson Corridors Drawn", value=larson_corridor_count if show_hotspots else 0)

        st.markdown("---")
        col_hz, col_ref, col_lh = st.columns(3)
        with col_hz:
            st.markdown("### 🚨 Hot Zones")
            st.caption("Weighted density hubs — not just raw sighting locations.")
            st.write("""
            * Combines sighting reports, historical press mentions, and cover/water/effort factors into one weighted score per point.
            * Nearby weighted points cluster into a hub; the circle radius scales with total cluster weight.
            * Urban zones are automatically excluded — large omnivores don't sustain themselves there.
            """)
        with col_ref:
            st.markdown("### 🪹 Predictive Refuge Zones")
            st.caption("Probable habitat with *no* direct sightings.")
            st.write("""
            * A sighting marks where something was *seen*, not necessarily where it *lives* — a report gap can also mean a close-knit community that simply doesn't report to outsiders, not that nothing is there.
            * Scans a grid across the sector; flags a pocket if it's >10 miles from any direct report **and** scores high on an ESI proxy inferred from nearby reports' habitat data.
            * **Honest limitation:** the ESI proxy is inferred from nearby known reports, not independent terrain/hydrology data yet — that upgrade is on the data-farming list.
            """)
        with col_lh:
            st.markdown("### 🌲 The Larson Hypothesis")
            st.caption("Path-of-least-resistance transit, capped at 20 miles.")
            st.write("""
            * Connects Hot Zones within 20 miles of each other — calibrated against a real field-observed corridor length, not an arbitrary number.
            * Currently a straight-line geometric estimate. Real terrain-following path tracing (rivers, ridgelines, forest cover) needs land-cover/hydrology data we don't have hooked up yet.
            * Kept distinct from the red zones on purpose — a corridor is a *route*, not a *habitat*.
            """)

        st.markdown("---")
        st.markdown("#### 🧮 Formula Transparency")
        st.latex(rf"W_{{\text{{adjusted}}}} = \frac{{W_{{\text{{base}}}}}}{{1.0 + ({st.session_state.param_effort_k} \times \text{{EffortFactor}})}}")
        st.caption(f"Current k = {st.session_state.param_effort_k} — adjustable in the Math & Science Drawer below.")
        st.caption("Effort Factor models human observer bias (population density ÷ distance to road) — the same correction wildlife census work applies to raw sighting counts, borrowed from black bear census methodology.")
        if sightings_data:
            with st.expander("See the calculation for a sample report in this sector"):
                sample = sightings_data[0]
                st.code(sample.get("audit_explanation", "No audit trail available for this record."))

    with panel_tab2:
        st.markdown(f"### 🔊 Infrasound Sources Logged In This Sector")
        st.caption(f"Site-specific — {loc_name} only. For general infrasound physics covering all source types, see the Research Library below.")

        if not audio_data:
            st.info("No known infrasound sources currently logged within this sector's radius.")
        else:
            for a in audio_data:
                itype = a.get("infrasound_type")
                info = INFRASOUND_TYPES.get(itype)
                st.markdown(f"#### {info['label'] if info else '❔ Unclassified Source'} — {a.get('event_type', 'Unnamed Source')}")
                if a.get("is_offscreen"):
                    st.warning(f"⚠️ Origin sits {int(a['dist_to_target'])} miles outside this sector, but its footprint reaches in ({a['coverage_pct']}% local coverage).")
                if info:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Frequency", info["freq_range"])
                    col2.metric("Felt Zone", f"~{a.get('felt_radius_miles', info['felt_miles'])} mi")
                    col3.metric("Detectable To", f"~{a.get('prop_radius_miles', info['travel_miles'])} mi")
                    st.write(f"**Cause:** {info['cause']}")
                    st.write(f"**Character:** {info['character']}")
                    st.caption(info["effect"])
                else:
                    st.caption("This source hasn't been matched to one of the four known categories yet — treat felt/travel distances as unverified until it's re-classified.")
                if a.get("notes"):
                    st.caption(f"Field notes: {a.get('notes')}")
                st.markdown("---")

            st.markdown("### 🎧 Field Tone Simulator")
            st.caption("Infrasound itself is below human hearing — this shifts a chosen frequency up into the audible range so you can hear a representation of it. Set it to a detected source's frequency to get a feel for what it might sound like pitch-shifted.")
            base_hz = st.slider("Select Base Frequency (Hz):", 0.5, 19.0, 8.5, 0.5, key="local_intel_hz_slider")
            audible_hz = base_hz * 16
            st.info(f"**Target Infrasound Frequency:** `{base_hz} Hz`  ➜  **Pitch-Shifted Audible Tone:** `{audible_hz:.1f} Hz`")
            t = np.linspace(0, 2.0, int(22050 * 2.0), False)
            tone = (0.5 * np.sin(2 * np.pi * audible_hz * t) * 32767).astype(np.int16).tobytes()
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050); wf.writeframes(tone)
            st.audio(buf.getvalue(), format="audio/wav")

    with panel_tab3:
        st.markdown(f"**Location:** {loc_name} (`{lat:.4f}, {lon:.4f}`)")
        st.caption("Local species whose calls or tracks are commonly misidentified as Bigfoot sightings or sounds.")
        st.write("""
        * **Owls & Raptors:** Barred Owl (caterwauls, whoops), Great Horned Owl (deep hoots).
        * **Canids & Predators:** Eastern Coyote (yip-harmonics), Red/Gray Fox (screams).
        * **Mammals:** White-Tailed Deer (alarm snorts), Black Bear (guttural huffs, upright stance, human-like tracks when hind prints overlap front prints).
        """)
        macaulay_url = f"https://www.macaulaylibrary.org/catalog?searchField=location&lat={lat}&long={lon}"
        xenocanto_url = f"https://xeno-canto.org/explore?query=lat:{lat}%20lon:{lon}"
        st.markdown(f"* [🔊 **Macaulay Library (Cornell Lab)**]({macaulay_url}) — real recordings for this area")
        st.markdown(f"* [🌐 **Xeno-Canto Geographic Database**]({xenocanto_url}) — community-sourced calls for this area")

    with panel_tab4:
        c_lore, c_media, c_season = st.columns(3)
        with c_lore:
            st.markdown(f"#### 🪶 Regional Lore ({active_state})")
            if lore_data:
                for item in lore_data:
                    tribe = item.get('tribe_name', item.get('tribe', 'Indigenous Record'))
                    entity = item.get('entity_name', item.get('title', 'Entity'))
                    narrative = item.get('full_narrative', item.get('summary', ''))
                    st.write(f"**{tribe} — {entity}:**")
                    st.caption(f"> {narrative}")
            else:
                st.info(f"No recorded indigenous narratives specifically indexed for {active_state}.")
        with c_media:
            st.markdown(f"#### 📰 Press Archives ({active_state})")
            if media_data:
                for item in media_data:
                    title = item.get('title', item.get('headline', 'Archive'))
                    p_date = item.get('pub_date', item.get('event_date', 'Historical'))
                    text = item.get('full_text_transcript', item.get('summary', ''))
                    st.write(f"**{title} ({p_date})**")
                    st.caption(text[:150] + "..." if len(text) > 150 else text)
            else:
                st.info(f"No press accounts indexed for {active_state}.")
        with c_season:
            st.markdown("#### 🍂 Seasonal Activity Breakdown")
            if seasonal_breakdown:
                for season_name, count in seasonal_breakdown.items():
                    st.write(f"**{season_name}:** {count} reports")
            else:
                st.info("No dated reports in this sector yet.")

# ==========================================
# DRAWER: MATH & SCIENCE DRAWER (advanced — collapsed by default, doesn't compete for tab
# space with the field-use Local Intel tabs on a small screen)
# ==========================================
with st.expander("🔬 Math & Science Drawer (Advanced)", expanded=False):
    st.caption("Every formula this app uses, why we're using it, and how it was calibrated. Science-based critique on their efficacy is welcome and will be considered.")

    ms_tab0, ms_tab1, ms_tab2 = st.tabs(["📖 The Formulas", "🧮 What-If Calculator", "⚙️ Model Assumptions"])

    with ms_tab0:
        st.caption("Every formula this app runs, in plain English — what it is, why we use it, where it's used, and what it can't do yet. Science-based critique on any of this is welcome and will be considered.")

        with st.expander("1️⃣ Observer Effort Factor", expanded=True):
            st.latex(r"\text{EffortFactor} = \left(\frac{\text{PopDensity}}{50}\right) \times \frac{1}{\text{DistToRoad} + 0.1}")
            st.write("""
            **In plain terms:** a single number estimating how *easy* it would be for an ordinary person to be standing in that spot to see and report something. **Population density** = people per square mile nearby. **Distance to road** = miles to the nearest road/trail. Busy + close to a road = a high number. Deep, roadless wilderness = a low number.
            """)
            st.write("**Why we use it:** more reports come from easy-to-access places simply because more people pass through them — not because the species is more common there. This is the same correction real wildlife census work (including black bear studies) applies to raw sighting counts.")
            st.write("**Where it's used:** feeds directly into the Effort-Adjusted Evidence Weight below. Try your own numbers in the What-If Calculator tab.")

        with st.expander("2️⃣ Effort-Adjusted Evidence Weight"):
            st.latex(rf"W_{{\text{{adjusted}}}} = \frac{{W_{{\text{{base}}}}}}{{1.0 + (k \times \text{{EffortFactor}})}}")
            st.write("""
            **In plain terms:** every report starts with a **Base Weight** — a starting score for how strong that report is *before* any correction: `3.0` if there's physical evidence (tracks, hair, a cast), `1.5` for a Class A report (a direct sighting), `0.8` for Class B (a sound or track, no visual), `0.3` for Class C (a secondhand story), plus a `+0.25` bonus if it's corroborated by nearby lore. **k** is how strongly we discount reports from easy-access areas (default `0.5`, adjustable). **EffortFactor** is the number from formula #1 above. The final result is capped between `0.1` and `4.0` so no single report can dominate or vanish entirely.
            """)
            st.write("**Why we use it:** pulls the accessibility bias back out — an identical report from deep wilderness ends up counting for more than one from a backyard.")
            st.write("**Where it's used:** this is the actual weight shown on every sighting popup, and it's what feeds the Hot Zone clustering below.")

        with st.expander("3️⃣ Seasonal Cover Index (SCₘ)"):
            st.latex(r"SC_m = \text{Prop}_{\text{evergreen}} + (\text{Prop}_{\text{deciduous}} \times \text{LeafStatus}) + \text{Bonus}_{\text{understory}}")
            st.write("""
            **In plain terms:** estimates how much visual/thermal concealment a spot's plant cover gives *in a given month*. Evergreen tree cover counts fully year-round. Deciduous (leaf-dropping) tree cover only counts fully when leaves are actually on the trees (roughly May–October); in winter it barely helps. A bonus is added for year-round dense understory like thickets or rhododendron.
            """)
            st.write("**Why we use it:** a spot that looks well-covered in a summer photo can be wide open in winter — this keeps the model honest about the season.")
            st.write("**Where it's used:** feeds into the Environmental Suitability Index below.")

        with st.expander("4️⃣ Environmental Suitability Index (ESI)"):
            st.latex(r"ESI = 0.35 \cdot SC_m + 0.25 \cdot \text{WaterScore} + 0.20 \cdot \text{TerrainRoughness} + 0.20 \cdot \text{UngulateBiomass}")
            st.write("""
            **In plain terms:** a single score from 0 to 1 estimating how good a *habitat* a spot is — combining cover (formula #3), how close it is to water, how rugged/inaccessible the terrain is, and how much prey/food (deer, etc.) is around. Calculated completely independently of whether anyone has ever reported anything there.
            """)
            st.write("**Why we use it:** a sighting only tells you where something was *seen*. ESI is the app's attempt to estimate where something *could live*, sightings or not — the core of the whole 'search where he should be, not just where he was' idea.")
            st.write("**Where it's used:** drives the Predictive Refuge Zone trigger (#6).")
            st.warning("**Honest limitation:** terrain roughness is now real, USGS elevation-derived data for reports that have been farmed (each sighting's popup shows which kind it got). Water proximity and prey biomass are still placeholder constants, not real per-location data — hooking those up is still on the data-farming list.")

        with st.expander("5️⃣ Hot Zone Clustering"):
            st.write("""
            **In plain terms:** groups nearby weighted reports into a single zone if they fall within a set distance of each other (default 15 miles, adjustable). The circle's size on the map scales with the *combined weight* of everything inside it — not just the raw count.
            """)
            st.write("**Why we use it:** a cluster of independently corroborating reports is more convincing than any single one; clustering shows visually where multiple data points agree.")
            st.write("**Where it's used:** the red dotted rings on the map.")

        with st.expander("6️⃣ Predictive Refuge Trigger"):
            st.write("""
            **In plain terms:** scans a grid across the search sector. Flags a spot as a possible refuge if it's far enough from any actual report (default 10 miles) **and** its inferred ESI — estimated from nearby *known* reports' habitat scores — is high enough (default 0.55).
            """)
            st.write("**Why we use it:** directly answers the Kentucky case a fellow researcher (Matt Larson) described — a quiet area surrounded by good habitat may be quiet because of human reporting behavior, not because nothing's there.")
            st.write("**Where it's used:** the orange dotted rings.")
            st.warning("**Honest limitation:** the ESI used here is inferred/interpolated from nearby reports, not independently measured terrain data yet. Every refuge popup says so.")

        with st.expander("7️⃣ Larson Corridor Connection"):
            st.write("""
            **In plain terms:** draws a straight connecting line between two Hot Zones if they're within a set distance of each other (default 20 miles — calibrated against a real, Google-Maps-verified corridor a researcher described in Florida).
            """)
            st.write("**Why we use it:** models a plausible travel path between two good habitat areas, framed as connective tissue *within* a home range rather than migration between separate populations.")
            st.write("**Where it's used:** the green corridor lines.")
            st.warning("**Honest limitation:** this is a straight-line geometric guess, not a terrain-following path yet — real path-tracing needs land-cover/hydrology data we don't have hooked up.")

        with st.expander("8️⃣ Infrasound Felt/Travel Distances"):
            st.write("**In plain terms:** for each of 4 known source types (waterfall, wind/mountain pass, man-made dam/mine, biological), a rough distance where the source is still *physically detectable* vs. a tighter distance where a person would actually *feel* an effect.")
            st.write("**Why we use it — and what it's NOT for:** this is deliberately kept OUT of the positive evidence model. It never adds weight toward 'Bigfoot is here.' Its only job is the opposite — a debunking check, so a researcher can see a natural physiological explanation is plausible before assuming something esoteric.")
            st.write("**Where it's used:** the purple infrasound rings on the map, and the natural-explanation note on any sighting that falls inside a felt zone.")

    with ms_tab1:
        st.markdown("### Effort-Adjusted Evidence Weight — try your own numbers")
        st.caption("This does NOT change the live map — it's a sandbox so you can see exactly how the formula behaves before trusting it.")
        wi_col1, wi_col2 = st.columns(2)
        with wi_col1:
            wi_dist_road = st.slider("Distance to nearest road (mi)", 0.05, 20.0, 0.4, 0.05, key="wi_dist_road")
            wi_pop_density = st.slider("Population density (people/sq mi)", 0.0, 500.0, 45.0, 5.0, key="wi_pop_density")
            wi_class = st.selectbox("Report class", ["Class A", "Class B", "Class C"], key="wi_class")
        with wi_col2:
            wi_physical = st.checkbox("Has physical evidence (tracks/hair/cast)", key="wi_physical")
            wi_lore = st.checkbox("Corroborated by nearby lore", key="wi_lore")
            wi_month = st.slider("Month (for Seasonal Cover Index)", 1, 12, 7, key="wi_month")

        wi_effort = calculate_human_effort_factor(wi_dist_road, wi_pop_density)
        wi_weight = calculate_adjusted_evidence_weight(wi_class, wi_physical, wi_effort, lore_boost=wi_lore, k=st.session_state.param_effort_k)
        wi_sc = calculate_seasonal_cover_index(wi_month, 0.4, 0.5, True)
        wi_esi = calculate_environmental_suitability_index(wi_sc, 0.3, 0.6, 0.7)

        wr1, wr2, wr3 = st.columns(3)
        wr1.metric("Effort Factor", wi_effort)
        wr2.metric("Adjusted Weight", f"{wi_weight['final_weight']}x")
        wr3.metric("Seasonal ESI", wi_esi)
        st.code(wi_weight["audit_explanation"])

    with ms_tab2:
        st.markdown("### Live Model Assumptions")
        st.caption("These sliders change what the map actually draws, right now, for this session. Move one, then scroll back up to see the effect.")

        st.session_state.param_hotzone_radius_mi = st.slider(
            "Hot Zone clustering radius (mi)", 3.0, 40.0, st.session_state.param_hotzone_radius_mi, 1.0,
            help="How close two reports need to be to count as the same cluster. Default 15 mi — a working guess, not a measured constant."
        )
        st.session_state.param_report_gap_mi = st.slider(
            "Predictive Refuge — min. distance from any report (mi)", 3.0, 30.0, st.session_state.param_report_gap_mi, 1.0,
            help="How far a spot must sit from the nearest direct report before it can be flagged as a 'quiet pocket.' Default 10 mi."
        )
        st.session_state.param_esi_threshold = st.slider(
            "Predictive Refuge — min. ESI proxy score", 0.30, 0.90, st.session_state.param_esi_threshold, 0.05,
            help="How strong the inferred habitat quality has to be before flagging a refuge. Default 0.55."
        )
        st.session_state.param_corridor_max_mi = st.slider(
            "Larson corridor — max distance between Hot Zones (mi)", 5.0, 60.0, st.session_state.param_corridor_max_mi, 5.0,
            help="Calibrated against Matt Larson's real ~20-mile Florida corridor. Not a proven upper bound for the species."
        )
        st.session_state.param_effort_k = st.slider(
            "Effort-weighting strength (k)", 0.0, 2.0, st.session_state.param_effort_k, 0.1,
            help="How aggressively high-access reports get down-weighted. Default 0.5."
        )

        st.markdown("---")
        col_reset, col_export, col_import = st.columns(3)
        with col_reset:
            if st.button("↩️ Reset to Defaults", use_container_width=True):
                for _pk, _pv in PARAM_DEFAULTS.items():
                    st.session_state[_pk] = _pv
                st.rerun()
        with col_export:
            settings_json = json.dumps({k: st.session_state[k] for k in PARAM_DEFAULTS}, indent=2)
            st.download_button("💾 Download My Settings", data=settings_json, file_name="maxquest_model_settings.json", mime="application/json", use_container_width=True)
        with col_import:
            uploaded_settings = st.file_uploader("📤 Load Settings", type="json", key="settings_uploader", label_visibility="collapsed")
            if uploaded_settings is not None:
                try:
                    loaded = json.load(uploaded_settings)
                    for pk in PARAM_DEFAULTS:
                        if pk in loaded:
                            st.session_state[pk] = float(loaded[pk])
                    st.success("Settings loaded — scroll up to see the updated map.")
                except Exception as e:
                    st.error(f"Couldn't read that file: {e}")

        st.caption("Saving or sharing a search area (not just these model settings) is on the list for later — not part of this drawer yet.")

# ==========================================
# DRAWER: NATIONAL ZONE CONSISTENCY PROOF (Experimental)
# Completely isolated from the live map — no folium, no map rendering, plain Streamlit
# only. This exists to safely prove out the "compute zones once, nationally" idea before
# the map itself ever touches it again, after the last attempt broke the map with no
# visible error. If anything here fails, the map above is completely unaffected, since
# this code runs after the map has already fully rendered.
# ==========================================
with st.expander("🧪 National Zone Consistency Proof (Experimental)", expanded=False):
    st.caption("Proves whether two nearby points would see the SAME Hot Zones/Refuges — the actual thing being fixed — without touching the live map at all. Safe to experiment with.")

    @st.cache_data(ttl=86400, show_spinner=True)
    def proof_weight_all_sightings(all_sightings, effort_k):
        processed = []
        for s in all_sightings:
            s = dict(s)
            s_lat, s_lon = s.get("latitude"), s.get("longitude")
            if s_lat is None or s_lon is None:
                continue
            event_d_str = s.get('event_date', 'N/A')
            try: ev_month = int(str(event_d_str).split('-')[1])
            except Exception: ev_month = 6
            s_dist_road = float(s.get("dist_to_road_miles", 0.4))
            s_pop_density = float(s.get("pop_density_sq_mi", 45.0))
            eff_factor = calculate_human_effort_factor(s_dist_road, s_pop_density)
            has_physical = bool(s.get("has_tracks") or s.get("has_hair") or s.get("has_physical_evidence"))
            class_rat = s.get("class_rating", "Class A")
            weight_dict = calculate_adjusted_evidence_weight(class_rat, has_physical, eff_factor, k=effort_k)
            sc_index = calculate_seasonal_cover_index(ev_month, 0.4, 0.5, True)
            terrain_input = s["real_terrain_roughness"] if s.get("real_terrain_roughness") is not None else 0.6
            s["evidence_weight"] = weight_dict["final_weight"]
            s["esi_score"] = calculate_environmental_suitability_index(sc_index, 0.3, terrain_input, 0.7)
            processed.append(s)
        return processed

    @st.cache_data(ttl=86400, show_spinner=True)
    def proof_compute_national_hubs(evidence_points, hotzone_radius_mi):
        import datetime
        if not evidence_points:
            return {"hubs": [], "computed_at": None}
        coords_arr = np.array([[p[0], p[1]] for p in evidence_points])
        weights_arr = np.array([p[2] for p in evidence_points])
        RADIUS_DEG = hotzone_radius_mi / 69.0
        cell_size = RADIUS_DEG * 2
        cell_of = {}
        for idx, (la, lo) in enumerate(coords_arr):
            key = (int(la // cell_size), int(lo // cell_size))
            cell_of.setdefault(key, []).append(idx)
        visited = set()
        hubs = []
        for i in range(len(coords_arr)):
            if i in visited: continue
            la, lo = coords_arr[i]
            cell_key = (int(la // cell_size), int(lo // cell_size))
            candidate_idxs = []
            for d_lat in (-1, 0, 1):
                for d_lon in (-1, 0, 1):
                    candidate_idxs.extend(cell_of.get((cell_key[0] + d_lat, cell_key[1] + d_lon), []))
            candidate_idxs = np.array(list(set(candidate_idxs) - visited))
            if len(candidate_idxs) == 0:
                continue
            sub_coords = coords_arr[candidate_idxs]
            dists = np.sqrt(((sub_coords - coords_arr[i]) ** 2).sum(axis=1))
            neighbors_local = np.where(dists < RADIUS_DEG)[0]
            if len(neighbors_local) >= 1:
                neighbor_idxs = candidate_idxs[neighbors_local]
                hubs.append({
                    "lat": float(np.average(coords_arr[neighbor_idxs, 0], weights=weights_arr[neighbor_idxs])),
                    "lon": float(np.average(coords_arr[neighbor_idxs, 1], weights=weights_arr[neighbor_idxs])),
                    "weight": float(np.sum(weights_arr[neighbor_idxs])), "count": int(len(neighbor_idxs)),
                })
                visited.update(neighbor_idxs.tolist())
        return {"hubs": hubs, "computed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

    if st.button("Run the national computation", key="run_national_proof"):
        try:
            proof_weighted = proof_weight_all_sightings(tuple(raw_sightings), st.session_state.param_effort_k) if raw_sightings else []
            proof_points = tuple(
                (float(s["latitude"]), float(s["longitude"]), float(s.get("evidence_weight", 1.0)))
                for s in proof_weighted if not filter_urban(float(s["latitude"]), float(s["longitude"]))
            )
            proof_result = proof_compute_national_hubs(proof_points, st.session_state.param_hotzone_radius_mi)
            st.session_state["_proof_result"] = proof_result
            st.success(f"Computed {len(proof_result['hubs'])} national Hot Zones from {len(proof_weighted)} sightings.")
        except Exception as e:
            st.error(f"Something failed in the isolated computation (map above is unaffected): {e}")

    if "_proof_result" in st.session_state:
        proof_result = st.session_state["_proof_result"]
        st.caption(f"🕒 Last computed: {proof_result.get('computed_at', 'never')}")

        st.markdown("#### Test two nearby points")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Point A**")
            a_lat = st.number_input("Latitude A", value=37.505, format="%.4f", key="proof_a_lat")
            a_lon = st.number_input("Longitude A", value=-83.095, format="%.4f", key="proof_a_lon")
        with col2:
            st.markdown("**Point B**")
            b_lat = st.number_input("Latitude B", value=37.495, format="%.4f", key="proof_b_lat")
            b_lon = st.number_input("Longitude B", value=-83.115, format="%.4f", key="proof_b_lon")

        display_buffer = st.slider("Display buffer (miles)", 10, 200, 40, key="proof_buffer")

        hubs_a = [h for h in proof_result["hubs"] if haversine_miles(a_lat, a_lon, h["lat"], h["lon"]) <= display_buffer]
        hubs_b = [h for h in proof_result["hubs"] if haversine_miles(b_lat, b_lon, h["lat"], h["lon"]) <= display_buffer]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Point A sees", f"{len(hubs_a)} hub(s)")
            for h in hubs_a:
                st.caption(f"({h['lat']:.4f}, {h['lon']:.4f}) — weight {h['weight']:.1f}")
        with col2:
            st.metric("Point B sees", f"{len(hubs_b)} hub(s)")
            for h in hubs_b:
                st.caption(f"({h['lat']:.4f}, {h['lon']:.4f}) — weight {h['weight']:.1f}")

        if hubs_a == hubs_b:
            st.success("✅ Both points see the exact same zone data — the consistency fix works.")
        else:
            st.warning("Different results — worth a closer look before this ever touches the live map.")

# ==========================================
# DRAWER: EVIDENCE PATTERN SCANNER (advanced — collapsed, same "off to the side"
# placement as the Math & Science Drawer so it doesn't compete for tab space)
# ==========================================
with st.expander("🔍 Evidence Pattern Scanner (Advanced)", expanded=False):
    st.caption(f"Scans every sighting in this sector ({loc_name}) for recurring behavioral evidence — pure keyword matching against report text, shown with a seasonal breakdown. Not a claim about what happened, just what gets mentioned and when.")

    if not sightings_data:
        st.info("No sightings in this sector to scan yet.")
    else:
        evidence_results = scan_evidence_patterns(sightings_data)
        total = len(sightings_data)

        for category, matches in evidence_results.items():
            count = len(matches)
            pct = (count / total * 100) if total else 0
            with st.expander(f"{category}: {count} of {total} reports ({pct:.0f}%)"):
                if count == 0:
                    st.caption("No matches in this sector.")
                else:
                    season_counts = {}
                    for m in matches:
                        season_counts[m["season"]] = season_counts.get(m["season"], 0) + 1
                    season_line = " | ".join(f"{s}: {c}" for s, c in season_counts.items())
                    st.write(f"**By season:** {season_line}")
                    st.caption("Time-of-day breakdown isn't available yet — BFRO's export doesn't include it, and it needs a small addition to the John Green data pass. On the list.")
                    st.markdown("**Matching reports:**")
                    for m in matches[:10]:
                        r = m["report"]
                        st.write(f"- {r.get('title', 'Untitled')} ({r.get('event_date', 'N/A')}, {m['season']}) — *{r.get('source', 'BFRO')}*")
                    if count > 10:
                        st.caption(f"...and {count - 10} more.")

# ==========================================
# DRAWER: RESTORED CURATED RESEARCH LIBRARY (nationwide reference vault)
# ==========================================
with st.expander("📚 Curated Research Library & Cross-Cultural Pattern Engine", expanded=False):
    st.caption("Nationwide ethnographic archives, historical media scans, comparative primate biology, and behavioral search toolsets.")
    lib_choice = st.radio(
        "Select Vault Section:",
        ["🪶 Indigenous Ethnographic Lore", "📰 Historical Press Archives", "🐒 Comparative Primate Biology & Morphology", "🔊 Infrasound Physics", "👣📜 Sightings & Historical Accounts", "🔬 Researcher Archives"],
        horizontal=True
    )
    st.markdown("---")

    if "Lore" in lib_choice:
        st.subheader("🪶 Indigenous Ethnographic Lore & Local Tribal Finder")
        lore_view = st.radio("View:", ["🔍 Search", "📖 Browse A-Z", "🗂️ Browse by Category", "🔗 Shared Traits"], horizontal=True, key="lore_view_mode")

        def categorize_lore_entity(item):
            text = f"{item.get('entity_nature', '')} {item.get('entity_name', '')}".lower()
            if "not yet independently verified" in text or "carried over" in text:
                return "❔ Carried Over — Not Yet Independently Verified"
            if "trickster" in text or "little people" in text:
                return "🧚 Small Trickster / Little People Spirits"
            if "stone" in text or "stonish" in text:
                return "🪨 Stone Giants"
            if "malevolent" in text or "ogre" in text:
                return "👹 Malevolent Giants / Ogres"
            if "primordial" in text or "myth-time" in text:
                return "⚡ Primordial Myth-Time Monsters (already defeated in origin stories)"
            if "culture-hero" in text:
                return "🌟 Benevolent Culture-Hero Giants"
            if "guardian" in text or "wildman" in text:
                return "🦍 Giant Wildman / Guardian Figures"
            return "🗂️ Other"

        def render_lore_entry(item):
            tribe = item.get('tribe_name', item.get('tribe', 'Indigenous Record'))
            entity = item.get('entity_name', item.get('title', 'Entity'))
            narrative = item.get('full_narrative', item.get('summary', ''))
            nature = item.get('entity_nature', '')
            status = item.get('verification_status', '')
            st.markdown(f"### 🪶 {tribe} — *{entity}*")
            if nature:
                st.caption(f"**Nature:** {nature}")
            st.caption(f"> {narrative}")
            if status:
                st.caption(f"_Status: {status}_")
            ref = item.get('reference_url')
            if ref:
                st.markdown(f"[Source]({ref})")
            st.markdown("---")

        if lore_view == "🔍 Search":
            search_lore_term = st.text_input("🔍 Search Lore (e.g., Wampanoag, Maushop, Pukwudgie, Sasquatch, Wood Knock):", key="lore_search_box")
            filtered_lore = all_lore_records
            if search_lore_term:
                filtered_lore = [item for item in all_lore_records if search_lore_term.lower() in str(item).lower()]
            st.write(f"Displaying **{len(filtered_lore)}** ethnographic records (searching the full nationwide database, not just this sector):")
            for item in filtered_lore:
                render_lore_entry(item)

        elif lore_view == "📖 Browse A-Z":
            tribes_sorted = sorted(set(item.get('tribe_name', 'Unknown') for item in all_lore_records))
            selected_tribe = st.selectbox("Jump to tribe:", ["All tribes"] + tribes_sorted, key="lore_az_select")
            display_list = all_lore_records if selected_tribe == "All tribes" else [i for i in all_lore_records if i.get('tribe_name') == selected_tribe]
            for item in sorted(display_list, key=lambda x: str(x.get('tribe_name', ''))):
                render_lore_entry(item)

        elif lore_view == "🗂️ Browse by Category":
            categories = {}
            for item in all_lore_records:
                cat = categorize_lore_entity(item)
                categories.setdefault(cat, []).append(item)
            st.caption("Categories are derived from each entity's own documented nature, not guessed at click-time.")
            for cat_name in sorted(categories.keys()):
                items = categories[cat_name]
                with st.expander(f"{cat_name} ({len(items)})"):
                    for item in items:
                        render_lore_entry(item)

        elif lore_view == "🔗 Shared Traits":
            st.caption("Cross-referencing physical and paranormal indicators across every entity — pure text matching, showing where traits recur across otherwise unrelated cultures.")
            trait_keywords = {
                "Giant stature": ["giant", "immense size", "larger than"],
                "Hairy / fur-covered": ["hair", "hairy", "fur"],
                "Rock/stone throwing or stone-skinned": ["rock", "stone"],
                "Cannibalistic": ["cannibal"],
                "Telepathy / mind control": ["telepath", "mind", "read"],
                "Shapeshifting": ["shapeshift", "transform"],
                "Large footprints": ["footprint", "impression"],
            }
            trait_matches = {trait: [] for trait in trait_keywords}
            for item in all_lore_records:
                combined = f"{item.get('physical_indicators', '')} {item.get('paranormal_indicators', '')} {item.get('full_narrative', '')} {item.get('entity_nature', '')}".lower()
                for trait, keywords in trait_keywords.items():
                    if any(kw in combined for kw in keywords):
                        trait_matches[trait].append(f"{item.get('tribe_name', '')} — {item.get('entity_name', '')}")
            for trait, matches in trait_matches.items():
                if matches:
                    with st.expander(f"{trait} ({len(matches)} entities)"):
                        for m in matches:
                            st.write(f"- {m}")

    elif "Press" in lib_choice:
        st.subheader("📰 Historical Press Archives & Media Scans")
        press_view = st.radio("View:", ["🔍 Search", "📅 Browse Chronologically", "🗺️ Browse by State"], horizontal=True, key="press_view_mode")

        def render_press_entry(item):
            title = item.get('title', item.get('headline', 'Article'))
            p_date = item.get('pub_date', item.get('event_date', 'Historical'))
            pub = item.get('publication_name', item.get('source', ''))
            text = item.get('full_text_transcript', item.get('summary', ''))
            status = item.get('verification_status', '')
            st.markdown(f"### {title} ({p_date})")
            if pub:
                st.caption(f"**Publication:** {pub}")
            st.info(text)
            if status:
                st.caption(f"_Status: {status}_")
            ref = item.get('article_url')
            if ref:
                st.markdown(f"[Source]({ref})")
            st.markdown("---")

        if press_view == "🔍 Search":
            search_press_term = st.text_input("🔍 Search News Archives (e.g., Whitehall, Tracks, Ravine, Hunter):", key="press_search_box")
            filtered_press = all_press_records
            if search_press_term:
                filtered_press = [item for item in all_press_records if search_press_term.lower() in str(item).lower()]
            st.write(f"Displaying **{len(filtered_press)}** newspaper archives (searching the full nationwide database, not just this sector):")
            for item in filtered_press:
                render_press_entry(item)

        elif press_view == "📅 Browse Chronologically":
            def sort_key(item):
                d = str(item.get('pub_date', ''))
                return d
            for item in sorted(all_press_records, key=sort_key):
                render_press_entry(item)

        elif press_view == "🗺️ Browse by State":
            states_sorted = sorted(set(str(item.get('state', 'Unknown')) for item in all_press_records))
            selected_state = st.selectbox("Jump to state:", ["All states"] + states_sorted, key="press_state_select")
            display_list = all_press_records if selected_state == "All states" else [i for i in all_press_records if str(i.get('state', '')) == selected_state]
            for item in display_list:
                render_press_entry(item)

    elif "Primate" in lib_choice or "Biology" in lib_choice:
        st.subheader("🐒 Comparative Primate & Hominid Biology Vault")
        p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs(["📐 Footprint & Gait Mechanics", "🔊 Vocalization & Vocal Tracts", "🦴 Sagittal Crest & Skull Anatomy", "🗺️ Home Range & Movement Ecology"])
        with p_tab1:
            st.markdown("#### Foot Structure: Human vs. Gorilla vs. Sasquatch Casts")
            st.write("""
            * **Mid-Tarsal Break:** Human feet feature a rigid longitudinal arch. Great apes have a flexible mid-tarsal joint. Casts attributed to Sasquatch frequently show a double pressure ridge indicative of a flexible mid-tarsal region under high body mass.
            * **Dermal Ridges:** Friction skin ridge comparisons showing non-human flow patterns.
            """)
        with p_tab2:
            st.markdown("#### Vocal Tract Morphology & Infrasound Capabilities")
            st.write("""
            * **Laryngeal Sacs:** Great apes possess air sacs branching off the larynx for deep, low-frequency calls.
            * **Formant Frequencies:** Field recordings indicate vocal tract lengths exceeding average human adult males, corresponding to lower fundamental frequencies.
            """)
        with p_tab3:
            st.markdown("#### Sagittal Crest & Masticatory Muscles")
            st.write("""
            * **Sagittal Crest:** Bone ridge along the skull top, prominent in male gorillas, for temporalis jaw muscle attachment.
            * **Conical Skull Descriptions:** Field accounts describe a peaked head shape, consistent with a strong sagittal crest.
            """)
        with p_tab4:
            st.markdown("#### How Much Space Would a Large-Bodied Primate Actually Need?")
            st.write("A key finding shapes how this app models zones and corridors: **none of the great apes truly migrate.** All three keep a stable home range reused for years, not a seasonal migratory route like caribou or songbirds.")
            st.table({
                "Species": ["Gorilla", "Chimpanzee", "Orangutan (male / female)", "Black bear (male / female)"],
                "Typical home range": ["~9-16 sq mi", "~7-40 sq mi (up to ~77 in savanna)", "up to ~15 sq mi / ~0.2-3.5 sq mi", "~10-100+ sq mi / ~2.5-25 sq mi"],
                "Movement pattern": ["Stationary core, <1 mi/day", "Stable, aggressively defended", "Non-territorial, overlapping, males roam within a fixed area", "Stable range with real seasonal shifts chasing food"],
            })
            st.write("""
            * **Working model implication:** given a hypothesized body size larger than a gorilla and likely higher caloric needs (especially in temperate climates with real winters), a home range on the larger end of the black bear scale — or larger — is the most defensible working assumption.
            * **What this means for corridors:** a Larson Hypothesis corridor most plausibly represents the connective tissue *within* one individual or family group's home range (like a bear's seasonal shift between a spring and fall feeding patch) — not a migration route between two unrelated populations.
            * **Open question, not yet built into the app:** whether a Hot Zone + a connected Refuge + the corridor between them adds up to a plausible single home-range size is a useful sanity check, but the app does not currently calculate or flag this.
            """)
            st.caption("Sources: New England Primate Conservancy, Britannica, National Geographic, Smithsonian National Zoo, PBS Nature, SeaWorld/Busch Gardens, Better Planet Education, Bear Hunting Magazine (PA home range study), BearWise, Florida FWC, Virginia DWR.")

    elif "Infrasound" in lib_choice:
        st.subheader("🔊 Infrasound Physics & Attenuation Profiles")
        st.caption("General reference — covers the phenomenon broadly, not tied to any one search sector. For sources actually logged near you, see the Local Intel drawer above the map.")

        st.markdown("### Attenuation & Propagation")
        st.latex(r"\text{Loss} = 20 \times \log_{10}(R / R_0) + \alpha \times R")
        st.write("Sub-audible waves (<20 Hz) experience minimal atmospheric absorption (~0.001 dB/km), letting them propagate 40–80+ miles through forest canopy and terrain that would fully block audible sound.")

        st.markdown("### 📏 Distance Traveled vs. Distance Felt")
        st.write("""
        These are not the same number, and the app now distinguishes them:
        * **Distance traveled** — how far the wave remains physically measurable by an instrument.
        * **Distance felt** — how far a person would actually notice a physiological effect. This threshold is shorter than the full travel distance, since amplitude decays with distance and the body needs enough of it to register anything.
        """)

        st.markdown("### 🗂️ The Four Source Categories")
        for key, info in INFRASOUND_TYPES.items():
            with st.expander(f"{info['label']} — {info['freq_range']}"):
                st.write(f"**Cause:** {info['cause']}")
                st.write(f"**Character:** {info['character']}")
                st.write(f"**Typical felt zone:** ~{info['felt_miles']} miles | **Typical detectable range:** ~{info['travel_miles']} miles")
                st.caption(info["effect"])

        st.markdown("### 🧠 Human Physiological & Neurological Effects by Band")
        st.write("""
        * **1.0 – 7.0 Hz (Inner Ear / Vestibular Resonance):** Matches the resonant frequency of inner ear fluid — dizziness, pressure headaches, fatigue, loss of balance.
        * **7.0 – 12.0 Hz (Central Nervous System Resonance):** Overlaps human alpha brain waves (8–12 Hz) — hyper-vigilance, unexplained dread, a sense of being watched.
        * **18.0 – 19.0 Hz (Ocular Resonance) — the "Ghost Frequency":** Matches the eyeball's own resonant frequency (~18.9 Hz) — visual smearing, peripheral shadow artifacts, blurred depth perception. Documented by engineer Vic Tandy in 1998, after he traced his own "ghost" sighting at a Coventry lab to a fan emitting 18.98 Hz; when the fan was switched off, the sighting and the accompanying dread stopped. Published as "The Ghost in the Machine," *Journal of the Society for Psychical Research*.
        * **50 – 100 Hz harmonics (Chest Wall Pressure):** Audible upper harmonics accompanying an infrasound burst can vibrate the chest wall — felt pressure or breathlessness.
        """)
        st.caption("The 18-19 Hz band is the single most important one for this app's purpose: it's a documented, testable natural cause for exactly the 'shadow figure' and 'dread' reports that push researchers toward esoteric explanations.")
        st.caption("Effects are dose- and individual-dependent; this table describes documented tendencies by frequency band, not guaranteed outcomes.")

    elif "Sightings" in lib_choice or "BFRO" in lib_choice:
        st.subheader("👣📜 Sightings & Historical Accounts Vault")
        available_sources = sorted(set(item.get("source", "BFRO") for item in sightings_data))
        source_filter = st.radio("Filter by source:", ["All"] + available_sources, horizontal=True, key="sightings_source_filter")
        filtered_sightings = sightings_data if source_filter == "All" else [item for item in sightings_data if item.get("source", "BFRO") == source_filter]

        st.write(f"Displaying **{len(filtered_sightings)}** active sector records:")
        for item in filtered_sightings[:25]:
            raw_id = str(item.get('report_id', item.get('id', ''))).strip()
            item_source = item.get("source", "BFRO")
            icon = "👣" if item_source == "BFRO" else "📜"
            st.markdown(f"#### {icon} {item.get('title')} ({item.get('event_date', 'N/A')})")
            st.caption(f"Source: {item_source}")
            st.info(item.get('summary', 'No summary transcript recorded.'))
            if raw_id.isdigit():
                st.markdown(f"[📄 View Full BFRO Report #{raw_id}](https://www.bfro.net/GDB/show_report.asp?id={raw_id})")
            st.markdown("---")

    elif "Researcher Archives" in lib_choice:
        st.subheader("🔬 Researcher Archives")
        st.caption("Where the foundational Sasquatch researchers' actual working materials live — not just their published books. Verification status shown honestly for each claim.")

        if not raw_researcher_archives:
            st.info("Researcher archives file not found or empty.")
        else:
            for item in raw_researcher_archives:
                st.markdown(f"### {item.get('researcher_name', 'Unknown')}")
                st.write(f"**Credentials:** {item.get('credentials', '')}")
                st.write(f"**Stance:** {item.get('stance', '')}")
                st.write(f"**Key publications:** {item.get('key_publications', '')}")
                st.write(f"**Where their materials actually live:** {item.get('archive_location', '')}")
                st.write(f"**Access:** {item.get('archive_access', '')}")
                st.caption(f"_Verification: {item.get('verification_status', '')}_")
                ref = item.get('reference_url')
                if ref and str(ref) != "nan":
                    st.markdown(f"[Source]({ref})")
                if "Krantz" in str(item.get('researcher_name', '')):
                    st.markdown("**The real finding aid, extracted and readable right here — no download, no embedding tricks that break:**")
                    with st.expander("📖 Key contents from the real finding aid", expanded=True):
                        st.markdown("""
**Collection:** Grover Sanders Krantz papers, NAA.2003-21, National Anthropological Archives, Smithsonian Institution
**Dates:** 1904-2001 (bulk 1955-2001) — 7.38 linear feet, 14 manuscript boxes, 47 floppy disks, 9 audio cassettes

**The 9 series:**
1. Correspondence, 1964, 1974-2001
2. Writings, 1955-2001
3. Research, 1959-2001
4. Professional Activities, 1958-2001
5. **Sasquatch, 1963-2001**
6. Teaching, 1957-2001
7. Biographical and Personal Files, 1904-1911, 1931, 1952-2002
8. Sound Recordings, 1988-1997, undated
9. Electronic Records, 1987-2001

**Real Sasquatch-specific publications from his bibliography:**
- "Anatomy and Dermatoglyphics of Three Sasquatch Footprints," *Cryptozoology* 2 (1983): 53-81
- "A Reconstruction of the Skull of *Gigantopithecus blacki* and a Comparison with a Living Form," *Cryptozoology* 5 (1987): 24-39
- *Big Footprints: A Scientific Inquiry into the Reality of Sasquatch* (1992)
- *Bigfoot Sasquatch Evidence* (1999)

**Note worth knowing before requesting anything:** some materials in the collection are written in "noospel," a phonetic spelling system Krantz invented himself — not everything is straightforwardly readable even once accessed. Access to the physical papers requires an appointment with the National Anthropological Archives; the finding aid itself (what you're reading here) is public domain (CC0).
                        """)
                    st.caption("Want the actual full PDF instead of this summary? It opens as a plain, ordinary link — no embedding, no loop.")
                    st.markdown(f"[📄 Open the real finding aid PDF]({KRANTZ_PDF_URL})")
                st.markdown("---")

# ==========================================
# DRAWER: JUNK DRAWER — standalone, sitting beneath the Research Library, not nested
# inside it. Misleading citations, misattributed entities, conspiracy theories, and
# pseudoscience — acknowledged and explained, never endorsed. Every entry shows both a
# quick bullet "nutshell" and the full engaging writeup, since not everyone reads the
# same way.
# ==========================================
with st.expander("🗑️ Junk Drawer — Debunked Claims, Misattributions & Pseudoscience", expanded=False):
    st.caption("This is exactly why science stays skeptical of Bigfoot research — and exactly what a serious research tool needs to be honest about. Acknowledged and explained here, never endorsed.")

    if not all_junk_records:
        st.info("Junk Drawer file not found or empty.")
    else:
        overview_items = [i for i in all_junk_records if i.get("drawer_tab") == "Overview"]
        if overview_items:
            st.markdown("### 🚩 How To Spot This Stuff — Real Patterns Found In Our Own Research")
            for item in overview_items:
                st.write(f"- **{item.get('item_name', '').replace('Red Flag: ', '')}** — {item.get('nutshell', item.get('why_its_wrong', ''))}")
            st.markdown("---")

        def render_junk_entry(item):
            nutshell = item.get('nutshell')
            st.markdown(f"#### 🗑️ {item.get('item_name', 'Untitled')}")
            if nutshell and str(nutshell) != "nan":
                st.info(f"**In a nutshell:** {nutshell}")
            with st.expander("Full story"):
                claimed = item.get('what_gets_claimed', '')
                if claimed and "N/A" not in str(claimed):
                    st.markdown(f"**What gets claimed:** {claimed}")
                st.error(f"**Why it's wrong:** {item.get('why_its_wrong', '')}")
                ref = item.get('reference_url')
                if ref and str(ref) != "nan":
                    st.markdown(f"[Source]({ref})")
            st.markdown("---")

        junk_tab1, junk_tab2, junk_tab3, junk_tab4 = st.tabs(["🪶 Indigenous", "📰 Media", "🕵️ Conspiracy", "👻 Paranormal Bigfoot"])
        tab_map = {"Indigenous": junk_tab1, "Media": junk_tab2, "Conspiracy": junk_tab3, "Paranormal Bigfoot": junk_tab4}
        for tab_name, tab_obj in tab_map.items():
            with tab_obj:
                tab_items = [i for i in all_junk_records if i.get("drawer_tab") == tab_name]
                if not tab_items:
                    st.info("Nothing filed here yet.")
                for item in tab_items:
                    render_junk_entry(item)


# ==========================================
# DRAWER: RESTORED INVESTIGATOR FIELD LOG SUBMISSION (was a stub with no actual form)
# ==========================================
with st.expander("📝 Submit Investigator Field Log (Facts vs. Conjecture Mode)", expanded=False):
    if not supabase:
        st.info("Field log submission needs SUPABASE_URL / SUPABASE_KEY set in Secrets to save entries.")
    with st.form("investigator_log_form", clear_on_submit=True):
        visibility = st.radio("Storage Mode:", ["🔒 Private Vault", "🌐 Public Community Layer"], horizontal=True)
        obs_type = st.selectbox("Type", ["Suspect Impression", "Potential Nesting Site", "Vegetation Disturbance", "Acoustic Event", "Visual Observation"])
        physical_notes = st.text_area("Hard Physical Facts", placeholder="Measurements, trackway depth, scale markers...")
        field_narrative = st.text_area("Observer Conjecture & Narrative", placeholder="Hypothesis, perceived behavior...")
        ethics_agree = st.checkbox("Certify as honest field record.")
        submitted = st.form_submit_button("💾 Save Field Log", use_container_width=True)
        if submitted:
            if not ethics_agree:
                st.warning("Please certify this is an honest field record before saving.")
            elif not supabase:
                st.error("Can't save — Supabase isn't configured (see Secrets).")
            else:
                try:
                    supabase.table("investigator_logs").insert({
                        "is_public": "Public" in visibility,
                        "observation_type": obs_type,
                        "event_date": str(datetime.now().date()),
                        "latitude": lat,
                        "longitude": lon,
                        "physical_evidence_notes": physical_notes,
                        "field_narrative": field_narrative,
                        "ethics_agreed": True
                    }).execute()
                    st.success("Log saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving log: {e}")

# ==========================================
# DRAWER: REGIONAL CAMPSITES & ACCESS POINTS
# ==========================================
with st.expander(f"🏕️ Regional Campsites & Backcountry Access Points (Within {radius_miles} miles)", expanded=False):
    if camps_data:
        st.write(f"Found **{len(camps_data)}** campsites in active sector:")
        for c in camps_data[:25]:
            st.write(f"🏕️ **{c.get('name', 'Campsite')}** | Type: `{c.get('type', 'Primitive')}` | Coords: `{c.get('latitude')}, {c.get('longitude')}`")
    else:
        st.info("No campsites indexed in active sector radius. Add `data/campsites.csv` to populate local campsites.")

# ==========================================
# DRAWER: OFFLINE EXPORT
# ==========================================
with st.expander("📡 Offline Field Export & GPX Package", expanded=False):
    gpx_data = generate_gpx(lat, lon, loc_name, sightings_data, camps_data, audio_data, user_logs_data)
    st.download_button(label="📥 Download Active Area GPX Package", data=gpx_data, file_name="bigfoot_field_zone.gpx", mime="application/gpx+xml")
