import streamlit as st
import time
from controller import TrafficSignalController
import asyncio
import torch
import nest_asyncio
import folium
from streamlit_folium import st_folium
import json
import pandas as pd
import requests
from typing import Tuple, Optional

# Fix for asyncio runtime error
try:
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")

# Initialize Session State
if 'current_direction_index' not in st.session_state:
    st.session_state.current_direction_index = 0
if 'remaining_time' not in st.session_state:
    st.session_state.remaining_time = 0
if 'signal_data' not in st.session_state:
    st.session_state.signal_data = None
if 'cycle_completed' not in st.session_state:
    st.session_state.cycle_completed = False
if 'controller' not in st.session_state:
    st.session_state.controller = TrafficSignalController(model_name="yolov8m")
if 'auto_restart' not in st.session_state:
    st.session_state.auto_restart = False
if 'page' not in st.session_state:
    st.session_state.page = None
if 'intersection_selected' not in st.session_state:
    st.session_state.intersection_selected = False
if 'intersection_confirmed' not in st.session_state:
    st.session_state.intersection_confirmed = False
if 'intersection_coords' not in st.session_state:
    st.session_state.intersection_coords = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [20.5937, 78.9629]

def verify_intersection(lat: float, lng: float) -> Tuple[bool, Optional[str], Optional[Tuple[float, float]]]:
    """Verify if coordinates represent a 4-way intersection using Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
        way(around:100,{lat},{lng})[highway];
        node(around:100,{lat},{lng})[highway=traffic_signals];
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.post(overpass_url, data=query)
        data = response.json()
        
        # Find nearest intersection node
        intersections = []
        roads = [elem for elem in data['elements'] if elem['type'] == 'way']
        
        if len(roads) >= 4:
            # Find nodes where multiple roads meet
            node_counts = {}
            for road in roads:
                for node in road.get('nodes', []):
                    node_counts[node] = node_counts.get(node, 0) + 1
            
            # Get coordinates of potential intersections
            for elem in data['elements']:
                if elem['type'] == 'node':
                    if node_counts.get(elem['id'], 0) >= 3:  # Node where 3 or more roads meet
                        distance = calculate_distance(lat, lng, elem['lat'], elem['lon'])
                        if distance <= 0.1:  # Within 100 meters
                            intersections.append((elem['lat'], elem['lon'], distance))
            
            if intersections:
                # Get the nearest intersection
                nearest = min(intersections, key=lambda x: x[2])
                return True, None, (nearest[0], nearest[1])
            
        return False, "No 4-way intersection found nearby", None
    except Exception as e:
        return False, f"Error verifying intersection: {str(e)}", None

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def create_map():
    """Create and display the map for intersection selection"""
    st.markdown('<p class="subtitle-text">Select a 4-way intersection on the map</p>', unsafe_allow_html=True)
    
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Add zoom control with custom position
    m.add_child(folium.LayerControl(position='topright'))
    m.add_child(folium.LatLngPopup())
    
    # If there's a detected intersection, show it
    if 'detected_intersection' in st.session_state:
        folium.CircleMarker(
            location=st.session_state.detected_intersection,
            radius=8,
            color='#2563eb',
            fill=True,
            popup='Detected Intersection'
        ).add_to(m)
    
    map_data = st_folium(
        m,
        height=500,
        width="100%",
        returned_objects=["last_clicked"]
    )
    
    if (map_data is not None and 
        "last_clicked" in map_data and 
        map_data["last_clicked"] is not None):
        
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        
        is_intersection, error_msg, intersection_coords = verify_intersection(lat, lng)
        
        if is_intersection and intersection_coords:
            st.session_state.detected_intersection = intersection_coords
            st.session_state.temp_coords = intersection_coords
            
            with st.container():
                st.markdown(f"""
                    <div class="status-box">
                        <span class="status-icon">📍</span>
                        <span class="status-text">Nearest intersection found at: {intersection_coords[0]:.6f}, {intersection_coords[1]:.6f}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="stSuccess">
                        <span>✅</span>
                        <span>Verified 4-way intersection detected! Starting simulation...</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Automatically start simulation when intersection is found
                st.session_state.intersection_coords = st.session_state.temp_coords
                st.session_state.map_center = st.session_state.temp_coords
                st.session_state.intersection_confirmed = True
                st.session_state.intersection_selected = True
                time.sleep(1)  # Brief pause to show the success message
                st.rerun()
        else:
            st.warning(f"⚠️ {error_msg}")
            st.info("Please select a different location or proceed manually if you're sure this is a 4-way intersection.")
            
            is_4way = st.checkbox("✓ I confirm this is a 4-way intersection")
            if is_4way:
                st.session_state.intersection_coords = [lat, lng]
                st.session_state.map_center = [lat, lng]
                st.session_state.intersection_confirmed = True
                st.session_state.intersection_selected = True
                time.sleep(1)  # Brief pause to show the confirmation
                st.rerun()

def main():
    st.set_page_config(
        page_title="AI Traffic Controller",
        layout="wide",
        page_icon="🚦",
        initial_sidebar_state="expanded"
    )

    # Load external CSS
    with open('static/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Add Google Fonts
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    st.markdown('<p class="header-text">AI Traffic Signal Control</p>', unsafe_allow_html=True)

    if st.session_state.page is None:
        st.markdown('<p class="subtitle-text">Choose your preferred operation mode</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="mode-button">
                    <div class="mode-icon">🗺️</div>
                    <div class="mode-title">Map-Based Mode</div>
                    <div class="mode-description">
                        Select an intersection from the map and monitor traffic in real-time
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Map Mode", key="map_mode", use_container_width=True):
                st.session_state.page = "map"
                st.rerun()

        with col2:
            st.markdown("""
                <div class="mode-button">
                    <div class="mode-icon">🚦</div>
                    <div class="mode-title">Direct Mode</div>
                    <div class="mode-description">
                        Start monitoring traffic immediately without location selection
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Select Direct Mode", key="direct_mode", use_container_width=True):
                st.session_state.page = "direct"
                st.rerun()
        return

    if st.session_state.page == "map":
        run_map_mode()
    else:
        run_direct_mode()

def run_map_mode():
    if not st.session_state.intersection_confirmed:
        create_map()
    else:
        show_traffic_control_interface(True)

def run_direct_mode():
    show_traffic_control_interface(False)

def show_traffic_control_interface(show_map=False):
    st.markdown("""
        <p class="subtitle-text">
            <strong>Real-time traffic management</strong> using YOLOv8 vehicle detection.<br>
            The system automatically adjusts signal timings based on detected vehicle density.
        </p>
    """, unsafe_allow_html=True)

    if show_map and st.session_state.intersection_coords:
        st.sidebar.map(pd.DataFrame({
            'lat': [st.session_state.intersection_coords[0]],
            'lon': [st.session_state.intersection_coords[1]]
        }))
        
        if st.sidebar.button("Change Intersection"):
            for key in ['intersection_selected', 'intersection_confirmed', 'intersection_coords', 'temp_coords', 'signal_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.map_center = [20.5937, 78.9629]
            st.rerun()

    if show_map and not st.session_state.get('signal_data'):
        run_detection(st.session_state.controller)
    else:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("▶️ Start New Detection Cycle", use_container_width=True):
                    st.session_state.auto_restart = False
                    run_detection(st.session_state.controller)

    if st.session_state.signal_data:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            total_vehicles = sum(st.session_state.signal_data['counts'].values())
            st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <p style="color: #000; font-size: 2rem; font-weight: 700; margin: 0.5rem 0 0 0;">🚗 Total Vehicles Detected: {total_vehicles}</p>
                </div>
            """, unsafe_allow_html=True)

        placeholder = st.empty()
        with placeholder.container():
            show_current_signal_state()
            countdown_and_cycle_signals(st.session_state.signal_data['timings'])

def run_detection(controller):
    with st.status("🚦 **Processing Traffic Data**", expanded=True) as status:
        st.markdown('<p style="color: #1e293b; font-size: 16px; font-weight: 500;">📸 Capturing lane images...</p>', unsafe_allow_html=True)
        counts, timings, images = controller.run_control_cycle()

        st.markdown('<p style="color: #1e293b; font-size: 16px; font-weight: 500;">🔍 Analyzing vehicle density...</p>', unsafe_allow_html=True)
        time.sleep(0.5)

        st.markdown('<p style="color: #1e293b; font-size: 16px; font-weight: 500;">📊 Calculating optimal signal timings...</p>', unsafe_allow_html=True)
        time.sleep(0.5)

        st.session_state.signal_data = {
            'counts': counts,
            'timings': timings,
            'images': images
        }

        st.session_state.current_direction_index = 0
        st.session_state.remaining_time = timings['Direction_1']

        status.update(label="✅ Detection Complete!", state="complete")

def show_current_signal_state():
    counts = st.session_state.signal_data['counts']
    images = st.session_state.signal_data['images']
    timings = st.session_state.signal_data['timings']

    cols = st.columns(4)
    for idx in range(4):
        direction = f"Direction_{idx + 1}"
        vehicle_count = counts[direction]
        is_green = idx == st.session_state.current_direction_index

        with cols[idx]:
            st.markdown(f"""
                <div class="direction-box">
                    <h3 style="margin:0; color:#37474F; text-align:center; font-size:20px;">
                        Direction {idx + 1}
                    </h3>
                    <p style="text-align:center; color:#78909C; margin:5px 0;">
                        Vehicles: {vehicle_count}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            img_placeholder = st.empty()
            img_placeholder.image(images[idx], use_container_width=True)

            status_placeholder = st.empty()
            if is_green:
                status_placeholder.markdown(f"""
                    <div class="green-light">
                        <h3 style='color:#4CAF50; margin:0; text-align:center;'>GREEN SIGNAL</h3>
                        <p class="timer-text" style='margin:5px 0; text-align:center;'>
                            ⏳ {st.session_state.remaining_time}s remaining
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                wait_time = time_until_green(idx, timings)
                status_placeholder.markdown(f"""
                    <div class="red-light">
                        <h3 style='color:#F44336; margin:0; text-align:center;'>RED SIGNAL</h3>
                        <p class="timer-text" style='margin:5px 0; text-align:center;'>
                            ⌛ Green in {wait_time}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            if is_green:
                progress = st.session_state.remaining_time / timings[direction]
                st.progress(progress)

def countdown_and_cycle_signals(timings):
    if st.session_state.remaining_time > 0:
        time.sleep(1)
        st.session_state.remaining_time -= 1
        st.rerun()
    else:
        handle_lane_switch(timings)

def handle_lane_switch(timings):
    st.session_state.current_direction_index += 1

    if st.session_state.current_direction_index >= 4:
        st.session_state.current_direction_index = 0
        st.session_state.auto_restart = True
        st.rerun()
    else:
        next_dir = f"Direction_{st.session_state.current_direction_index + 1}"
        st.session_state.remaining_time = timings[next_dir]
        st.rerun()

def time_until_green(target_index, timings):
    current_index = st.session_state.current_direction_index
    wait_time = calculate_wait_time(target_index, current_index, timings)
    return f"{wait_time}s" if isinstance(wait_time, int) else wait_time

def calculate_wait_time(target_index, current_index, timings):
    if target_index == current_index:
        return "Now"

    wait_time = 0
    if current_index < target_index:
        wait_time += st.session_state.remaining_time
        for i in range(current_index + 1, target_index):
            wait_time += timings[f"Direction_{i + 1}"]
    else:
        wait_time += st.session_state.remaining_time
        for i in range(current_index + 1, 4):
            wait_time += timings[f"Direction_{i + 1}"]
        for i in range(0, target_index):
            wait_time += timings[f"Direction_{i + 1}"]

    return wait_time

if __name__ == "__main__":
    main()