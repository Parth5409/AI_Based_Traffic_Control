import streamlit as st
import time
from controller import TrafficSignalController
import asyncio
import torch
import nest_asyncio

# Fix for asyncio runtime error
try:
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Ensure PyTorch is properly initialized
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")

# --- Initialize Session State ---
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


def main():
    st.set_page_config(
        page_title="AI Traffic Controller",
        layout="wide",
        page_icon="🚦"
    )

    # [CSS styles remain unchanged]
    st.markdown("""
        <style>
        .header-text { 
            font-size: 42px !important;
            font-weight: 800 !important;
            background: linear-gradient(45deg, #00B4DB, #0083B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 30px !important;
            padding: 20px;
        }
        .subtitle-text {
            font-size: 18px !important;
            color: #E0E0E0 !important;
            text-align: center;
            margin-bottom: 30px !important;
            line-height: 1.6 !important;
        }
        .metric-label {
            font-size: 16px !important;
            color: #E0E0E0 !important;
            font-weight: 500 !important;
        }
        .green-light {
            background: linear-gradient(145deg, #E9F7EF, #D5F5E3);
            border-radius: 15px;
            padding: 20px;
            border: 2px solid #28B463;
            box-shadow: 0 4px 15px rgba(40, 180, 99, 0.1);
            transition: all 0.3s ease;
        }
        .red-light {
            background: linear-gradient(145deg, #FDEDEC, #FADBD8);
            border-radius: 15px;
            padding: 20px;
            border: 2px solid #C0392B;
            box-shadow: 0 4px 15px rgba(192, 57, 43, 0.1);
            transition: all 0.3s ease;
        }
        .progress-bar {
            height: 12px !important;
            border-radius: 6px !important;
            background-color: #F0F0F0 !important;
        }
        .direction-box {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        .timer-text {
            color: #2C3E50 !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            margin-top: 8px !important;
        }
        .stButton > button {
            background: linear-gradient(145deg, #00B4DB, #0083B0);
            color: white !important;
            border: none !important;
            padding: 15px 25px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(0, 180, 219, 0.2);
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 180, 219, 0.3);
            background: linear-gradient(145deg, #0083B0, #00B4DB);
        }
        .streamlit-expanderHeader {
            display: none;
        }
        .stSubheader {
            font-size: 24px !important;
            color: #00B4DB !important;
            font-weight: 600 !important;
            margin-bottom: 20px !important;
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px !important;
            color: #00B4DB !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 16px !important;
            color: #E0E0E0 !important;
        }
        div[data-testid="stImage"] {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="header-text">🚦 AI-Powered Traffic Signal Control System</p>', unsafe_allow_html=True)
    st.markdown("""
        <p class="subtitle-text">
            <strong>Real-time traffic management</strong> using YOLOv8 vehicle detection.<br>
            The system automatically adjusts signal timings based on detected vehicle density.
        </p>
    """, unsafe_allow_html=True)

    # Control Section
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Start New Detection Cycle", use_container_width=True) or st.session_state.auto_restart:
                st.session_state.auto_restart = False
                run_detection(st.session_state.controller)

    # Display total vehicles metric separately
    if st.session_state.signal_data:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            total_vehicles = sum(st.session_state.signal_data['counts'].values())
            st.metric("🚗 Total Vehicles Detected", total_vehicles)

        placeholder = st.empty()
        with placeholder.container():
            show_current_signal_state()
            countdown_and_cycle_signals(st.session_state.signal_data['timings'])


def run_detection(controller):
    """Run vehicle detection and update session state"""
    with st.status("🚦 **Processing Traffic Data**", expanded=True) as status:
        st.write("📸 Capturing lane images...")
        counts, timings, images = controller.run_control_cycle()

        st.write("🔍 Analyzing vehicle density...")
        time.sleep(0.5)

        st.write("📊 Calculating optimal signal timings...")
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
    """Enhanced visualization of signal states"""
    counts = st.session_state.signal_data['counts']
    images = st.session_state.signal_data['images']
    timings = st.session_state.signal_data['timings']

    cols = st.columns(4)
    for idx in range(4):
        direction = f"Direction_{idx + 1}"
        vehicle_count = counts[direction]
        is_green = idx == st.session_state.current_direction_index

        with cols[idx]:
            img_placeholder = st.empty()
            img_placeholder.image(images[idx], use_container_width=True)

            status_placeholder = st.empty()
            if is_green:
                status_placeholder.markdown(f"""
                    <div class="green-light">
                        <h3 style='color:#28B463; margin:0;'>GREEN LIGHT</h3>
                        <p class="timer-text" style='margin:0;'>⏳ Remaining: {st.session_state.remaining_time}s</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                wait_time = time_until_green(idx, timings)
                status_placeholder.markdown(f"""
                    <div class="red-light">
                        <h3 style='color:#C0392B; margin:0;'>RED LIGHT</h3>
                        <p class="timer-text" style='margin:0;'>⌛ Next green in: {wait_time}</p>
                    </div>
                """, unsafe_allow_html=True)

            if is_green:
                progress = st.session_state.remaining_time / timings[direction]
                st.progress(progress)
            else:
                st.empty()


def countdown_and_cycle_signals(timings):
    """Enhanced countdown with visual feedback"""
    if st.session_state.remaining_time > 0:
        time.sleep(1)
        st.session_state.remaining_time -= 1
        st.rerun()
    else:
        handle_lane_switch(timings)


def handle_lane_switch(timings):
    """Handle lane switching with visual feedback"""
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
    """Formatted wait time display"""
    current_index = st.session_state.current_direction_index
    wait_time = calculate_wait_time(target_index, current_index, timings)
    return f"{wait_time}s" if isinstance(wait_time, int) else wait_time


def calculate_wait_time(target_index, current_index, timings):
    """Calculate wait time with formatted output"""
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