import streamlit as st
import time
from controller import TrafficSignalController

# --- Initialize Session State ---
if 'current_direction_index' not in st.session_state:
    st.session_state.current_direction_index = 0
if 'remaining_time' not in st.session_state:
    st.session_state.remaining_time = 0
if 'signal_data' not in st.session_state:
    st.session_state.signal_data = None
if 'cycle_completed' not in st.session_state:
    st.session_state.cycle_completed = False


def main():
    # MUST be the first Streamlit command
    st.set_page_config(
        page_title="AI Traffic Controller",
        layout="wide",
        page_icon="🚦"
    )

    # Custom CSS (now comes after set_page_config)
    st.markdown("""
        <style>
        .header-text { 
            font-size:32px !important;
            font-weight:bold !important;
            color: #2E86C1 !important;
        }
        .metric-label {
            font-size:14px !important;
            color: #666666 !important;
        }
        .green-light {
            background-color: #D5F5E3;
            border-radius: 10px;
            padding: 15px;
            border: 2px solid #28B463;
        }
        .red-light {
            background-color: #FADBD8;
            border-radius: 10px;
            padding: 15px;
            border: 2px solid #C0392B;
        }
        .progress-bar {
            height: 10px !important;
            border-radius: 5px;
        }
        .direction-box {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .timer-text {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Rest of your code remains the same...
    st.markdown('<p class="header-text">🚦 AI-Powered Traffic Signal Control System</p>', unsafe_allow_html=True)
    st.markdown("""
        **Real-time traffic management** using YOLOv8 vehicle detection.  
        The system automatically adjusts signal timings based on detected vehicle density.
    """)

    # Initialize controller
    controller = TrafficSignalController(model_name="yolov8m")

    # Control Section
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("▶️ Start New Detection Cycle", use_container_width=True):
                run_detection(controller)
        with col2:
            if st.session_state.signal_data:
                total_vehicles = sum(st.session_state.signal_data['counts'].values())
                st.metric("🚗 Total Vehicles Detected", total_vehicles)

    # Display current state if available
    if st.session_state.signal_data:
        run_signal_simulation(controller)

def run_detection(controller):
    """Run vehicle detection and update session state"""
    with st.status("🚦 **Processing Traffic Data**", expanded=True) as status:
        st.write("📸 Capturing lane images...")
        counts, timings, images = controller.run_control_cycle()

        st.write("🔍 Analyzing vehicle density...")
        time.sleep(0.5)  # Simulate processing

        st.write("📊 Calculating optimal signal timings...")
        time.sleep(0.5)

        st.session_state.signal_data = {
            'counts': counts,
            'timings': timings,
            'images': images
        }

        st.session_state.current_direction_index = 0
        st.session_state.remaining_time = timings['Direction_1']
        st.session_state.cycle_completed = False

        status.update(label="✅ Detection Complete!", state="complete")


def show_current_signal_state():
    """Enhanced visualization of signal states"""
    counts = st.session_state.signal_data['counts']
    images = st.session_state.signal_data['images']
    timings = st.session_state.signal_data['timings']

    # Use a single container for all signal states
    with st.container():
        st.subheader("Real-time Signal Status")
        cols = st.columns(4)

        for idx in range(4):  # Explicitly loop through 4 directions
            direction = f"Direction_{idx + 1}"
            vehicle_count = counts[direction]
            is_green = idx == st.session_state.current_direction_index

            with cols[idx]:  # Use index-based column access
                with st.expander(f"Lane {idx+1} | 🚗 {vehicle_count}", expanded=True):
                    # Use a placeholder for dynamic image updates
                    img_placeholder = st.empty()
                    img_placeholder.image(images[idx], use_container_width=True)

                    # Use single element for status display
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

                    # Progress bar handling
                    if is_green:
                        progress = st.session_state.remaining_time / timings[direction]
                        st.progress(progress)
                    else:
                        # Clear any existing progress bar
                        st.empty()


def countdown_and_cycle_signals(timings):
    """Enhanced countdown with visual feedback"""
    placeholder = st.empty()

    if st.session_state.remaining_time > 0:
        with placeholder.container():
            show_current_signal_state()
            time.sleep(1)
            st.session_state.remaining_time -= 1
            st.rerun()
    else:
        handle_lane_switch(timings)


def handle_lane_switch(timings):
    """Handle lane switching with visual feedback"""
    st.session_state.current_direction_index += 1

    if st.session_state.current_direction_index >= 4:
        # Reset to first direction instead of completing cycle
        st.session_state.current_direction_index = 0
        st.session_state.remaining_time = timings['Direction_1']
        st.rerun()
    else:
        next_dir = f"Direction_{st.session_state.current_direction_index + 1}"
        st.session_state.remaining_time = timings[next_dir]
        st.rerun()

def run_signal_simulation(controller):
    """Main simulation control flow"""
    show_current_signal_state()
    countdown_and_cycle_signals(st.session_state.signal_data['timings'])


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

    # Calculate time until target_index gets green
    if current_index < target_index:
        # Add remaining time of current direction
        wait_time += st.session_state.remaining_time
        # Add full durations of intermediate directions
        for i in range(current_index + 1, target_index):
            wait_time += timings[f"Direction_{i + 1}"]
    else:
        # Add remaining time of current direction
        wait_time += st.session_state.remaining_time
        # Add all directions after current until end
        for i in range(current_index + 1, 4):
            wait_time += timings[f"Direction_{i + 1}"]
        # Add directions from start to target
        for i in range(0, target_index):
            wait_time += timings[f"Direction_{i + 1}"]

    return wait_time


# def run_signal_simulation(controller):
#     """Main simulation control flow"""
#     if st.session_state.cycle_completed:
#         run_detection(controller)
#         st.rerun()
#     else:
#         show_current_signal_state()
#         countdown_and_cycle_signals(st.session_state.signal_data['timings'])


if __name__ == "__main__":
    main()