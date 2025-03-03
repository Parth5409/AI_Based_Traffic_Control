import streamlit as st
import time
from controller import TrafficSignalController

# --- Setup initial states when app loads ---
if 'current_direction_index' not in st.session_state:
    st.session_state.current_direction_index = 0  # Start from first direction
if 'remaining_time' not in st.session_state:
    st.session_state.remaining_time = 0  # Time left for current green light
if 'signal_data' not in st.session_state:
    st.session_state.signal_data = None  # This will hold latest detection data
if 'cycle_completed' not in st.session_state:
    st.session_state.cycle_completed = False  # Tracks if all 4 lanes are done


# --- Main App Function ---
def main():
    st.set_page_config(page_title="AI Traffic Signal Controller", layout="wide")

    st.title("🚦 AI Traffic Signal Controller")
    st.markdown("This system detects vehicles using **YOLOv8** and adjusts signal timings based on vehicle density.")

    # Initialize controller
    controller = TrafficSignalController(model_name="yolov8m")

    if st.button("🚦 Start Detection & Control"):
        run_detection(controller)

    if st.session_state.signal_data:
        run_signal_simulation(controller)


# --- Detection Function (runs once per cycle) ---
def run_detection(controller):
    """Detect vehicles and calculate timings for 4 new random images"""
    with st.spinner("🔎 Running vehicle detection..."):
        counts, timings, images = controller.run_control_cycle()  # Your method to pick & process 4 random images

        # Store results in session state
        st.session_state.signal_data = {
            'counts': counts,
            'timings': timings,
            'images': images
        }

        # Start at first lane
        st.session_state.current_direction_index = 0
        st.session_state.remaining_time = timings['Direction_1']
        st.session_state.cycle_completed = False  # Reset cycle status
        st.success("✅ Detection completed! Starting signal control.")


# --- Simulation Function (handles green/red signal flow) ---
def run_signal_simulation(controller):
    """Manage the countdown and automatically trigger new detection after all lanes are completed"""
    if st.session_state.cycle_completed:
        run_detection(controller)
        st.rerun()
        return

    show_current_signal_state()  # Show current signal (green/red) and vehicle counts
    countdown_and_cycle_signals(st.session_state.signal_data['timings'])  # Handle countdown logic


# --- Signal State Display (live update view) ---
def show_current_signal_state():
    counts = st.session_state.signal_data['counts']
    images = st.session_state.signal_data['images']
    timings = st.session_state.signal_data['timings']

    st.subheader("🚥 Current Traffic Signal Status")

    cols = st.columns(4)

    for idx, col in enumerate(cols):
        direction = f"Direction_{idx + 1}"
        vehicle_count = counts[direction]
        is_currently_green = (idx == st.session_state.current_direction_index)

        with col:
            st.image(images[idx], caption=f"Direction {idx + 1}", use_container_width=True)
            st.markdown(f"### 🚗 {vehicle_count} vehicles")

            if is_currently_green:
                st.markdown(f"**🟢 Green Light: {st.session_state.remaining_time} sec**")
            else:
                st.markdown(f"**🔴 Red Light** (Next green in: {time_until_green(idx, timings)})")


# --- Countdown Timer and Lane Switch Logic ---
def countdown_and_cycle_signals(timings):
    """Counts down the green light, then switches to next lane"""
    if st.session_state.remaining_time > 0:
        time.sleep(1)
        st.session_state.remaining_time -= 1
        st.rerun()  # Refresh UI
    else:
        # Time over for current lane, move to next
        st.session_state.current_direction_index += 1

        if st.session_state.current_direction_index >= 4:
            # All 4 lanes completed — trigger new detection cycle
            st.session_state.cycle_completed = True
        else:
            # Start green light for next direction
            next_direction = f"Direction_{st.session_state.current_direction_index + 1}"
            st.session_state.remaining_time = timings[next_direction]

        st.rerun()


# --- Helper: Calculate wait time until a lane gets green light ---
def time_until_green(target_index, timings):
    """Calculates how long until a specific lane gets green light"""
    current_index = st.session_state.current_direction_index

    if target_index == current_index:
        return "Now"

    wait_time = st.session_state.remaining_time

    if target_index > current_index:
        for i in range(current_index + 1, target_index):
            wait_time += timings[f"Direction_{i + 1}"]
    else:
        for i in range(current_index + 1, 4):
            wait_time += timings[f"Direction_{i + 1}"]
        for i in range(0, target_index):
            wait_time += timings[f"Direction_{i + 1}"]

    return f"{wait_time} sec"


# --- Run the app ---
if __name__ == "__main__":
    main()
