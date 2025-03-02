import streamlit as st
from controller import TrafficSignalController

def main():
    st.set_page_config(page_title="AI Traffic Signal Controller", layout="wide")

    st.title("🚦 AI Traffic Signal Controller")

    st.markdown("""
    This system detects vehicles in images using **YOLOv8** and dynamically calculates signal timings based on vehicle density.
    """)

    controller = TrafficSignalController(model_name="yolov8m")

    if st.button("🚦 Run Control Cycle"):
        with st.spinner("Detecting vehicles and calculating timings..."):
            counts, timings, images = controller.run_control_cycle()

            st.success("Control cycle completed!")

            cols = st.columns(4)

            for idx, col in enumerate(cols):
                direction = f"Direction_{idx + 1}"
                vehicle_count = counts[direction]
                timing = timings[direction]

                with col:
                    st.image(images[idx], caption=f"🚗 {vehicle_count} vehicles", use_container_width=True)
                    st.markdown(f"**🕒 Green Time: {timing} sec**")

if __name__ == "__main__":
    main()
