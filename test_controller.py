from controller import TrafficSignalController

# Initialize controller
controller = TrafficSignalController()

# Run one complete detection + timing calculation cycle
try:
    counts, timings, images = controller.run_control_cycle()

    # Show results
    print("\nVehicle Counts:", counts)
    print("Signal Timings:", timings)

except Exception as e:
    print(f"Error: {e}")
