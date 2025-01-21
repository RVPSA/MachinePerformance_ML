import pickle
import time
# import numpy as np
import pandas as pd
import psutil
import pyshark
from datetime import datetime


def current_milli_time():
    """
    Returns the current time in milliseconds.
    """
    return round(time.time() * 1000)


def cpu_probe(data_dict):
    """
    Reads CPU data from the system and updates the dictionary.
    """
    cpu_t = psutil.cpu_times()
    data_dict["idle_time"] = cpu_t.idle
    data_dict["usr_time"] = cpu_t.user
    data_dict["interrupt_time"] = cpu_t.softirq


def vm_probe(data_dict):
    """
    Reads VM data from the system and updates the dictionary.
    """
    vm_data = psutil.virtual_memory()
    data_dict["mem_total"] = vm_data.total
    data_dict["mem_available"] = vm_data.available
    data_dict["mem_percent"] = vm_data.percent


def pyshark_parse(data_dict, pk_capture):
    """
    Parses PyShark data into the existing dictionary.
    """
    n_packets = 0
    is_tcp = 0
    byte_size = 0
    for packet in pk_capture._packets:
        n_packets += 1
        byte_size += int(packet.captured_length)
        is_tcp += 1 if packet.transport_layer == 'TCP' else 0
    data_dict["n_packets"] = n_packets
    data_dict["tcp_packets"] = is_tcp
    data_dict["packet_size"] = byte_size


# Simulated monitor function (replace with actual monitoring logic)
def monitor():
    """
    Simulates monitoring by collecting system and network data.
    """
    # Collect system data
    data_dict = {"time": current_milli_time(), "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    cpu_probe(data_dict)
    vm_probe(data_dict)

    # Collect PyShark data
    capture = pyshark.LiveCapture(interface='wlp3s0')  # Replace 'wlp3s0' with your network interface
    try:
        capture.sniff(timeout=10)
        pyshark_parse(data_dict, capture)
    except Exception as e:
        print(f"Error in PyShark capture: {e}")
    finally:
        capture.close()

    return data_dict


def log_result(result, log_file="detection_log.txt"):
    """
    Logs the detection result to a file.
    :param result: The result from the detector (e.g., anomaly or normal)
    :param log_file: The log file to save the results
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - Detection Result: {result}\n")
    print(f"Logged: {timestamp} - Detection Result: {result}")


def load_detector(model_filename):
    """
    Loads the detector model from a pickle file.
    :param model_filename: The path to the pickle file containing the trained model
    :return: The loaded model
    """
    with open(model_filename, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Load the trained model from file
    model_filename = "trained_model.pkl"
    detector = load_detector(model_filename)
    print(f"Detector loaded from {model_filename}")

    # Start monitoring and detecting anomalies
    print("Starting monitoring and anomaly detection...")
    try:
        while True:
            # Step 1: Monitor data
            monitored_data = monitor()

            # Convert monitored data (dictionary) into a DataFrame
            monitored_df = pd.DataFrame([monitored_data])

            # Drop non-feature columns (time, datetime) if not used in training
            monitored_df = monitored_df.drop(columns=["time", "datetime"], errors="ignore")

            # Step 2: Send monitored data to the detector
            result = detector.predict(monitored_df)

            # Step 3: Log the result
            log_result(result[0])  # Assuming result is a single prediction

            # Simulate delay between monitoring iterations
            time.sleep(2)  # Adjust based on monitoring frequency
    except KeyboardInterrupt:
        print("Monitoring stopped.")
