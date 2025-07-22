import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from .utils import show_image


class DataManager:
    '''
    This class is used to load and visualize the data from .pkl files.
    '''

    def __init__(self, pkl_path: str):
        '''
        Initializes the DataManager class.

        Args:
            pkl_path (str): The path to the .pkl file.
        '''
        self.pkl_path = pkl_path
        self.metadata_entry, self.data_entries = self.load_objects_and_metadata()
        self.capture_idx = 0

    def load_objects(self):
        """Load the objects from the .pkl file."""
        with open(self.pkl_path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def load_objects_and_metadata(self):
        """
        Load the objects and metadata from the .pkl file.
        Returns a tuple of (metadata_entry, data_entries).
        """
        try:
            # Convert generator to list to get all objects
            all_objects = list(self.load_objects())

            if not all_objects:
                print(f"ERROR: PKL file at '{self.pkl_path}' is empty or corrupted.")
                return None, []

            # First object is metadata, rest are data entries
            metadata_entry = all_objects[0]
            data_entries = all_objects[1:]

            return metadata_entry, data_entries

        except FileNotFoundError:
            print(f"ERROR: PKL file not found at '{self.pkl_path}'. Please double-check the path.")
            return None, []
        except Exception as e:
            print(f"ERROR: Failed to load PKL file at '{self.pkl_path}': {e}")
            return None, []

    def get_metadata(self):
        """Returns the metadata of the .pkl file."""
        if self.metadata_entry:
            return self.metadata_entry.get("metadata", {})
        return {}

    def display_metadata(self):
        """Displays the metadata of the .pkl file."""
        print("\n--- Metadata ---")
        metadata = self.get_metadata()
        for key, value in metadata.items():
            if key == "realsense_intrinsics":
                print(f"  {key}:")
                for stream, intrinsics in value.items():
                    print(f"    {stream}:")
                    for k, v in intrinsics.items():
                        print(f"      {k}: {v}")
            else:
                print(f"  {key}: {value}")

    def get_capture_data(self):
        """Returns the capture data of the .pkl file."""
        num_captures = len(self.data_entries)
        if num_captures == 0:
            print("No actual capture data entries found.")
            return None

        while True:
            try:
                idx_str = input(f"Enter capture index (0 to {num_captures - 1}): ").strip()
                self.capture_idx = int(idx_str)
                if 0 <= self.capture_idx < num_captures:
                    break
                else:
                    print(f"Invalid index. Please enter a number between 0 and {num_captures - 1}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        selected_capture = self.data_entries[self.capture_idx]
        return selected_capture

    def display_capture_data(self, selected_capture):
        """Displays selected capture data of the .pkl file."""
        print(f"\n--- Displaying Capture Index {self.capture_idx} ---")
        print(f"  Iteration: {selected_capture.get('iter')}")
        print(f"  Position: {selected_capture.get('pos')}")


    def plot_capture_data(self, selected_capture):
        """Plot Aligned RGB, Aligned Depth, and Histogram in a single row"""
        print(f"\n--- Plotting Capture Index {self.capture_idx} ---")
        realsense_data = selected_capture.get("realsense_data", {})
        aligned_rgb = realsense_data.get("aligned_rgb_image")
        print(f'type of aligned_rgb: {type(aligned_rgb)}')
        aligned_depth = realsense_data.get("aligned_depth_image")
        histogram = selected_capture.get("histogram")

        plt.figure(figsize=(20, 6)) # Adjusted figure size for 3 plots in a row

        # Plot Aligned RGB
        if aligned_rgb is not None:
            plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
            plt.imshow(aligned_rgb[:, :, ::-1]) # Convert BGR to RGB for matplotlib
            plt.title(f"Aligned RGB (Idx {self.capture_idx})")
            plt.axis('off')
        else:
            print("Warning: Aligned RGB image not found.")
            plt.subplot(1, 3, 1)
            plt.text(0.5, 0.5, "Aligned RGB Not Found", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')

        # Plot Aligned Depth
        if aligned_depth is not None:
            plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
            # Scale depth for visualization (adjust vmax for your depth sensor's range)
            plt.imshow(aligned_depth, cmap='jet', vmin=0, vmax=np.max(aligned_depth) * 0.5)
            plt.title(f"Aligned Depth (Idx {self.capture_idx})")
            plt.colorbar(label='Depth (units)')
            plt.axis('off')
        else:
            print("Warning: Aligned Depth image not found.")
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, "Aligned Depth Not Found", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')

        # Plot Histogram
        if histogram is not None:
            plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
            if histogram.ndim > 1: # If multiple channels, average them for a single plot
                avg_hist = histogram.mean(axis=0)
                plt.bar(range(len(avg_hist)), avg_hist)
                plt.title(f"Averaged Histogram (Idx {self.capture_idx})")
            else:
                plt.bar(range(len(histogram)), histogram)
                plt.title(f"Histogram (Idx {self.capture_idx})")
            plt.xlabel("Bin")
            plt.ylabel("Count")
            plt.grid(axis='y', alpha=0.75)
        else:
            print("Warning: Histogram data not found.")
            plt.subplot(1, 3, 3)
            plt.text(0.5, 0.5, "Histogram Not Found", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
