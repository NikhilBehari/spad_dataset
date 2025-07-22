#!/usr/bin/env python3
"""
Environment generation script for creating 3D scenes in Blender.

This script uses the BlenderEnvironmentBuilder from the processing package
to create 3D environments from scene data.
"""

import sys
sys.path.insert(0, "pkgs/processing")
from cc_hardware.processing.blender_env import BlenderEnvironmentBuilder

if __name__ == "__main__":
    # Default scene file - change this to your scene JSON file
    scene_file = "test_scene.json"

    # Filter out Blender-specific arguments and script name
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--') and not arg.endswith('.py')]
    if args:
        scene_file = args[0]

    print(f"Building Blender environment from: {scene_file}")

    # Check if file exists and add debug info
    import os
    abs_path = os.path.abspath(scene_file)
    print(f"Absolute path: {abs_path}")
    print(f"File exists: {os.path.exists(abs_path)}")

    if os.path.exists(abs_path):
        print(f"File size: {os.path.getsize(abs_path)} bytes")
        with open(abs_path, 'r') as f:
            content = f.read()
            print(f"First 100 chars: {content[:100]}")

    # Create the environment builder
    builder = BlenderEnvironmentBuilder(abs_path)

    # Build the scene
    # Options:
    # - save_file: Save the .blend file
    # - quit_blender: Close Blender after building
    builder.build(save_file=False, quit_blender=False)

    print("Environment generation complete!")
    print("🔄 Keeping Blender open for debugging...")
