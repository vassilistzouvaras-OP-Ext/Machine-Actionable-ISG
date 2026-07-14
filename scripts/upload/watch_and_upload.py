#!/usr/bin/env python3
"""
Watch ISGAccessInterface.py for changes and auto-upload to Open WebUI
Usage: python scripts/upload/watch_and_upload.py
   Or: cd scripts/upload && python watch_and_upload.py

This will watch for file changes and automatically upload the pipeline
whenever you save changes to ISGAccessInterface.py
"""

import argparse
import os
import time
import subprocess

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PIPELINE_FILE = "ISGAccessInterface.py"
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "pipelines", PIPELINE_FILE)
CONTAINER_NAME = "pipelines"
CONTAINER_PATH = "/app/pipelines/"
CHECK_INTERVAL = 2  # seconds

def container_from_instance(instance_name: str) -> str:
    return f"{instance_name.strip().lower()}-pipelines"

def dependency_paths_for(pipeline_path: str):
    dependencies = []
    pipeline_dir = os.path.dirname(os.path.abspath(pipeline_path))
    pipeline_file = os.path.basename(pipeline_path)
    if pipeline_file == "ISGAccessInterface.py":
        dependencies.append((os.path.join(pipeline_dir, "ingest.py"), "ISGAccessInterface/ingest.py"))
    return dependencies

def upload_pipeline(pipeline_path: str, container_name: str):
    """Copy pipeline to container and restart"""
    pipeline_file = os.path.basename(pipeline_path)
    print(f"\n Change detected! Uploading {pipeline_file}...")

    # Copy file to container
    files_to_upload = [(pipeline_path, pipeline_file), *dependency_paths_for(pipeline_path)]
    for source_path, target_path in files_to_upload:
        if not os.path.exists(source_path):
            print(f"Dependency not found, skipping: {source_path}")
            continue
        target_dir = os.path.dirname(target_path)
        if target_dir:
            mkdir_result = subprocess.run(
                ["docker", "exec", container_name, "mkdir", "-p", f"{CONTAINER_PATH}{target_dir}"]
            )
            if mkdir_result.returncode != 0:
                print(f"Failed to create dependency directory: {target_dir}")
                return False
        result = subprocess.run(
            ["docker", "cp", source_path, f"{container_name}:{CONTAINER_PATH}{target_path}"]
        )

        if result.returncode != 0:
            print(f"Failed to copy file: {source_path}")
            return False
    
    # Restart container
    result = subprocess.run(["docker", "restart", container_name])
    
    if result.returncode != 0:
        print("Failed to restart container!")
        return False
    
    print(f"Pipeline uploaded successfully at {time.strftime('%H:%M:%S')}")
    return True

def watch_file(pipeline_path: str, container_name: str):
    """Watch file for changes and auto-upload"""
    pipeline_path = os.path.abspath(pipeline_path)
    if not os.path.exists(pipeline_path):
        print(f"Error: {pipeline_path} not found!")
        print(f"   Looking for: {pipeline_path}")
        return
    
    print(f"Watching {pipeline_path} for changes...")
    print(f"   Container: {container_name}")
    print("   Press Ctrl+C to stop\n")
    
    last_modified = os.path.getmtime(pipeline_path)
    
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            
            current_modified = os.path.getmtime(pipeline_path)
            
            if current_modified != last_modified:
                last_modified = current_modified
                upload_pipeline(pipeline_path, container_name)
                
    except KeyboardInterrupt:
        print("\n\n Stopped watching. Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a pipeline file and upload it after changes.")
    parser.add_argument("--pipeline-file", default=PIPELINE_PATH, help="Pipeline Python file to watch.")
    parser.add_argument("--container-name", default=CONTAINER_NAME, help="Pipelines container name.")
    parser.add_argument("--instance-name", help="Named generic deployment. Sets container name to '<instance-name>-pipelines'.")
    args = parser.parse_args()
    container_name = container_from_instance(args.instance_name) if args.instance_name else args.container_name
    watch_file(args.pipeline_file, container_name)
