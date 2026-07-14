#!/usr/bin/env python3
"""
Automatically upload/update ISGAccessInterface to Open WebUI
Usage: python scripts/upload/upload_pipeline.py
   Or: cd scripts/upload && python upload_pipeline.py
"""

import argparse
import sys
import os
import time
import subprocess

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONTAINER_PATH = "/app/pipelines/"

def container_from_instance(instance_name: str) -> str:
    return f"{instance_name.strip().lower()}-pipelines"

def dependency_paths_for(pipeline_path: str):
    """Return sibling runtime dependencies for pipelines that import local modules."""
    dependencies = []
    pipeline_dir = os.path.dirname(os.path.abspath(pipeline_path))
    pipeline_file = os.path.basename(pipeline_path)
    if pipeline_file == "ISGAccessInterface.py":
        dependencies.append((os.path.join(pipeline_dir, "ingest.py"), "ISGAccessInterface/ingest.py"))
    return dependencies

def copy_file_to_container(source_path: str, container_name: str, target_path: str = None):
    source_path = os.path.abspath(source_path)
    target_path = target_path or os.path.basename(source_path)
    target_dir = os.path.dirname(target_path)
    if target_dir:
        mkdir_result = subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", f"{CONTAINER_PATH}{target_dir}"]
        )
        if mkdir_result.returncode != 0:
            return False
    result = subprocess.run(
        ["docker", "cp", source_path, f"{container_name}:{CONTAINER_PATH}{target_path}"]
    )
    return result.returncode == 0

def upload_via_docker(pipeline_path: str, container_name: str):
    """Copy pipeline directly to the container and restart"""
    pipeline_path = os.path.abspath(pipeline_path)
    pipeline_file = os.path.basename(pipeline_path)
    print(f"🚀 Uploading {pipeline_file} to Open WebUI Pipelines...")
    print(f"   Source: {pipeline_path}")
    
    # Check if file exists
    if not os.path.exists(pipeline_path):
        print(f"❌ Error: {pipeline_path} not found!")
        print(f"   Make sure you're running this from the project root or scripts/upload directory")
        return False
    
    # Check if container is running
    check_cmd = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if container_name not in check_cmd.stdout.splitlines():
        print(f"❌ Error: Container '{container_name}' is not running!")
        print("   Start it with scripts/deploy.py or docker compose.")
        return False
    
    files_to_upload = [(pipeline_path, pipeline_file)]
    for dependency_path, target_path in dependency_paths_for(pipeline_path):
        if os.path.exists(dependency_path):
            files_to_upload.append((dependency_path, target_path))
        else:
            print(f"⚠️  Warning: dependency not found and will not be uploaded: {dependency_path}")

    # Copy files to container
    print(f"📤 Copying to container '{container_name}'...")
    for source_path, target_path in files_to_upload:
        if not copy_file_to_container(source_path, container_name, target_path):
            print(f"❌ Failed to copy file to container: {source_path}")
            return False
    
    print("✅ File copied successfully!")
    
    # Restart the container to reload pipelines
    print("🔄 Restarting pipelines container...")
    result = subprocess.run(["docker", "restart", container_name])
    
    if result.returncode != 0:
        print("❌ Failed to restart container!")
        return False
    
    # Wait for container to be ready
    print("⏳ Waiting for pipeline to initialize (15 seconds)...")
    time.sleep(15)
    
    # Check container health
    print("🔍 Checking pipeline status...")
    logs_cmd = subprocess.run(
        ["docker", "logs", "--tail", "20", container_name],
        capture_output=True, text=True
    )
    
    # Check for errors in recent logs
    logs_output = logs_cmd.stdout + logs_cmd.stderr
    if "error" in logs_output.lower() and "Error" not in "CrossEncoder":
        print("⚠️  Warning: Possible errors in pipeline logs:")
        print("-" * 50)
        # Print last few lines
        for line in logs_output.split('\n')[-10:]:
            if line.strip():
                print(f"   {line}")
        print("-" * 50)
    
    # Check if pipeline loaded successfully
    if "Hybrid RAG Pipeline created" in logs_output or "INIT" in logs_output:
        print("✅ Pipeline loaded successfully!")
    
    print("✅ Pipeline uploaded and container restarted!")
    print("\n💡 The pipeline should now be available in Open WebUI")
    print("   Go to Admin Panel → Settings → Connections to verify")
    print(f"\n📋 To check logs: docker logs {container_name} --tail 50")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a pipeline file into a running Pipelines container.")
    parser.add_argument(
        "--pipeline-file",
        default=os.path.join(PROJECT_ROOT, "pipelines", "ISGAccessInterface.py"),
        help="Pipeline Python file to upload.",
    )
    parser.add_argument(
        "--container-name",
        default="pipelines",
        help="Pipelines container name. Defaults to the legacy Ottobot container.",
    )
    parser.add_argument(
        "--instance-name",
        help="Named generic deployment. Sets container name to '<instance-name>-pipelines'.",
    )
    args = parser.parse_args()

    container_name = container_from_instance(args.instance_name) if args.instance_name else args.container_name
    success = upload_via_docker(args.pipeline_file, container_name)
    sys.exit(0 if success else 1)
