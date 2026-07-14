#!/usr/bin/env python3
"""
View and follow pipeline logs in real-time
Usage: python debug_logs.py [--follow] [--tail N]
"""

import subprocess
import argparse

CONTAINER_NAME = "pipelines"

def container_from_instance(instance_name: str) -> str:
    return f"{instance_name.strip().lower()}-pipelines"

def view_logs(container_name=CONTAINER_NAME, follow=False, tail=50):
    """View pipeline container logs"""
    cmd = ["docker", "logs"]
    
    if follow:
        cmd.append("-f")
    
    if tail:
        cmd.extend(["--tail", str(tail)])
    
    cmd.append(container_name)
    
    try:
        print(f"📋 Viewing logs for '{container_name}' container...")
        if follow:
            print("   Press Ctrl+C to stop following\n")
        print("=" * 80)
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Stopped following logs")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View pipeline logs")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    parser.add_argument("-t", "--tail", type=int, default=50, help="Number of lines to show (default: 50)")
    parser.add_argument("-a", "--all", action="store_true", help="Show all logs (no tail limit)")
    parser.add_argument("--container-name", default=CONTAINER_NAME, help="Pipelines container name")
    parser.add_argument("--instance-name", help="Named generic deployment. Sets container name to '<instance-name>-pipelines'.")
    
    args = parser.parse_args()
    
    tail = None if args.all else args.tail
    container_name = container_from_instance(args.instance_name) if args.instance_name else args.container_name
    view_logs(container_name=container_name, follow=args.follow, tail=tail)
