#!/usr/bin/env python3
"""
Interactive debugging shell for the pipeline
Usage: python debug_shell.py
"""

import subprocess
import argparse

CONTAINER_NAME = "pipelines"

def container_from_instance(instance_name: str) -> str:
    return f"{instance_name.strip().lower()}-pipelines"

def main():
    parser = argparse.ArgumentParser(description="Open a shell in a Pipelines container.")
    parser.add_argument("--container-name", default=CONTAINER_NAME, help="Pipelines container name.")
    parser.add_argument("--instance-name", help="Named generic deployment. Sets container name to '<instance-name>-pipelines'.")
    args = parser.parse_args()
    container_name = container_from_instance(args.instance_name) if args.instance_name else args.container_name

    print("🐚 Opening interactive shell in pipelines container...")
    print(f"   Container: {container_name}")
    print("\n💡 Useful commands:")
    print("   - ls /app/pipelines/          # List pipeline files")
    print("   - cat ISGAccessInterface.py   # View pipeline code")
    print("   - python                      # Start Python REPL")
    print("   - exit                        # Exit shell")
    print("\n" + "=" * 80 + "\n")
    
    try:
        subprocess.run(["docker", "exec", "-it", container_name, "/bin/bash"])
    except KeyboardInterrupt:
        print("\n\n👋 Exiting shell")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the pipelines container is running:")
        print("   docker ps")

if __name__ == "__main__":
    main()
