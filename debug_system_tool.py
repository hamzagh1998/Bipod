import asyncio
import os
import json
import platform
import multiprocessing
from datetime import datetime
import subprocess

async def get_info_logic():
    print("--- STARTING SYSTEM DATA COLLECTION ---")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cores = multiprocessing.cpu_count()
    
    gpu_info = "No GPU detected or nvidia-smi failed."
    try:
        gp = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if gp.returncode == 0: 
            gpu_info = gp.stdout.strip()
            print(f"GPU Data found: {gpu_info}")
    except Exception as e:
        print(f"GPU Error: {e}")
    
    cpu_usage = "Unknown"
    try:
        # Simplified grep to avoid top weirdness in scripts
        cp = subprocess.run("top -bn1 | head -n 10", shell=True, capture_output=True, text=True, timeout=2)
        if cp.returncode == 0: 
            cpu_usage = cp.stdout.strip()
            print(f"CPU/Top Data captured (first 10 lines)")
    except Exception as e:
        print(f"CPU Error: {e}")

    result = f"Time: {now}\nOS: {os_info}\nCPU: {cores} cores, {cpu_usage}\nGPU: {gpu_info}"
    print("\n--- FINAL RESULT FOR MODEL ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(get_info_logic())
