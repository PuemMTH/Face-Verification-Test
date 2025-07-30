#!/usr/bin/env python3
import subprocess
import sys
import os

def run_silent():
    """Run the main script with all warnings and errors suppressed"""
    
    # Set environment variables to suppress all output
    env = os.environ.copy()
    env.update({
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'GLOG_minloglevel': '3',
        'CUDA_VISIBLE_DEVICES': '-1',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'ABSL_LOGGING_MIN_LEVEL': '3',
        'TF_ENABLE_DEPRECATION_WARNINGS': '0',
        'TF_LOGGING_LEVEL': '3',
        'GLOG_logtostderr': '0',
        'PYTHONWARNINGS': 'ignore',
        'TF_ENABLE_GPU_GARBAGE_COLLECTION': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false'
    })
    
    # Run the main script with output redirected
    try:
        result = subprocess.run([
            sys.executable, 'run.py'
        ], 
        env=env,
        capture_output=True,
        text=True,
        check=True
        )
        
        # Print only the essential progress messages
        lines = result.stdout.split('\n')
        for line in lines:
            if any(keyword in line for keyword in [
                'Starting face verification test',
                'Processing folder:',
                'Processing:',
                'Results saved to',
                'Completed processing',
                'All folders processed successfully'
            ]):
                print(line)
                
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)

if __name__ == "__main__":
    run_silent() 