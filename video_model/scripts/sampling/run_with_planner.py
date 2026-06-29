#!/usr/bin/env python3
"""
Wrapper script to run RoboCasa experiments with VideoPolicy planner
"""

import os
import sys
import subprocess
import argparse

# Add the videopolicy module path
sys.path.append('/home/hanan/dev/videopolicy/video_model')

def main():
    parser = argparse.ArgumentParser(description='Run RoboCasa with VideoPolicy planner')
    parser.add_argument('--config', default='scripts/sampling/configs/svd_xt.yaml', 
                       help='Path to config file')
    parser.add_argument('--planner_api_key', default=None, 
                       help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    # Set up environment
    env = os.environ.copy()
    if args.planner_api_key:
        env['GEMINI_API_KEY'] = args.planner_api_key
    
    # Run the original experiment script with planner flag
    cmd = [
        'python', 'scripts/sampling/robocasa_planner.py',  # Using the modified version
        '--config', args.config,
        '--use_planner'
    ]
    
    if args.planner_api_key:
        cmd.extend(['--planner_api_key', args.planner_api_key])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())