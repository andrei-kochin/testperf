#!/usr/bin/env python3

import os
import sys
import subprocess
from time import perf_counter

script_run_time  = perf_counter()

# Docker configurations
docker_configs = [
#   Example configuration
#    {
#        'docker_image': 'test_image_1',
#        'docker_file': './Dockerfile1',
#        'dont_remove': False,
#        'docker_custom_run': '',
#        'docker_hostname': 'epyc_gpu', # optional, useful for understanding test environment
#        'tests': ['models.yolo8n.ort', 'models.yolo8n.ort_dml']
#    },
#    {
#        'docker_image': 'test_image_2',
#        'docker_file': '',
#        'dont_remove': True,
#        'docker_custom_run': 'docker run -it --gpus all',
#        'docker_hostname': 'epyc_all_gpu', # optional, useful for understanding test environment
#        'tests': ['models.yolo11l.ort', 'models.yolo11l.ort_ov']
#    },
]

def help_message():
    help_text = """
Docker Runner - Performance Testing Tool

Usage: python docker_runner.py [OPTIONS]

Configuration:
  --show-config              Display current docker configurations and exit

Test Selection:
  --case <test_name>         Run a specific test case (e.g., 'models.yolo8n.ort')
  --batch-size <sizes>       Set batch sizes as comma-separated list (default: 1)
                             Example: --batch-size 1,4,8

Container Selection:
  --continue <index>         Start execution from a specific configuration index
  --count <value>            Limit the number of configurations to run
  --single <index>           Run only the configuration at the specified index
                             Supports negative indexing (e.g., -1 for last config)

Container Management:
  --dont-remove              Keep Docker containers after execution (don't use --rm)
  --shell                    Open an interactive shell in the container
                             (must be used with --single)

Execution Control:
  --fake                     Dry-run mode: show commands without executing them

Examples:
  # Show current configuration
  python docker_runner.py --show-config

  # Run all configurations
  python docker_runner.py

  # Run specific configuration with custom batch sizes
  python docker_runner.py --single 0 --batch-size 1,4,8,16

  # Run a specific test case on configuration 2
  python docker_runner.py --single 2 --case models.yolo11l.ort

  # Continue from configuration 3 onwards
  python docker_runner.py --continue 3

  # Open shell in the first configuration
  python docker_runner.py --single 0 --shell

  # Dry-run to see what would be executed
  python docker_runner.py --fake

Configuration File:
  The script looks for 'docker_runner.json' in the current directory.
  Configuration should be a JSON object with the following fields:
    - docker_image: (Optional) Name of the Docker image to use
    - docker_file: Path to Dockerfile (used to build image if it doesn't exist)
    - dont_remove: (Optional) Keep container after execution
    - docker_custom_run: (Optional) Custom docker run command
    - docker_hostname: (Optional) Hostname to set in container
    - tests: List of test cases to run

  Example 'docker_runner.json':
  [
    {
      "docker_file": "./dockers/com.org_name.docker_image_name",
      "dont_remove": false,
      "docker_custom_run": "",
      "docker_hostname": "test_machine_A",
      "tests": ["models.yolo8n.ort", "models.yolo8n.ort_dml"]
    },
    {
      "docker_image": "my_image_2",
      "docker_file": "",
      "dont_remove": true,
      "docker_custom_run": "docker run -it --gpus all",
      "docker_hostname": "test_machine_B",
      "tests": ["models.yolo11l.ort", "models.yolo11l.ort_ov"]
    }
  ]
"""
    print(help_text)

if '--help' in sys.argv or '-h' in sys.argv or '-help' in sys.argv:
    help_message()
    exit(0)

if os.path.exists('./docker_runner.json'):
    import json
    with open('./docker_runner.json', 'r') as f:
        docker_configs = json.load(f)
else:
    print("No docker_runner.json file found, using default configurations")

print("Current docker_configs:")
for idx, config in enumerate(docker_configs):
    print(f"[{idx}] {config}")

if '--show-config' in sys.argv:
    exit(0)
if len(docker_configs) == 0:
    print("No docker configurations found, exiting")
    exit(1)

# Create batches array as in test_perf.py
batches = [1]
if '--batch-size' in sys.argv:
    try:
        batches = [int(x) for x in sys.argv[sys.argv.index('--batch-size') + 1].split(',')]
    except Exception as e:
        print(f'Error: Failed to set batch size {e}, using default [1]')

# Get --continue argument and following index or 0 if no argument
start_index = 0
if '--continue' in sys.argv:
    try:
        continue_idx = sys.argv.index('--continue')
        if continue_idx + 1 < len(sys.argv):
            start_index = int(sys.argv[continue_idx + 1])
    except (ValueError, IndexError) as e:
        print(f'Error: Invalid --continue argument {e}, using index 0')
        start_index = 0

end_index = len(docker_configs)
if '--last' in sys.argv:
    try:
        count_idx = sys.argv.index('--count')
        if count_idx + 1 < len(sys.argv):
            value = int(sys.argv[count_idx + 1]) + 1
            if value > 0:
                end_index = start_index + value
            elif (end_index + -value) > 0:
                end_index = end_index - value
    except (ValueError, IndexError) as e:
        print(f'Error: Invalid --last argument {e}, using {end_index}')

if '--single' in sys.argv:
    try:
        single_idx = sys.argv.index('--single')
        if single_idx + 1 < len(sys.argv):
            value = int(sys.argv[single_idx + 1])
            if value > 0:
                start_index = value
                end_index = value + 1
            else:
                start_index = end_index - value
                end_index = end_index - value + 1
    except (ValueError, IndexError) as e:
        print(f'Error: Invalid --single argument {e}, using range {start_index}..{end_index}')

# First pass: Validate docker_image or docker_file exists
print(f"Validating configurations in range {start_index}..{end_index}...")
for i in range(start_index, end_index):
    config = docker_configs[i]
    if not config.get('docker_image') and not config.get('docker_file'):
        raise Exception(f"Configuration at index {i} must have either 'docker_image' or 'docker_file' set")
    print(f"  [{i}] Valid: image='{config.get('docker_image', '')}' file='{config.get('docker_file', '')}'")

print("\nStarting docker operations...\n")

# Get current script's folder
script_folder = os.path.dirname(os.path.abspath(__file__))

# Second pass: Build and run containers
for i in range(start_index, end_index):
    config = docker_configs[i]
    docker_image = config.get('docker_image', 'testperf:latest')
    docker_file = config.get('docker_file', None)
    dont_remove = config.get('dont_remove', False) or ('--dont-remove' in sys.argv)
    docker_custom_run = config.get('docker_custom_run', '')
    tests = config.get('tests', [])
    
    print(f"=== Processing configuration {i} ===")
    # Check if docker image exists
    docker_image_exists = False
    if docker_image:
        check_cmd = ['docker', 'image', 'inspect', docker_image]
        print(f"  Command: {' '.join(check_cmd)}")
        result = subprocess.run(check_cmd, cwd=script_folder, capture_output=True)
        docker_image_exists = (result.returncode == 0)
    
    # Build image if docker_image doesn't exist but docker_file does
    if (not docker_image_exists) or (not docker_image and docker_file):
        if docker_file.startswith('docker pull '):
            print(f"Pulling docker image from {docker_file}...")
            build_cmd = docker_file.split()
        else:
            print(f"Building docker image from {docker_file}...")
            build_cmd = ['docker', 'build', '-f', docker_file, '-t', docker_image, '.']
        print(f"  Command: {' '.join(build_cmd)}")
        if not '--fake' in sys.argv:
            result = subprocess.run(build_cmd, cwd=script_folder)
            if result.returncode != 0:
                print(f"Error: Failed to build image from {docker_file}, to continue from this point, use --continue {i}, or --single {i}")
                continue
        if docker_file.startswith('docker pull ') and docker_image != docker_file[len('docker pull '):].strip():
            tag_cmd = ['docker', 'tag', docker_file[len('docker pull '):], docker_image]
            print(f"  Command: {' '.join(tag_cmd)}")
            if not '--fake' in sys.argv:
                result = subprocess.run(tag_cmd, cwd=script_folder)
                if result.returncode != 0:
                    print(f"Error: Failed to build image from {docker_file}, to continue from this point, use --continue {i}")
                    continue
    
    # Build the docker run command
    if docker_custom_run:
        # Use custom run command as base
        docker_cmd = docker_custom_run.split()
    else:
        # Default docker container run command
        docker_cmd = ['docker', 'container', 'run']
    
    # Add volume mount for current script's folder
    docker_cmd.extend(['-v', f'{script_folder}:/root/testperf'])
    
    # Remove container after run
    docker_cmd.append('--rm')
    
    # Set workdir
    docker_cmd.extend(['--workdir=/root/testperf'])

    # set docker hostname if it is provided
    if 'docker_hostname' in config:
        docker_cmd.extend(['--hostname', config['docker_hostname']])
    
    # Add docker image
    docker_cmd.append(docker_image)
    
    if (('--single' in sys.argv) and ('--shell' in sys.argv)):
        docker_cmd.insert(-1, '-it');
        print(f"Running shell: {' '.join(docker_cmd)}")
        if not '--fake' in sys.argv:
            result = subprocess.run(docker_cmd, cwd=script_folder)
        exit(0)

    # Add the command to run inside container
    docker_cmd.extend(['sh', '-c', 'pip3 install openpyxl && python3 /root/testperf/test_perf.py '])

    if '--case' in sys.argv:
        try:
            tests = [sys.argv[sys.argv.index('--case') + 1]]
            print(f"Running single test: {tests}")
        except Exception as e:
            continue
    
    # Run each test
    for test in tests:
        print(f"\n--- Running test: {test} ---")
        
        # Build the full command with test name and batches
        test_cmd = docker_cmd.copy()
        # Append test name and batch-size to the python command
        test_cmd[-1] = test_cmd[-1] + f'{test} --batch-size {",".join(map(str, batches))}'
        
        print(f"Command: {' '.join(test_cmd)}")
        print()
        
        if not '--fake' in sys.argv:
            # Run the docker command
            result = subprocess.run(test_cmd, cwd=script_folder)
            
            if result.returncode != 0:
                print(f"Error: Test {test} failed with exit code {result.returncode}, to continue from this point, use --continue {i} --case {test}, or --single {i} --case {test}")
                continue


print("All operations completed")
total_time = perf_counter() - script_run_time
print(f"Total time: {total_time} seconds")

