# Performance Benchmarking framework

## Overview

This framework is used to run a different workloads (originally focused on model inference), measure time of each run, and provide results in machine and human readable formats.

## Installation

### Requirements
 - Python
 - [Optional] OpenPyXL - for generating reports in Excel format
 - [Optional] Docker - for running tasks in predefined environment
 - [Optional] Excel - for running aggregation tool

## Running a single test

Using 'test_perf.py' script you can automate your performance testing by using a common tasks for performance calculations (prepares environment, reading tests, warming up before tests, measuring time of each run).

Most important feature is automatic generation of performance analysis with charts.




### Arguments
- --batch-size - list of batch sizes has to be verified

### Examples

Simple run of YOLO11 Large model benchmarking using ONNXRuntime with default settings, batch size is 1.

```bash
python test_perf.py models.yolo11l.ort
```

Simple run of YOLO11 Large model benchmarking using ONNXRuntime with default settings, custom set of batch size: 1, 2, 4, 8, 16.

```bash
python test_perf.py models.yolo11l.ort --batch-size 1,2,4,8,16
```

## Running batch tasks using docker images

The provided `docker_runner.py` script allows you to automate the running of multiple benchmarking tasks across different Docker container configurations. It supports running batched tests, managing container lifecycle, and customizing Docker execution.

### Usage

```bash
python docker_runner.py [OPTIONS]
```

### Key Options

- `--show-config`
  Display current docker configurations and exit.

- `--case <test_name>`
  Run a specific test case (e.g., `models.yolo8n.ort`).

- `--batch-size <sizes>`
  Set batch sizes as a comma-separated list (default: 1).
  Example: `--batch-size 1,4,8`

- `--continue <index>`
  Start execution from a specific configuration index.

- `--count <value>`
  Limit the number of configurations to run.

- `--single <index>`
  Run only the configuration at the specified index.
  Supports negative indexing (e.g., `-1` for last config).

#### Container Management

- `--dont-remove`
  Keep Docker containers after execution (the default is to remove them).

- `--shell`
  Open an interactive shell in the container (must be used with `--single`).

#### Execution Control

- `--fake`
  Dry-run mode: show command lines to be executed without actually running them.

### Examples

Show current configuration:
```bash
python docker_runner.py --show-config
```

Run all configurations:
```bash
python docker_runner.py
```

Run specific configuration with custom batch sizes:
```bash
python docker_runner.py --single 0 --batch-size 1,4,8,16
```

Run a specific test case on configuration 2:
```bash
python docker_runner.py --single 2 --case models.yolo11l.ort
```

Continue from configuration 3 onward:
```bash
python docker_runner.py --continue 3
```

Open a shell in the first configuration:
```bash
python docker_runner.py --single 0 --shell
```

Perform a dry-run to see the commands that would be executed:
```bash
python docker_runner.py --fake
```

### Configuration File

The script looks for a file named `docker_runner.json` in the current directory.
This file should be a JSON list of configuration objects, each containing:

- `docker_image`: (Optional) Name of the Docker image to use
- `docker_file`: Path to Dockerfile (used to build the image if it doesn't exist)
- `dont_remove`: (Optional) Whether to keep the container after execution
- `docker_custom_run`: (Optional) Custom docker run command
- `docker_hostname`: (Optional) Hostname to set in the container
- `tests`: List of test cases to run

#### Example `docker_runner.json`

```json
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
```

For more details, use `python docker_runner.py --help`.