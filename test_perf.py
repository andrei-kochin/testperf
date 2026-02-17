import os
import sys
from time import perf_counter
import platform

# Accept model name with dot delimiter or directory separator (backslash or forward slash)
model_name = sys.argv[1].replace('/', '.').replace('\\', '.') if len(sys.argv) > 1 else 'test_model'
if model_name.endswith('.py'):
  model_name = model_name[:-3]

print(f'{{ "Model": "{model_name}",')
print(f'"Hostname": "{platform.node()}",')
print(f'"Platform": "{platform.system()} {platform.release()} {platform.version()}",')
print('"Steps": [')

script_run_time  = perf_counter()

try:
  test_model = __import__(model_name, fromlist=["Model"])
except Exception as e:
  print(f'{{ "Error": "Failed to load model {e}" }},')
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  path_parts = model_name.split('.')
  model_name = ''
  for part in path_parts:
    last_dir = str(cur_dir)
    cur_dir = os.path.join(cur_dir, part)
    if not os.path.exists(cur_dir):
      print('\n\nAvailable options: ')
      print("\n".join([f'{model_name[1:]}.{x}...' for x in os.listdir(last_dir)]))
      break
    model_name = model_name + '.' + part
  exit(1)

if not os.path.exists('./temp'):
  os.makedirs('./temp')

# Setting batch sizes from command line
batches = [1]
if '--batch-size' in sys.argv:
  try:
    batches = [int(x) for x in sys.argv[sys.argv.index('--batch-size') + 1].split(',')]
  except Exception as e:
    print(f'{{ "Error": "Failed to set batch size {e}, using default [{', '.join(batches)}]" }},')

mul_time = []
def checkpoint(do_reset = True):
  global mul_time
  if do_reset:
    mul_time = []
  mul_time.append(perf_counter())

def spent(category):
  global mul_time
  mul_time.append(perf_counter())
  spent_time = mul_time[-1] - mul_time[0]
  print(f'{{ "{category}" : "{spent_time}" }},')

model = test_model.Model()
if '--runs' in sys.argv:
  try:
    model.total_inference_runs = int(sys.argv[sys.argv.index('--runs') + 1])
  except Exception as e:
    print(f'{{ "Error": "Failed to set runs {e}, using default {model.total_inference_runs} runs" }},')

checkpoint()
for batch in batches:
  print(f'{{ "Preparing Batch Size": {batch} }},')
  model.prepare_batch(batch)
spent("Total Preparing Batches")

if '--only-prepare' in sys.argv:
  checkpoint()
  model.shutdown()
  spent("Model Shutdown")
  print(f'{{ "Status": "Done" }}')
  print('] }')
  exit(0)

checkpoint()
model.read1st()
spent("Model 1st Read")

first_read_time = mul_time[-1] - mul_time[0]

if first_read_time < 60:
  read_runs = min(50, (600 // int(first_read_time if first_read_time > 1 else 1)))
  print(f'{{ "Read Runs": {read_runs} }},')
  checkpoint()
  for i in range(read_runs):
    model.readnth()
    checkpoint(False)
  spent(f"Total Read")
else:
  print(f'{{ "Error": "Read time is too long {mul_time[-1] - mul_time[0]}, skipping read tests" }},')
  mul_time.append(0) # Fake time to avoid index error

read_times = []
for item in range(1, len(mul_time) - 1):
  read_times.append(mul_time[item] - mul_time[item - 1])

last = mul_time[0]
max_time = mul_time[1] - last
min_time = mul_time[1] - last
for item in mul_time[1:]:
  spent_time = item - last
  last = item
  max_time = max(spent_time, max_time)
  min_time = min(spent_time, min_time)

avg_time = (mul_time[-1] - mul_time[0]) / (len(mul_time) - 1)

print('{ "Read Times": [')
for item in read_times[:-1]:
  print(f'{{ "Time" : "{item}" }},')
print(']},')

print(f'"Read Minimum" : "{min_time}",')
print(f'"Read Maximum" : "{max_time}",')
print(f'"Read Average" : "{avg_time}"')
print('},')

read_times.append({"Minimum": min_time, "Maximum": max_time, "Average": avg_time})

model.batch_size = None
inference_times = {}
warm_up_times = {}

for batch in batches:
  print(f"{{ \"Running Batch\": {batch} }},")
  if model.batch_size is None or model.batch_size != batch:
    model.shutdown()
    model.batch_size = batch

  checkpoint()
  model.read()
  model.warm_up()
  spent(f"Model Warm Up {batch}")
  warm_up_times[batch] = mul_time[-1] - mul_time[0]

  model.reset_inference_run()
  model.prepare()

  # Few empty runs
  cnt = 0
  while cnt < 10:
    model.inference()
    cnt += 1

  checkpoint()
  while model.next_inference_run():
    model.inference()
    checkpoint(False)
  spent(f"Total Inference {batch}")

  inference_times[batch] = []
  for item in range(1, len(mul_time) - 1):
    inference_times[batch].append(mul_time[item] - mul_time[item - 1])
  last = mul_time[0]
  max_time = mul_time[1] - last
  min_time = mul_time[1] - last
  for item in mul_time[1:]:
    spent_time = item - last
    last = item
    max_time = max(spent_time, max_time)
    min_time = min(spent_time, min_time)
  avg_time = (mul_time[-1] - mul_time[0]) / (len(mul_time) - 1)

  print('{ "Inference Times": [')
  for item in inference_times[batch][:-1]:
    print(f'{{ "Time" : "{item}" }},')
  print('] },')

  inference_times[batch].append({"Minimum": min_time, "Maximum": max_time, "Average": avg_time})

  print('{')
  print(f'"Batch Size" : "{batch}",')
  print(f'"Minimum" : "{min_time}",')
  print(f'"Maximum" : "{max_time}",')
  print(f'"Average" : "{avg_time}"')
  print('},')

checkpoint()
model.shutdown()
spent("Model Shutdown")

total_time = perf_counter() - script_run_time
print(f"{{ \"Total Time\": {total_time} }},")

try:
  import reports
  reports.performance_report(model, model_name, read_times, inference_times, warm_up_times, batches)
except Exception as e:
  print(f'{{ "Error": "Failed to generate XLS report {e}" }},')

print(f'{{ "Status": "Done" }}')
print('] }')
