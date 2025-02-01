""" Open PyLAD Viewer

This can be used to simplify opening pylad data files.

Run it and specify <data_type> <run_name> <data_num>
"""

from pathlib import Path
import subprocess
import sys

# Set this to the base of your run directory
RUN_DIR = Path.home() / 'Runs'

VALID_FILE_TYPES = [
    'background',
    'data',
    'data_ds',
    'post_shot_background',
]

# The files look like this:
# Run_114_evt_3_Varex1_background.tiff

if len(sys.argv) < 4:
    sys.exit('<script> <file_type> <run_name> <data_num>')

file_type = sys.argv[1]
run_name = sys.argv[2]
data_num = int(sys.argv[3])

if file_type not in VALID_FILE_TYPES:
    sys.exit(
        f'Invalid file type "{file_type}". '
        f'Valid types are: {VALID_FILE_TYPES}'
    )

run_dir = RUN_DIR / run_name
ending = f'{file_type}.tiff'

# First, gather all of the data files
data_files = []
for path in run_dir.iterdir():
    if path.name.endswith(ending):
        data_files.append(path)

num_files_per_detector = len(data_files) // 2
if data_num > num_files_per_detector:
    sys.exit(f'Max valid data_num: {num_files_per_detector}')

# Find the smallest event number for this set of data files
event_numbers = [int(path.name.split('_')[3]) for path in data_files]
starting_event_num = min(event_numbers)
selected_event = data_num - 1 + starting_event_num

# Find the two files that match
matching_paths = []
for path in data_files:
    if int(path.name.split('_')[3]) == selected_event:
        matching_paths.append(path)

# Sort them so that 'Varex1' is first
if 'Varex1' not in matching_paths[0].name:
    matching_paths = reversed(matching_paths)

print('Opening pylad-viewer with files:', list(map(str, matching_paths)))
subprocess.run([
    'pylad-viewer',
    str(matching_paths[0]),
    str(matching_paths[1]),
])
