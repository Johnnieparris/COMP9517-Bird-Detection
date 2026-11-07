To access the datasets and images I (member A) have created the following architecture:

Step 1: 
Clone this project into VSCode and create a new branch with your member number. (Main is sacred).

Step 2:
Install the relavent extensions, these include:
- Jupyter
- Python
- Rainbow CSV

Step 3:
Define your python Environment:

Run the following in the powershell terminal of vscode:
- "Remove-Item -Recurse -Force .venv"
- "conda create -n birds python=3.12 -y"
- "{your anaconda3 folder address}\envs\birds\python.exe -m pip install -r requirements.txt"

Note: the anaconda3 folder is usually installed in users/{your username}

Step 4:
Switch to the newly defined birds conda environment

1. Ctrl + Shift + P on anywhere in vscode, it will open a window
2. In this window select Python: Select Interpreter
3. Click the new environment you just created (should be named "birds" and be py 3.12)

Step 5:
Download the data to the project by running "python scripts/download_dataset.py" in the terminal

This will generate a folder called data in the root directory.

Step 6:
Confirm data has loaded correctly by running "scripts/check_dataset.py" in the terminal. It will print basic information about the structure and labels to identify the database has loaded correctly.

If loaded in correctly, check your extensions and dependencies are installed correctly and you are in the correct virtual environment, delete the data folder and resume from step 5.

Step 7: 
To use the data is your code
"""
import sys
sys.path.insert(0, "src")   # use "../src" if your file is in a subfolder

from data.dataset import load_split, get_image, DEFAULT_ROOT

records = load_split(DEFAULT_ROOT, "val")   # "train", "val" or "test"
img = get_image(records[0])                 # numpy array, RGB
boxes = records[0].boxes                    # numpy array, one row per bird
"""
Each box is x1, y1, x2, y2 in pixels:

x1, y1 = top-left corner
x2, y2 = bottom-right corner

Images with no birds have zero boxes.

Everyone must use the same train/val/test images, so results can be compared. Load them from the saved file splits/official.json (do not make your own split):
"""
from data.dataset import DEFAULT_ROOT
from data.split import load_split_manifest, load_split_from_manifest

manifest = load_split_manifest("splits/official.json")
train = load_split_from_manifest(DEFAULT_ROOT, manifest, "train")
val = load_split_from_manifest(DEFAULT_ROOT, manifest, "val")
test = load_split_from_manifest(DEFAULT_ROOT, manifest, "test")
"""
Only use the test split for your final results, never for tuning.

Step 8:
Put your code in your own folder (e.g. src/m1/). Always load data through src/data/dataset.py. Do not edit src/data/ or splits/.

Data layout
data/Beach Bird Detection Dataset v1.0/
├── images/ (train, val, test)
└── labels/ (train, val, test)
Images: jpg, jpeg, png
Labels: YOLO .txt, one bird per line: class cx cy w h (values 0–1)
Example: 0 0.512 0.483 0.072 0.095
Only one class: 0 = bird

Step 10: 
Save your predictions for analysis

Every method saves its predictions in the same format, so they can be scored and compared fairly. Use the sys.path lines from Step 7 first.

"""
from evaluation import run_and_time, evaluate, print_results
from predictions import save_predictions

records = load_split_from_manifest(DEFAULT_ROOT, manifest, "val")

preds, timing = run_and_time(records, my_method)   # my_method(image) -> (N, 5) array
print_results(evaluate(records, preds, timing), "Method 1")

save_predictions("m1", "val", records, preds)      # creates predictions/m1_val.csv
"""

Rules:

my_method returns one row per detected bird: x1, y1, x2, y2, score (pixels, same corners as record.boxes). Return an empty array if it finds no birds. Methods without a score can use 1.0.
Keep low-scoring detections too (for YOLO use conf=0.001). Do not filter them out.
Method name: letters, numbers, - and _ only (e.g. m1, yolo_n).
Saving again with the same method name and split replaces your old file. Other people's files are never touched.
Only the best 100 detections per image are saved. (Let me know if this is too little but the highest value in the whole dataset is high 60's)
Use val while tuning. Save test once, for your final results.
Always use the same splits/official.json and the shared evaluation.py. Never write your own scoring.

Look at everyone's results together:

"""
from predictions import load_all_predictions, export_combined_csv

rows = load_all_predictions()      # list of dicts, one per detection, all methods
export_combined_csv()              # writes predictions/combined.csv
"""

The predictions/ folder is generated on your own computer and is not stored in git. Add predictions/ to .gitignore. To compare methods, run each method yourself, or send predictions/combined.csv through Drive or Teams.

Data layout
data/Beach Bird Detection Dataset v1.0/
├── images/ (train, val, test)
└── labels/ (train, val, test)
Images: jpg, jpeg, png
Labels: YOLO .txt, one bird per line: class cx cy w h (values 0–1)
Example: 0 0.512 0.483 0.072 0.095
Only one class: 0 = bird

The analysis is only performed on the test data. evalutation.py only works on this data.

Step 11:
Save overlay pictures of one method's predictions against the real boxes.
 
    GREEN = TP  correct detection (found a real bird)
    RED   = FP  false alarm (detected a bird that is not there)
    BLUE  = FN  missed bird (a real bird nobody found)
 
Uses the same matching and settings as src/evaluation.py, so the pictures agree
with the scores in your report. Use for report figures and the video demo.
 
    python scripts/make_overlays.py m1                       # 12 random test images
    python scripts/make_overlays.py m1 --pick worst --n 8    # images with the most mistakes
    python scripts/make_overlays.py m1 --pick best --n 8     # cleanest images
    python scripts/make_overlays.py m1 --ids test_0003,test_0010
    python scripts/make_overlays.py m1 --pick all            # every image (for a video)
 


Additional Information:

If you want to test an image not in the dataset, here is the process of importing it:
"""
from data.importers import import_roboflow
import_roboflow("{your roboflow export location}/roboflow_export", "data/extra_birds")
"""