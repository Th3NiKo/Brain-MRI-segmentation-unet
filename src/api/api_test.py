"""
Simple api prediction test.
"""
import os
import requests

SCRIPT_PATH = os.path.dirname(__file__)
TEST_FILE_PATH = os.path.join(
    SCRIPT_PATH, "..", "..", "data", "kaggle_3m",
    "TCGA_CS_5393_19990606", "TCGA_CS_5393_19990606_9.tif"
)

files = {
    'file': open(TEST_FILE_PATH, 'rb')
}

g = requests.post("http://127.0.0.1:8000/predict/image", files=files)

if g.status_code == 200:
    with open("mask.png", 'wb') as f:
        f.write(g.content)
