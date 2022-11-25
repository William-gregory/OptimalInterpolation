import gdown
import zipfile
import os

from OptimalInterpolation import get_data_path

print("Downloading the zipped version")

data_dir = get_data_path()

# https://drive.google.com/file/d/1djlaZ2EKbm9pNAEt3w58WJtBA4NyQsNE/view?usp=sharing
id_zip = [
    {"id": "1djlaZ2EKbm9pNAEt3w58WJtBA4NyQsNE", "zip": "new_aux.zip", "dirname": "aux"},
    {"id": "1cIh9lskzmL6C7EYV8lmJJ5YaJgKqOZHT", "zip": "CS2S3_CPOM.zip", "dirname": "CS2S3_CPOM"},
    {"id": "1gXsvtxZcWpBALomgeqn9kcfyCtKD3fkz", "zip": "raw_along_track.zip", "dirname": "RAW"},
]

# TODO: check if output dir already exists: aux and CS2S3_CPOM
for _ in id_zip:
    id = _['id']
    zip = _['zip']
    dirn = _.get('dirname', "")

    if os.path.exists(os.path.join(data_dir, dirn)):
        # print(f"dir{}")
        continue
    # put data in data dir in repository
    output = os.path.join(data_dir, zip)
    gdown.download(id=id, output=output, use_cookies=False)

    # un zip to path
    print("unzipping")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(path=data_dir)

    # remove zip folder
    os.remove(os.path.join(data_dir, zip))
