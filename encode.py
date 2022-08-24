# https://github.com/tea-vg/google_brain_ventilator_pressure_prediction/blob/master/competitions/google-brain-2021/v38/encode.py

import base64
import gzip
import sys
from pathlib import Path
from typing import List

import git

template = """
import gzip
import base64
import os
from pathlib import Path
from typing import Dict
# this is base64 encoded source code
file_data: Dict = {file_data}
for path, encoded in file_data.items():
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))
# output current commit hash
print('{commit_hash}')
"""

jupyter_template = """
"import gzip\\n",
"import base64\\n",
"import os\\n",
"from pathlib import Path\\n",
"from typing import Dict\\n",
"# this is base64 encoded source code\\n",
"file_data: Dict = {file_data}\\n",
"for path, encoded in file_data.items():\\n",
"    path = Path(path)\\n",
"    path.parent.mkdir(exist_ok=True)\\n",
"    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))\\n",
"# output current commit hash\\n",
"print('{commit_hash}')\\n",
"""

# Uncomment this line if you replace from jupyter notebook.
template = jupyter_template


def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode("utf-8")


def build_script(modules: List[str]):
    global template

    all_data = {}
    for module in modules:
        to_encode = list(Path(module).glob("**/*.py")) + list(Path(module).glob("**/*.yaml"))
        file_data = {str(path).replace("\\", "/").replace("working/", ""): encode_file(path) for path in to_encode}
        all_data.update(file_data)

    template = template.replace("{file_data}", str(all_data))
    template = template.replace("{commit_hash}", get_current_commit_hash())

    with open("./notebook/base-inference.ipynb") as f:
        file_data = f.read()

    file_data = file_data.replace('"##### INSERT SOURCE CODE HERE FOR SUBMISSION #####\\n",', template)
    file_data = file_data.replace("develop-rapids", "python3")  # kernel name

    with open("./notebook/inference.ipynb", "w") as f:
        f.write(file_data)


if __name__ == "__main__":
    args = sys.argv
    roots = args[1:]

    build_script(roots)
