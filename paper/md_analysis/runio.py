"""Load an MD run written by h2_md.py: <name>.json in the working directory, or the archived
compressed copy ../data/md/<name>.npz shipped with the repository."""
import json, os
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "md")


def load_run(name):
    if os.path.exists(name + ".json"):
        return json.load(open(name + ".json"))
    z = np.load(os.path.join(_DATA, name + ".npz"), allow_pickle=False)
    d = {k: z[k].tolist() for k in z.files if k != "meta"}
    d.update(json.loads(str(z["meta"])))
    return d
