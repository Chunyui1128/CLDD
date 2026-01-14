import os
import pickle

root = "../../datasets/smoke_test"

def dump(folder, out):
    paths = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(('.jpg', '.png', '.jpeg')):
            paths.append(os.path.join(folder, fn))
    with open(out, "wb") as f:
        pickle.dump(paths, f)

dump(os.path.join(root, "real"), "real_train.pickle")
dump(os.path.join(root, "fake"), "fake_train.pickle")

dump(os.path.join(root, "real"), "real_val.pickle")
dump(os.path.join(root, "fake"), "fake_val.pickle")

print("Done.")

