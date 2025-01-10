# ----------------------
# Data preperation
# ----------------------

from Helpers.preprocessingHelpers import encodeCompounds, decode_compounds
from sklearn.model_selection import train_test_split
import numpy as np

compounds = []
ppms = []
y = []

with open("./data/peakGraph/combos.txt", 'r') as f:
    content = f.read()
    for line in content.split('\n'):
        if line == "":
            continue
        split_line = line.split("\t")
        compounds.append(split_line[0])
        ppms.append(split_line[1])
        y.append(split_line[2])

df, compounds = encodeCompounds(compounds)


x = []
for i in range(len(compounds)):
    x.append([*compounds[i], float(ppms[i])])

x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.int8)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=.3,
                                                    random_state=42,
                                                    shuffle=True)

# ----------------------
# SMOTE
# ----------------------

from imblearn.over_sampling import SMOTE
import numpy as np

unique, count = np.unique(y_train, return_counts=True)
y_val_count = { k:v for (k, v) in zip(unique, count) }

smote = SMOTE(random_state=42, sampling_strategy='minority')
decoded = decode_compounds(x_train)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

unique, count = np.unique(y_train_res, return_counts=True)
y_val_count = { k:v for (k, v) in zip(unique, count) }
print(y_val_count)

decoded = decode_compounds(x_train_res[:, :12])
print(decoded[0], x_train_res[0][12])