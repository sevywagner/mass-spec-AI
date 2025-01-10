from Helpers.preprocessingHelpers import encodeCompounds, unsimplifyCompound, decode_compounds, getAllPossibleCompounds
from Helpers.chemHelpers import calculateMass, checkCriteria
import numpy as np

c = ["C4H7N2O"]
print(c)
ec = np.array(encodeCompounds(c)[1][0])
print(ec)
uec = unsimplifyCompound(ec)
print(uec)
dec = decode_compounds([uec])[0]
print(dec)