from Helpers.preprocessingHelpers import getAllPossibleCompounds, read_txt, decode_compounds, encodeCompounds, unsimplifyCompound
from Helpers.chemHelpers import calculateMass, compareCompounds, checkCriteria
from Helpers.metricsHelpers import calculatePPM
from Helpers.preprocessingHelpers import getData
import numpy as np
import random
import math

# -----------------------------
# Dataset creation functions
# -----------------------------

def trainTestSplit(file):
    '''
    '''

    with open(file, mode='r') as f:
        lines = f.readlines()
        top = lines[0]
        lines = np.array(lines)[1:]
        mask = []
        for _ in lines:
            if random.randint(0, 100) < 10:
                mask.append(True)
            else:
                mask.append(False)

        mask = np.array(mask)
        test = lines[mask]
        mask = ~mask
        train = lines[mask]

        f.close()

    with open('./data/peakGraph/labeled/test.txt', mode='w+') as fw:
        fw.write(top)
        for i in test:
            fw.write(i)

        fw.close()

    with open('./data/peakGraph/labeled/train.txt', mode='w+') as fw:
        fw.write(top)
        for i in train:
            fw.write(i)

        fw.close()
        

def checkForNotFounds():
    '''
        Function:
            Check if there were any x0 values for which a correct possible compound was
            not provided (currently on works on "comp feat" data)
    '''
    notFound = []
    notFoundIdx = []

    with open('./data/peakGraph/test2.txt', 'w') as file:
        for i in range(len(x0)):
            compounds, masses = getAllPossibleCompounds(x0[i])
            found = False
            for j in range(len(compounds)):
                found = True if compareCompounds(compounds[j], decoded_compounds[i]) else False
                if found:
                    found = True
                    break
            if not found:
                notFound.append(compounds)
                notFoundIdx.append(i)

        for x in range(len(notFound)):
            file.write(str(x0[notFoundIdx[x]]) + "\n")
            for y in notFound[x]:
                file.write(y + "\n")
            file.write("\n")
        file.close()
    print(len(notFound))


def createSmoteSamples():
    '''
        Function:
            Utilize SMOTE oversampling technique to create more positive samples using data
            found in generated "combo" files
    '''
    import numpy as np

    x_train, x_test, y_train, y_test = getData('newCombos.txt')

    from imblearn.over_sampling import SMOTE
    import numpy as np

    unique, count = np.unique(y_train, return_counts=True)
    y_val_count = { k:v for (k, v) in zip(unique, count) }
    print(y_val_count)
    # sampling_strat = { 0: 3604, 1: 2000 }

    # smote = SMOTE(random_state=42, sampling_strategy=sampling_strat)
    # decoded = decode_compounds(x_train)
    # x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    # unique, count = np.unique(y_train_res, return_counts=True)
    # y_val_count = { k:v for (k, v) in zip(unique, count) }
    # print(y_val_count)

    # decoded = decode_compounds(x_train_res[:, :12])
    # ppms = x_train_res[:, 12]
    # print(decoded[0], x_train_res[0][12])

    # with open("./data/peakGraph/SMOTE_gen_samples.txt", "w") as file:

    #     for i in range(len(decoded)):
    #         file.write(str(decoded[i]) + '\t' + str(ppms[i]) + '\t' + str(y_train_res[i]) + '\n')

    #     # Seperate SMOTE gen samples from genuine
    #     decoded = decode_compounds(x_test[:, :12])
    #     ppms = x_test[:, 12]

    #     for i in range(len(decoded)):
    #         file.write(str(decoded[i]) + '\t' + str(ppms[i]) + '\t' + str(y_test[i]) + '\n')

    #     file.close()

def deleteDuplicates():
    '''
        Function:
            Clean up data set and get rid of unknown samples as well as duplicates in labeled data file
    '''
    lines = []
    x0 = []

    with open("./data/peakGraph/a.txt", 'r') as f:
        data = f.read().split('\n')
        for line in data[1:]:
            if line == "" or "unknown" in line.split("\t")[2]:
                continue
            if line.split("\t")[3] not in x0:
                lines.append(line)
                x0.append(line.split("\t")[3])
        f.close()

    with open("./data/peakGraph/b.txt", 'w') as f:
        for line in lines:
            f.write(line + '\n')

        f.close()

def createComboFile(mode, percentZeroSamples, outputFileName):
    '''
        Function:
            Create a file with all labeled possible compound combinations for all x0's in
            labeled peak data file
        Parameters:
            mode (str): the features to use for the data sample ('comp', 'crit', 'comb')
            percentZeroSamples (int): percent of samples labeled 0 to include
            outputFileName (str): name of file to output to
    '''
    with open(f'./data/peakGraph/labeled/comboFiles/{outputFileName}', 'w+') as fw:
        for i in range(len(x0)):
            dc, _ = getAllPossibleCompounds(x0[i])

            _, ec = encodeCompounds(dc)
            ec = np.array(ec)
            ec = [unsimplifyCompound(compound) for compound in ec]
            for j in range(len(ec)):
                _, criterea_encoding = checkCriteria(ec[j])
                ppm = calculatePPM(ec[j], x0[i])
                isFound = compareCompounds(decoded_compounds[i], dc[j])
                if random.randint(0, 100) < percentZeroSamples and not isFound:
                    continue
                if mode == 'comb' or mode == 'crit':
                    for x in criterea_encoding:
                        fw.write(str(x) + '\t')
                if mode == 'comb' or mode == 'comp':
                    for x in ec[j]:
                        fw.write(str(x) + '\t')
                fw.write(str(ppm) + '\t' + str(int(isFound)) + '\n')
        fw.close()

# -----------------------------
# Dataset creation
# -----------------------------

df, encoded_compounds = read_txt("./data/peakGraph/labeled/test.txt")
decoded_compounds = decode_compounds(encoded_compounds)

x0 = np.array(df['x_0'], dtype=float)
xLo = np.array(df['x_lo'], dtype=float)
xHi = np.array(df['x_hi'], dtype=float)

x = []

createComboFile('comb', 0, 'combFeatTestCombos.txt')
# trainTestSplit('./data/peakGraph/labeled/b.txt')