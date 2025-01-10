from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

chems = { 'H3O': 0, 'NH4': 1, 'NH3': 2, 'H2O': 3, 'C': 4, 'H': 5, 'O': 6, 'N': 7, 'S': 8, 'F': 9, 'Si': 10, '+': 11}
chem_masses = { 'H3O': 19.01838971, 'NH4': 18.03437412, 'NH3': 17.02654909, 'H2O': 18.010564679999998, 'C': 12, 'H': 1.00782503, 'O': 15.99491462, 'N': 14.00307400, 'S': 31.97207117, 'F': 18.998403, 'Si': 28.0855, '+': -0.000548 }

def encodeCompounds(chemicals):
    '''
        Function:
            Encode a list chemical compounds
        
        Parameters:
            chemicals (list(str)): Unencoded compound strings

        Returns:
            df, encoded_compounds (pd.DataFrame, list(int)): 
                Encoded compounds in a dataframe
                Encoded compounds in a list
    '''
    encoded_compounds = []

    for chemical in chemicals:
        encoding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        while '(' in chemical:
            end = chemical.index(")")
            a = chemical[chemical.index("(") + 1 : end]
            encoding[chems[a]] = .01
            chemical = chemical[end + 1:]
        
        for i in range(len(chemical)):
            c = chemical[i]
            if c == '+':
                encoding[chems[c]] = .01
            if (not (c >= 'A' and c <= 'Z')):
                continue
            
            num = ""
            start = 1
            if i + 1 < len(chemical) - 1:
                if (chemical[i + 1] >= 'a' and chemical[i + 1] <= 'z'):
                    c = chemical[i : i + 2]
                    start = 2
            for j in range(i + start, len(chemical)):
                if ((chemical[j] >= 'A' and chemical[j] <= 'Z') or chemical[j] == '+'):
                    break

                num += chemical[j]

            if num == '':
                num = 1
            encoding[chems[c]] = float(num) / float(100)
        if encoding != [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            encoded_compounds.append(encoding)

    df = pd.DataFrame()
    ctr = 0
    for key in chems:
        df.insert(loc=ctr, column=key, value=[i[ctr] for i in encoded_compounds])
        ctr += 1

    return df, encoded_compounds

def allPossibleCompounds(res, arr, masses, temp, sum, index):
    '''
        Function:
            Recursive method to find all possible compunds that have a mass close to the target
        
        Paramaters:
            res (list(list(doubles))): All of the masses that add up to the mass of the compound
            arr (list(doubles)): Values of the masses of the chemicals were working with
            masses (list(double)): All of the summed masses of all the lists in the "res" 2d list
            temp (list(double)): List of current masses in the iteration of the recursive loop
            sum (double): sum of "temp" list
            index (int): Current index
    '''
    if math.isclose(a=sum, b=0, abs_tol=.01) and temp not in res:
        mass = np.sum(np.array(list(temp)))
        masses.append(mass)
        res.append(temp[:])
        return
    
    for i in range(index, len(arr)):
        if (sum - arr[i]) > -.01:
            temp.append(arr[i])
            allPossibleCompounds(res, arr, masses, temp, sum-arr[i], i)
            temp.pop()

def getAllPossibleCompounds(sum):
    '''
        Function: 
            Gets all possible mass combinations and converts them to compounds

        Parameters:
            sum (double): Mass from which the combinations derive
        
        Returns:
            compounds (list(str)): All of the compounds
            masses (list(double)): All of the compounds' summed masses
    '''
    chem_mass_key = [i for i in chem_masses][4:8]
    chem_mass_value = [i for i in chem_masses.values()][4:8]


    # Manual bubble sort so same changes are made to keys for decoding purposes
    for i in range(len(chem_mass_key)):
        for j in range(i, len(chem_mass_key)):
            if chem_mass_value[i] < chem_mass_value[j]:
                temp = chem_mass_value[i]
                chem_mass_value[i] = chem_mass_value[j]
                chem_mass_value[j] = temp

                temp = chem_mass_key[i]
                chem_mass_key[i] = chem_mass_key[j]
                chem_mass_key[j] = temp

    res = []
    masses = []

    allPossibleCompounds(res, chem_mass_value, masses, [], sum, 0)


    # turn back into compound strings
    swapped = {v: k for k, v in chem_masses.items()}
    decoded_masses = []
    for i in res:
        decoded = []
        for j in i:
            decoded.append(swapped[j])
        decoded_masses.append(decoded)

    compounds = []
    for i in decoded_masses:
        c = ''
        for key in chems:
            count = i.count(key)
            if count > 0:
                c += key
                if (count > 1):
                    c += str(count)
        compounds.append(c)

    compounds = [c + "+" for c in compounds]

    return compounds, masses

def decode_compounds(chemicals):
    '''
        Function:
            Take encoded compounds and turn them back to compound strings

        Parameters:
            chemicals (list(int)): the encoded compounds

        Returns:
            decoded_compounds (list(str)): the decoded compounds
        
    '''
    decoded_compounds = []
    for i in chemicals:
        c = ''
        for idx, k in enumerate(chems.keys()):
            if i[idx] > 0:
                c += k
                if (i[idx] * 100) > 1:
                    c += str(int(i[idx] * 100))
        decoded_compounds.append(c)

    for i in range(len(decoded_compounds)):
        for j in list(chems.keys())[1:3]:
            decoded_compounds[i] = decoded_compounds[i].replace(j, "(" + j + ")")
    
    return decoded_compounds

def read_txt(path):
    '''
        Function:
            Read labeled peak data file and structure in a list and data frame
        
        Parameters:
            path (str): path to text file

        Returns:
            df, encoded_compounds (pd.DataFrame, list(int)): 
                dataframe with the correct ions, the range of x values, and the x0 values
                all of the encoded ions
    '''
    df = pd.DataFrame()
    ec = []
    with open(path) as file:
        a = file.readlines()
        res = []

        for i, line in enumerate(a):
            if i % 2 == 0:
                res.append(line)
        res = [x.split('\t') for x in res]
        res = [x for x in res if 'unknown' not in x[2]]

        labels = res[0]

        df, ec = encodeCompounds([x[labels.index('tag') + 1] for x in res[1:]])
        df.insert(loc=12, column="x_lo", value=[x[labels.index('x_Lo')] for x in res[1:]])
        df.insert(loc=13, column="x_hi", value=[x[labels.index('x_Hi')] for x in res[1:]])
        df.insert(loc=14, column="x_0", value=[x[labels.index('x0')] for x in res[1:]])

        file.close()

    return df, ec

def simplifyCompound(compound):
    '''
        Function:
            Encode compound only using C, H, O, and N
        
        Parameters:
            compound (list(int)): encoded compound

        Returns:
            compound (np.array(np.float32)): encoded compound with only C, H, O, and N
    '''
    compound *= 100
    for idx, key in enumerate(list(chems.keys())[:4]):
        if (compound[idx] > 0):
            for i in range(len(key)):
                if key[i] >= 'A' and key[i] <= 'Z':
                    j = i + 1
                    num = ''
                    while (j < len(key)):
                        if (key[j] >= 'A' and key[j] <= 'Z'):
                            break
                        num += key[j]
                        j += 1

                    if num == '':
                        num = '1'
                    compound[chems[key[i]]] += float(num)
                    compound[chems[key]] = 0

    return np.array(compound, dtype=np.float32) / 100

def unsimplifyCompound(compound):
    '''
        Function:
            Add polyatomics into the formula
        
        Parameters:
            compound (np.array(np.float32)): simplified compounds
        
        Returns:
            unsimplifiedCompounds (np.array(np.float32)): unsimplified compounds
    '''

    compound *= 100
    if compound[chems['H']] >= 4 and compound[chems['N']] > 0:
        compound[chems['NH4']] = 1
        compound[chems['H']] = int(compound[chems['H']]) - 4
        compound[chems['N']] = int(compound[chems['N']]) - 1

    if compound[chems['H']] >= 3 and compound[chems['N']] > 0:
        compound[chems['NH3']] = 1
        compound[chems['H']] = int(compound[chems['H']]) - 3
        compound[chems['N']] = int(compound[chems['N']]) - 1


    return compound / 100

def find_peaks(mz_b, mz_av, fileName, plot=False):
    '''
        Function:
            Find peaks on an mz_base to mz_av function and writes it to a file
        
        Parameters:
            mz_b (str): path to m/z data
            mz_av (str): path to intensity data
            filename (str): name of the file to write the x0 of the peaks in
            plot (bool): whether or not to plot the (mz_base, mz_av) graph
    '''
    from chemHelpers import gaussianFit, multiPeakGaussianFit
    
    df = pd.read_csv(mz_b)
    df.insert(1, "mz_av", pd.read_csv(mz_av)["mz_av"])
    df = df.set_index("mz_base")

    p, heights = signal.find_peaks(x=df['mz_av'], height=1500, distance=None, prominence=1000)
    half_width = signal.peak_widths(df['mz_av'], p)

    x_low = df.index[(np.round(half_width[2]).astype(int))]
    x_high = df.index[(np.round(half_width[3]).astype(int))]
    peak_height = heights['peak_heights']
    x0 = df.index[p]
    std = []

    gaussian = []

    for i in range(len(x_low)):
        try:
            x_range = np.linspace(x_low[i], x_high[i], 100)
            sigma = np.std(x_range)
            std.append(sigma)
            gaussian.append(gaussianFit(x_range, x0[i], peak_height[i], sigma))
        except Exception as e:
            print(e)

    corrected_gaussians = []
    corrected_x0 = []
    replaced_gaussians = []
    replaced_gaussians_first = []
    gaussian = np.array(gaussian)
    std = np.array(std)

    for i in range(len(gaussian)):
        idx = [i]
        left = i
        right = i + 1
        while right < len(gaussian) and (gaussian[left][0][99] >= gaussian[right][0][0]):
            idx.append(right)
            left += 1
            right += 1
        
        if (len(idx) > 1):
            if 1e-7 in std[idx]:
                continue
            x_range = np.array(df.index[df.index.get_loc(x_low[idx[0]]) : df.index.get_loc(x_high[idx[-1]])])
            y_range = np.array(df['mz_av'][x_low[idx[0]] : x_high[idx[-1]]])
            initial_guesses = []
            for j in idx:
                initial_guesses.append(x0[j])
                initial_guesses.append(peak_height[j])
                initial_guesses.append(std[j])
            multi_peak = multiPeakGaussianFit(x_range, y_range, initial_guesses)
            if multi_peak:
                corrected_gaussians.append(multi_peak[:2])
                replaced_gaussians.append(idx)
                replaced_gaussians_first.append(idx[0])
                for j in multi_peak[2]:
                    print(j)
                    corrected_x0.append(j)

    final_gaussians = []
    final_x0 = []
    
    i = 0
    ctr = 0
    while i < len(gaussian) or ctr < len(replaced_gaussians):
        if i in replaced_gaussians_first or i >= len(gaussian):
            final_gaussians.append(corrected_gaussians[ctr])
            final_x0.append(corrected_x0[ctr])
            i += len(replaced_gaussians[ctr])
            ctr += 1
        else:
            final_gaussians.append(gaussian[i])
            final_x0.append(x0[i])
            i += 1

    x0 = sorted(final_x0)
    with open(f"./data/output/peaks/{fileName}.txt", 'w') as f:
        for i in x0:
            f.write(str(i) + '\n')
        f.close()

    if plot:
        plt.plot(df.index, df["mz_av"])
        plt.scatter(data=df.iloc[p].reset_index(), x='mz_base', y='mz_av')
        # plt.show()

        for j in final_gaussians:
            plt.plot(j[0], j[1])
        plt.show()
        

def getData(comboFile):
    '''
        Function:
            Get data from labeled "combo" file and seperate train and test sets
        
        Parameters:
            comboFile (str): name of the combo file

        Returns:
            x_train, x_test, y_train, y_test (list(list(float)), list(list(float)), list(int), list(int)):
                training data samples,
                test data samples,
                train labels,
                test labels
    '''

    X = []
    y = []
    with open(comboFile) as f:
        lines = f.readlines()
        for line in lines:
            segments = line.split('\t')

            sample = [float(i) for i in segments[:-1]]
            label = int(segments[-1])

            X.append(sample)
            y.append(label)
            
        f.close()

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,
                                                    random_state=42,
                                                    shuffle=True)

    return x_train, x_test, y_train, y_test