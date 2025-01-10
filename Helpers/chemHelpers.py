import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Helpers.preprocessingHelpers import encodeCompounds, simplifyCompound
import math

chems = { 'H3O': 0, 'NH4': 1, 'NH3': 2, 'H2O': 3, 'C': 4, 'H': 5, 'O': 6, 'N': 7, 'S': 8, 'F': 9, 'Si': 10, '+': 11}
chem_masses = { 'H3O': 19.01838971, 'NH4': 18.03437412, 'NH3': 17.02654909, 'H2O': 18.010564679999998, 'C': 12, 'H': 1.00782503, 'O': 15.99491462, 'N': 14.00307400, 'S': 31.97207117, 'F': 18.998403, 'Si': 28.0855, '+': -0.000548 }

def gaussian(x, A, x0, std):
    '''
        Function:
            Create a gaussian curve

        Parameters:
            x (np.array(float32)): x data
            A (float32): peak of curve
            x0 (float32): center of the curve
            std (float32): standard deviation

        Returns:
            gaussianCurve (np.array(float32)): Gaussian Curve
    '''

    return A * np.exp((-(x - x0) ** 2) / (2 * (std ** 2)))

def gaussianFit(x_range, x0, a, std, plot=False):
    '''
        Function:
            Fits and plots a curve to a gaussian wave

        Parameters:
            x_range (np.array(float32)): the range of x_values
            x0 (float): x_value (mass) at the peak,
            a (float): y_value (intensity) at the peak
            plot (bool): whether or not you want to plot the curve

        Returns:
            x, y (np.array(float32), np.array(float32)): Gaussian Curve
    '''
    
    np.random.seed(42)
    y_data = gaussian(x_range, a, x0, std) + 0.5 * np.random.normal(size=len(x_range))

    params, _ = curve_fit(gaussian, x_range, y_data, p0=[a, x0, std])
    fitted_a, fitted_x0, fitted_sigma = params
    fitted_y_data = gaussian(x_range, fitted_a, fitted_x0, fitted_sigma)

    if plot:
        plt.plot(x_range, fitted_y_data)
        plt.show()

    return x_range, fitted_y_data

def multiGaussian(x, *params):
    length = len(params) // 3
    result = np.zeros_like(x)
    for i in range(length):
        result += gaussian(x, params[i * 3 + 1], params[i * 3], params[i * 3 + 2])
    return result

def multiPeakGaussianFit(x, y, initial_guesses):
    '''
        Function:
            Fit multiple peaks whose singular fit curves intersect

        Parameters:
            x (np.array(np.float32)): range of m/z values on the actual mass to intensity graph
            y (np.array(np.float32)): range of intensity values on the actual mass to intensity graph
        
        Return:
            x, fitted_y, x0 (np.array(np.float32), np.array(np.float32), np.array(np.float32)):
                full extent of the x range,
                fit intensity,
                all fit x0 values
        
    '''
    try:
        x = np.linspace(x[0], x[-1], 100)
        y = multiGaussian(x, *initial_guesses)
        params, _ = curve_fit(multiGaussian, x, y, initial_guesses)
        fitted_y = multiGaussian(x, *params)

        return x, fitted_y, params[0::3]
    except Exception as e:
        print('optimal params not found')

def calculateMass(compound, encoded=True):
    '''
        Function:
            Calculate the mass of a compound

        Parameters:
            compound (str || list(int)): the encoded or unencoded compound
            encoded (bool): whether or not the compound is encoded
        
        Returns:
            mass (double): the mass of the compound
    '''
    mass = 0.0
    if not encoded:
        df, compound = encodeCompounds([compound])
        compound = compound[0]

    for idx, value in enumerate(chem_masses.values()):
        mass += (float(value) * (compound[idx] * 100))
    return mass

def compareCompounds(compound1, compound2):
    '''
        Function:
            Compare 2 unencoded compounds

        Parameters:
            compound1 (str): 1st compound
            compound2 (str): 2nd compound

        Returns:
            doesMatch (bool): whether or not the compounds are equal
    '''

    compound1 = compound1.replace("+", "")
    compound2 = compound2.replace("+", "")
    ec = encodeCompounds([compound1, compound2])[1]
    ec = [simplifyCompound(i) for i in ec]
    for i in range(len(ec[0])):
        if (ec[0][i] != ec[1][i]):
            return False

    return True

def checkCriteria(compound):
    '''
        Function:
            Check if a compound is possible and create one hot encoded array of critea checks

        Parameters:
            compound (np.array(np.float64)): the compound to check

        Returns:
            doesPass, encoding (bool, list(int)): if the compound is possible, one hot encoding
    '''
    compound *= 100
    encoding = [1, 1, 1, 1]
    passVal = True
    if compound[chems['NH4']] == 1:
        if compound[chems['H']] % 2 == 1:
            encoding[0] = 0
            passVal = False

        if compound[chems['H']] > (2 * compound[chems['C']]) + 2:
            encoding[1] = 0
            passVal = False

    if compound[chems['N']] == 0 and compound[chems['NH4']] == 0 and compound[chems['NH3']] == 0:
        if compound[chems['H']] % 2 == 0:
            encoding[2] = 0
            passVal = False

        if compound[chems['H']] > (2 * compound[chems['C']]) + 3:
            encoding[3] = 0
            passVal = False


    compound /= 100
    return passVal, encoding