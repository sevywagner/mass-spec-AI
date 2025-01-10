import numpy as np
from Helpers.preprocessingHelpers import encodeCompounds, unsimplifyCompound, decode_compounds, getAllPossibleCompounds
from Helpers.chemHelpers import calculateMass, checkCriteria
from Helpers.metricsHelpers import calculatePPM

def sortPreds(ec, compounds, preds):
    '''
        Functions:
            Sort encoded and decoded compounds by preds

        Parameters:
            ec (list(list(float))): encoded compounds
            compounds (list(str)): decoded simplified compounds
            preds (list(float)): predicted labels

        Returns:
            ec, compounds, result (list(list(float)), list(str), dict(string, float)): 
                encoded_compounds, 
                decoded unsimplified compounds,
                map of all unsimplified decoded compounds to their predictied label
    '''
    ec = np.array(ec)
    compounds = np.array(compounds)
    preds = np.array(preds)

    mask = np.argsort(preds)
    preds = preds[mask]
    compounds = compounds[mask]
    ec = np.flip(ec[mask], axis=0)
    
    result = [ (c, p) for (c, p) in zip(list(compounds), list(preds)) ]
    result.reverse()

    return ec, decode_compounds(ec), result

def getPreds(model, value, mode='comp'):
    '''
        Function:
            Make predicions on data
        
        Parameters:
            model (Objects.Model || tf.keras.Model): the model to be used to make predictions
            value (float): the mass value(x0) of the peak we are analyzing
            mode (str): the features to use for the data sample ('comp', 'crit', 'comb')

        Returns:
            compounds, ec, preds (list(str), list(list(float)), list(float)):
                the decoded simplified compounds,
                the encoded unsimplified compounds,
                the predicted labels
    '''
    ec = getAllPossibleCompounds(value)[0]
    compounds = ec
    if len(ec) == 0:
        return

    df, ec = np.array(encodeCompounds(ec))
    if (len(ec) == 0):
        return None
    ec = np.array([unsimplifyCompound(j) for j in ec])


    inp = []
    if mode == 'comp':
        inp = [[*j, calculatePPM(j, value)] for j in ec]
    elif mode == 'crit':
        for j in range(len(ec)):
            _, encoding = checkCriteria(ec[j])
            ppm = calculatePPM(ec[j], value)
            inp.append([*encoding, ppm])
    elif mode == 'comb':
        for j in range(len(ec)):
            _, encoding = checkCriteria(ec[j])
            ppm = calculatePPM(ec[j], value)
            inp.append([*encoding, *ec[j], ppm])

    preds = np.array(model.predict(np.array(inp)))
    # reshape only for tensorflow models, doesn't affect Object.Model models
    preds = np.reshape(preds, newshape=(len(inp),))

    return compounds, ec, preds

def getMostLikelyCompounds(model, peakFile, outputFile, mode='comb'):
    '''
        Funtion:
            Find all of the most likely compounds for every peak in a (mz_base, mz_av) graph
            and write them to a file

        Parameters:
            model (Objects.Model || tf.keras.Model): the model to be used to make predictions
            peakFile (str): name of the file that contains the x0 value at every peak
            outputFile (str): name of the file to output the predictions to
            mode (str): the features to use for the data sample ('comp', 'crit', 'comb')

    '''
    with open(f'./data/output/peaks/{peakFile}', 'r') as f:
        lines = f.readlines()
        for i in lines:
            print(float(i))
            predPack = getPreds(model, float(i), mode)
            if (predPack == None):
                continue
            compounds, ec, preds = predPack
            ec, dec, allPreds = sortPreds(ec, compounds, preds)

            with open(f'./data/output/{outputFile}', 'a') as fw:
                fw.write(str(i))
                for j in range(len(dec)):
                    if (checkCriteria(ec[j])[0]):
                        fw.write(str(dec[j]) + '\t' + str(allPreds[j][1]) + '\n')
                fw.write('----------------------\n')
                for j in allPreds:
                    fw.write(str(j[0]) + ",\t" + str(j[1]) + '\n')

                fw.write('\n')
                fw.close()

        f.close()

def processMassValue(model, value, mode):
    '''
        Function:
            Predict and display all of the most likely compounds represented by a mass value

        Parameters:
            model (Objects.Model || tf.keras.Model): the model to be used to make predictions
            value (float): the mass value(x0) of the peak we are analyzing
            mode (str): the features to use for the data sample ('comp', 'crit', 'comb')
    '''
    compounds, ec, preds = getPreds(model, value, mode)
    ec, dec, allPreds = sortPreds(ec, compounds, preds)
    mostLikelyComp = []
    mostLikelyScore = []

    print(str(value) + ':')
    for j in range(len(dec)):
        if (checkCriteria(ec[j])[0]):
            print(str(dec[j]) + '\t' + str(allPreds[j][1]))
            mostLikelyComp.append(dec[j])
            mostLikelyScore.append(str(allPreds[j][1]))
    print('----------------------')
    for j in allPreds:
        print(j[0] + ', ', j[1])

    return [str(i[0]) for i in allPreds], [str(i[1]) for i in allPreds], dec, mostLikelyComp, mostLikelyScore