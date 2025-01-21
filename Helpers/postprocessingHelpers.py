import numpy as np
from Helpers.preprocessingHelpers import encodeCompounds, unsimplifyCompound, decode_compounds, getAllPossibleCompounds
from Helpers.chemHelpers import calculateMass, checkCriteria
from Helpers.metricsHelpers import calculatePPM

def sortPreds(ec, compounds, preds, encodings, ppms):
    '''
        Functions:
            Sort encoded and decoded compounds by preds

        Parameters:
            ec (list(list(float))): encoded compounds
            compounds (list(str)): decoded simplified compounds
            preds (list(float)): predicted labels
            encodings (list(list(int))): criterea encodings

        Returns:
            ec, compounds, result (list(list(float)), list(str), dict(string, float)): 
                encoded_compounds, 
                decoded unsimplified compounds,
                map of all unsimplified decoded compounds to their predictied label
    '''
    mask = np.argsort(preds)
    preds = preds[mask]
    compounds = compounds[mask]
    encodings = np.flip(encodings[mask], axis=0)
    ppms = np.flip(ppms[mask], axis=0)
    ec = np.flip(ec[mask], axis=0)
    
    result = [ (c, p) for (c, p) in zip(list(compounds), list(preds)) ]
    result.reverse()

    return ec, decode_compounds(ec), result, encodings, ppms

def getPreds(model, value, mode='comp'):
    '''
        Function:
            Make predicions on data
        
        Parameters:
            model (Objects.Model || tf.keras.Model): the model to be used to make predictions
            value (float): the mass value(x0) of the peak we are analyzing
            mode (str): the features to use for the data sample ('comp', 'crit', 'comb')

        Returns:
            compounds, ec, preds, encodings (list(str), np.array(np.array(np.float32)), np.array(np.float32), np.array(np.array(np.int8))):
                the decoded simplified compounds,
                the encoded unsimplified compounds,
                the predicted labels,
                the criterea encodings
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
    crit_enc = []
    ppms = []
    if mode == 'comp':
        inp = [[*j, calculatePPM(j, value)] for j in ec]
    elif mode == 'crit':
        for j in range(len(ec)):
            _, encoding = checkCriteria(ec[j])
            ppm = calculatePPM(ec[j], value)
            ppms.append(ppm)
            inp.append([*encoding, ppm])
    elif mode == 'comb':
        for j in range(len(ec)):
            _, encoding = checkCriteria(ec[j])
            crit_enc.append(encoding)
            ppm = calculatePPM(ec[j], value)
            ppms.append(ppm)
            inp.append([*encoding, *ec[j], ppm])

    preds = np.array(model.predict(np.array(inp)))
    # reshape only for tensorflow models, doesn't affect Object.Model models
    preds = np.reshape(preds, newshape=(len(inp),))

    return np.array(compounds), np.array(ec), np.array(preds), np.array(crit_enc), np.array(ppms)

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
    with open(peakFile, 'r') as f:
        lines = f.readlines()
        for i in lines:
            print(float(i))
            predPack = getPreds(model, float(i), mode)
            if (predPack == None):
                continue
            compounds, ec, preds, encodings = predPack
            ec, dec, allPreds, encodings = sortPreds(ec, compounds, preds, encodings)

            with open(outputFile, 'w+') as fw:
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

        Returns:
            compounds, preds, unsimplified_compounds, crit_encodings (list(str), list(str), list(str), list(list(str))):
                the simplified possible compounds,
                the confidence scores,
                the unsimplified compounds,
                the criterea encodings
    '''
    compounds, ec, preds, encodings, ppms = getPreds(model, value, mode)
    ec, dec, allPreds, encodings, ppms = sortPreds(ec, compounds, preds, encodings, ppms)

    print(str(value) + ':')
    for j in range(len(dec)):
        if (checkCriteria(ec[j])[0]):
            print(str(dec[j]) + '\t' + str(allPreds[j][1]))
    print('----------------------')
    for j in allPreds:
        print(j[0] + ', ', j[1])

    return [str(i[0]) for i in allPreds], [str(i[1] * 100)[:5] for i in allPreds], dec, [[str(j) for j in i] for i in encodings], [str(i * 100)[:6] for i in ppms]