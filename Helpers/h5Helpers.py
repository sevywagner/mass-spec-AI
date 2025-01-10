import h5py

def iterateThroughGroups(f, datasets, belongsTo = "", data={}):
    '''

    Function: 
        Recursive function for crawling the h5 file, creating a data structure, and printing the architecture

    Parameters:
        f (h5py.KeysViewHDF5): root at which to crawl
        datasets (dict): contains all of the data from the h5 file
        belongsTo (string): the group to which the current folder to crawl belongs
        data (dict): contains all of the data in a certain group

    '''
    for key in list(f.keys()):
        print(belongsTo, " ", key)
        if (not isinstance(f[key], h5py.Dataset)):
            data = {}
            iterateThroughGroups(f[key], datasets=datasets, belongsTo=key, data=data)
            datasets[key] = data
        else:
            data[key] = f[key]
            print(f[key])
    print('\n')


def getH5Data(file):
    '''

    Function:
        Prints structure of h5 file as well as gathers all of the datasets

    Parameters:
        file (h5py.File): .h5 file

    Returns:
        data (dict): all of the data

    '''

    data = {}
    iterateThroughGroups(file, data)
    return data