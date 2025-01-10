import pickle

class Model:
    def __init__(self):
        print("initialized")

    def __getstate__(self):
        '''
            Function:
                Get state of object for saving models
        '''
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        '''
            Function:
                Get state of object for loading models
        '''
        self.__dict__.update(state)

    def saveModel(self, pathname):
        '''
            Function:
                Save fit model to a file

            Parameters:
                pathname (str): name of file to save to
        '''
        with open(pathname, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def loadModel(pathname):
        '''
            Function:
                Load fit model
            
            Parameters:
                pathname (str): name of file where model lives
        '''
        with open(pathname, 'rb') as file:
            model = pickle.load(file)
            file.close()
        return model