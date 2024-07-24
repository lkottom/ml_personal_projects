import numpy as np 

class Conv3x3():
    # A Convolutional layer using 3x3 filters

    def __init__(self, number_filters):
        self.number_filters = number_filters

        # Filters is a 3D array with dimensions (number_filters, 3, 3)
        # We divide by 9 here to reduce the varience of our intitial values 
        self.filters = np.random.randn(number_filters, 3, 3)
    
    def iterate_regions(self, image):
        '''
        Generates all the possible 3x3 image regions uisng valid padding 

        :param image: The input image that is being iterated over
        :type image: 2D numpy array

       '''