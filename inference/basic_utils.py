import numpy as np

def get_vectors(filepath):
    vectors = []
    with open(filepath) as fp:
        for line in fp:
            vectors.append( #Append the list of numbers to the result array
                [float(item) #Convert each number to an integer
                 for item in line.split() #Split each line of whitespace
                 ])
    return np.array(vectors)

def get_str_vectors(filepath):
    vectors = []
    with open(filepath) as fp:
        for line in fp:
            vectors.append( line.split()[0])
    return np.array(vectors)

def round3(x):
    return np.around(x, decimals=3)