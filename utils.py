import numpy as np

def crossover(list1,list2):
    # Realiza el cruce de un punto entre dos soluciones (padres)
    half1 = list1[0:int(len(list1)/2)]
    half2 = list2[int(len(list2)/2):]

    return np.concatenate((half1,half2))

def mutation(list1):
    # Realiza mutación de una solución
    
    return list1