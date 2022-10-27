import numpy as np
import random

def crossover(list1,list2):
    # Realiza el cruce de un punto random entre dos soluciones (padres), dando dos hijos y sin comprobar que este repetido, eso se hace fuera
    split_index = random.sample(range(29),1)[0]
    half_a1 = np.array(sorted(list1[0:int(split_index)]))
    half_a2 = np.array(sorted(list2[int(split_index):]))
    half_b1 = np.array(sorted(list1[int(split_index):]))
    half_b2 = np.array(sorted(list2[0:int(split_index)]))

    child1 = np.concatenate((half_a1,half_a2))
    child2 = np.concatenate((half_b1,half_b2))

    return np.vstack((child1,child2))

def mutation(list1):
    # Realiza mutación de una componente aleatoria de la solucion sin comprobar que este repetido, eso se hace fuera

    mutation_index = random.sample(range(30),1)[0]
    new_val = random.sample(range(100),1)[0]

    list1[mutation_index] = new_val

    return np.array(list1)

def test_duplicated(self):
    pass


def check_bounds(self):
    pass

def calculate_euclidean_distance(p1,p2):
    a = np.power(p1[0]-p2[0],2)
    b = np.power(p1[1]-p2[1],2)
    return np.sqrt([a+b])