import numpy as np
import random
import cv2
import pandas as pd

def crossover(list1,list2):
    # Realiza el cruce de un punto random entre dos soluciones (padres), dando dos hijos y sin comprobar que este repetido, eso se hace fuera
    split_index = random.sample(range(29),1)[0]
    half_a1 = np.array(sorted(list1[0:int(split_index)]))
    half_a2 = np.array(sorted(list2[int(split_index):]))
    half_b1 = np.array(sorted(list1[int(split_index):]))
    half_b2 = np.array(sorted(list2[0:int(split_index)]))

    child1 = test_duplicated_solution(np.concatenate((half_a1,half_a2)))
    child2 = test_duplicated_solution(np.concatenate((half_b1,half_b2)))

    return np.vstack((child1,child2))

def mutation(list1):
    # Realiza mutación de una componente aleatoria de la solucion sin comprobar que este repetido, eso se hace fuera

    mutation_index = random.sample(range(30),1)[0]
    new_val = random.sample(range(100),1)[0]

    list1[mutation_index] = new_val

    return np.array(list1)

def test_duplicated_solution(solution):
    # Comprueba que no haya duplicados en una solucion, si los hay los elimina e introduce un nuevo valor aleatorio

    if len(solution) == len(set(solution)):
        # print("Good solution")
        return solution
    else:
        while len(solution) != len(set(solution)):
            # print('Avoided duplicated solution')
            unique_sol = set(solution)
            n_new_elements = len(solution) - len(unique_sol)
            new_elements = random.sample(range(100),n_new_elements)
            solution = np.array(list(unique_sol) + new_elements)    
        return solution



def check_bounds(self):
    pass

def calculate_euclidean_distance(p1,p2):
    a = np.power(p1[0]-p2[0],2)
    b = np.power(p1[1]-p2[1],2)
    return np.sqrt([a+b])


# print points with cv2 and return image
def print_bt(points, img):
    for idx, bt in points.iterrows():
        cv2.circle(img,(int(bt['x']*1000/2),int(bt['y']*1000/2)),2,(0,0,255),-1)
    
    #show image
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img

def print_solution(sol,img):
    for idx in range(len(sol)):
        cv2.circle(img,(int(sol[idx][0]*1000/2),int(sol[idx][1]*1000/2)),2,(255,0,0),-1)
   
    
    #show image
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img


