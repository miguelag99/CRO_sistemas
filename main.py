from re import X
import numpy as np
import pandas as pd
import random

from scipy.io import loadmat

from utils import crossover, mutation

# Caracteristicas
# 2x2 Km
# 100 puntos especificados en bt (en km x,y)
# 500 usuarios especificados en xp
# Necesitamos 30 puntos de bt optimizando coste de cobertura y coste.
# Radio de cobertura de 0.35 km
# C1 es la componente de cobertura igual al numero de usuario dentro del rango
# C2 coste total de todas las estaciones
# g(x) = aplha*C1 + beta*f(C2)

class CRO():
    def __init__(self,file_name):
        extra_data = loadmat(file_name)
        self.points_df = pd.DataFrame(data=np.concatenate((np.array(extra_data['bt']),\
            np.array(extra_data['C'])),axis = 1),columns=['x','y','cost'])
        self.points_df['clients_in_range'] = 0
        self.clients_df = pd.DataFrame(data= np.array(extra_data['xp']),columns=['x','y'])


        for index in range(len(self.points_df)):
           self.points_df.iloc[index]['clients_in_range'] = [1,2,3]

        print(self.points_df)

        self.N = 100    #Se trabaja con 100 individuos
        self.M = 30     #Cada solucion se compone por 30 puntos
        self.coral_map = np.zeros((self.N,self.M))

        self.rho = 0.4  #Define la cantidad inicial (ratio) de individuos libres
        
        # Parametros de la funcion de fitness
        self.alpha = 1
        self.beta = 1

        self.cro_init()


    def cro_init(self, verbose = False):
        # Rellena la lista inicial de individuos con soluciones aleatorias

        full_cells = int((1-self.rho)*self.N)   #Numero de soluciones (corales) llenas

        # Generamos las soluciones iniciales que se van a poblar, el resto son espacios vacios
        self.poblated_id = random.sample(range(100),full_cells)
        # print(f'Se han rellenado las siguientes id{sorted(id)}')
        
        for index in self.poblated_id[0:full_cells]:
            # Para cada solucion seleccionada rng, se rellena con posiciones aleatorias (de bt)
            solution = random.sample(range(100),self.M)
            self.coral_map[index,:] = solution

        if verbose:
            pd.DataFrame(self.coral_map).astype(int).to_csv('coral_map.csv',header=False,index=False)

    def get_coral_data(self):
        return self.poblated_id, self.coral_map

    def get_sol_fitness(self,sol):
        # Metodo para el calculo del fitness

        # Iteramos por cada punto de la solucion
        for index in range(len(sol)):
            pos_id = sol[index]
            x = self.points_df.iloc[pos_id]['']

    def get_clients_in_coverage(self):
        # A partir de la posicion de un ap se calcula cuantos clientes estan dentro
        pass





def broadcast_spawning(corals_id,Fb,larvae_list,coral_map):

    # Seleccionar una fracción de los corales para hacer broadcast spawning
    broadcast_spawners = corals_id[0:int(Fb*len(corals_id))]
    # print(f'Los broadc spawn son{broadcast_spawners}')

    if len(broadcast_spawners)%2 != 0:
        broadcast_spawners.pop()        # Si es impar quitamos el último
    
    # Como la lista de corales ya esta randomizado se eligen parejas 2 a 2

    for index in range(int(len(broadcast_spawners)/2)):

        parents_id = (broadcast_spawners[index*2],broadcast_spawners[(index*2)+1])

        parent1 = coral_map[parents_id[0],:]
        parent2 = coral_map[parents_id[1],:]
        larvae_list =  np.vstack((larvae_list,crossover(parent1,parent2))) if\
            larvae_list.size else crossover(parent1,parent2)

    return larvae_list, corals_id[int(Fb*len(corals_id)):]    # La funcion debe devolver una matriz de larvas y la lista de poblated_id quitando los corales usados.
    

def brooding(corals_id,larvae_list,coral_map):
    
    for index in range(int(len(corals_id))):    ###################################################### cambiar por un map por favor

        parent = coral_map[index,:]

        # hacemos la mutacion
        larvae_list =  np.vstack((larvae_list,mutation(parent))) if\
             larvae_list.size else mutation(parent)

    # Comprobar en las mutaciones que estas dentro de los limites y no se repiten
    return larvae_list  
        




def main():

    iter = 1000     # Numero de iteraciones del algoritmo
    coral_class = CRO('Practica_Sist_Tec_Teleco.mat')

    poblated_id, coral_map = coral_class.get_coral_data()   # Poblated_id son los indices de coral_map que estan con corales
    larvae = np.array([])       # Aqui se van a almacenar las larvas que se generan

    # corals_left son los indices que se van a utilizar para brooding (los que sobran de broadcast_spawning)
    larvae, corals_left = broadcast_spawning(poblated_id, 0.9, larvae, coral_map) #Fb define la cantidad de broadcast spawners respecto al total de corales

    larvae = brooding(corals_left ,larvae ,coral_map)

    # print(larvae[:40,:]) ########################################## Probar a guardarlo y verlo fuera completo

    



    # Falta fase de asexual reproduction

    ## Poblated_id debe actualizarse con la posición de los hijos y con las mu


    

    
    



if __name__ == '__main__':
    main()