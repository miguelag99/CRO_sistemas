import numpy as np
import pandas as pd
import random

from scipy.io import loadmat

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
        self.points_df['id'] = list(range(0,100))   #ID de cada posible punto

        self.clients_df = pd.DataFrame(data= np.array(extra_data['xp']),columns=['x','y'])
   
        self.N = 100    #Se trabaja con 100 individuos
        self.M = 30     #Cada solucion se compone por 30 puntos
        self.coral_map = np.zeros((self.N,self.M))

        self.rho = 0.4  #Define la cantidad inicial (ratio) de individuos libres
        

        self.cro_init()

    def cro_init(self):
        # Rellena la lista inicial de individuos con soluciones aleatorias

        full_cells = int((1-self.rho)*self.N)   #Numero de soluciones (corales) llenas

        # Generamos las soluciones iniciales que se van a poblar, el resto son espacios vacios
        id = random.sample(range(100),full_cells)
        # print(f'Se han rellenado las siguientes id{sorted(id)}')
        
        

        for index in id[0:full_cells]:
            # Para cada solucion seleccionada rng, se rellena con posiciones aleatorias (de bt)
            solution = random.sample(range(100),self.M)
            self.coral_map[index,:] = solution

        pd.DataFrame(self.coral_map).astype(int).to_csv('aaaaa.csv',header=False,index=False)

    def test_duplicated(self):
        pass

        
        

# Comprobar en las mutaciones que estas dentro de los limites


def main():

    iter = 1000
    alg_obj = CRO('Practica_Sist_Tec_Teleco.mat')

    
    



if __name__ == '__main__':
    main()