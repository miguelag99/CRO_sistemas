from re import X
from turtle import pos
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

from scipy.io import loadmat

from utils import crossover, mutation, calculate_euclidean_distance

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
           self.points_df.at[index,'clients_in_range'] = [np.zeros((1,500))]

        self.get_clients_in_coverage()

        self.N = 100    #Se trabaja con 100 individuos
        self.M = 30     #Cada solucion se compone por 30 puntos
        self.coral_map = np.zeros((self.N,self.M))

        self.rho = 0.4  #Define la cantidad inicial (ratio) de individuos libres
        
        # Parametros de la funcion de fitness
        self.alpha = 1
        self.beta = 1

        # Parametro de depredacion
        self.Pd = 0.1

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
        # self.update_poblated_id
        return self.poblated_id, self.coral_map

    def update_coral(self, new_coral_map):
        # actualiza el coral y poblated id (ver que soluciones estan libres y cuales no)
        self.coral_map = new_coral_map
        summary = np.sum(self.coral_map,axis = 1,dtype=int)     # Se suma cada fila para ver cuales de las pos estan vacias
        summary[summary > 0] = 1                                # Se convierte a binario
        self.poblated_id = summary                              # Se actualiza en la clase
        

    def get_sol_fitness(self,sol):
        # Metodo para el calculo del fitness de una solucion de 30 elementos o id

        global_fitness = 0 #Fitness total de la solucion (30 elementos)

        ## Iteramos por cada punto de la solucion que se corresponde a un id del dataframe de 100 puntos
        # for index in range(len(sol)):
        #     pos_id = int(sol[index])
        #     C2 = float(self.points_df.iloc[pos_id]['cost'])
        #     C1 = np.sum(self.points_df.iloc[pos_id]['clients_in_range'][0][0])
        #     local_fitness = self.alpha*(C1) + self.beta*(1/C2)  # Fitness de un elemento (id) de la solucion
        #     global_fitness = global_fitness + local_fitness

        local_fitness = np.vectorize(lambda x: self.alpha*(np.sum(self.points_df.iloc[int(x)]['clients_in_range'][0][0])) + \
                                            self.beta*(1/float(self.points_df.iloc[int(x)]['cost'])))(sol)

        return np.sum(local_fitness, dtype = np.float32)


    def get_coral_fitness(self, ranked = True):

        coral_ranking = pd.DataFrame(data={'id':range(self.coral_map.shape[0]),\
            'fitness':np.zeros((self.coral_map.shape[0]))} ,dtype=int)

        coral_ranking_fitness = np.vectorize(lambda i:  self.get_sol_fitness(self.coral_map[i]) \
                                                if int(self.poblated_id[i]) == 1 \
                                                else coral_ranking['fitness'].iloc[i] \
                                                )(list(range(len(self.poblated_id))))
        coral_ranking['fitness'] = [i for _,i in sorted(zip(coral_ranking.iloc[:,0],coral_ranking_fitness))]

        # for full_flag, solution, row in zip(self.poblated_id,self.coral_map,coral_ranking.iterrows()):      # Iteramos a la vez por el coral map y la lista en bin si esta ocupad@
        #     if int(full_flag) == 1:                                                                         # Si hay solucion
        #        coral_ranking['fitness'].iloc[row[0]] = self.get_sol_fitness(solution)                      # dataframe con fitness y id

        if ranked:
            return coral_ranking.sort_values(by='fitness',ascending=False)                                  # se ordenan por fitness
        else:
            return coral_ranking

    def get_best_solution(self):

        # Sacamos la mejor solución final
        sorted_sol = self.get_coral_fitness(ranked=True)
        positions = np.zeros((30,2))

        for id,i in zip(self.coral_map[sorted_sol['id'].iloc[0],:], range(30)):
            element = self.points_df.iloc[int(id)]
            positions[i,0] = element['x']
            positions[i,1] = element['y']

        return positions

    def get_clients_in_coverage(self):
        # A partir de la posicion de un ap se calcula cuantos clientes estan dentro
        tic = time.time()
        print('Calculando los clientes en cada zona de cobertura')
        for index in tqdm(range(len(self.points_df))):

            ap_pose = (self.points_df.iloc[index]['x'],self.points_df.iloc[index]['y'])     #posicion de la estacion base
            clients_in_range = self.points_df.iloc[index]['clients_in_range'][0][0]         #lista de clientes en rango (vacia) de un AP
            
            for client_i in range(len(clients_in_range)):
                client_pose = (self.clients_df.iloc[client_i]['x'],self.clients_df.iloc[client_i]['y']) #posicion del cliente
                dist = calculate_euclidean_distance(ap_pose,client_pose)
                
                if dist < 0.35:
                    clients_in_range[client_i] = int(1)

            self.points_df.at[index,'clients_in_range'] = [clients_in_range]
        print(f'Tiempo de calculo distancias de clientes es {time.time()-tic}')



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

    for index in corals_id: 

        parent = coral_map[index,:]
        # Hacemos la mutacion
        larvae_list =  np.vstack((larvae_list,mutation(parent))) if\
             larvae_list.size else mutation(parent)

    # Comprobar en las mutaciones no se repiten
    return larvae_list  
        

def larvae_setting(larvae_list, coral_map, coral_class, k):

    for index in range(len(larvae_list)):
        larvae_sol = larvae_list[index,:]
        larvae_fitness = coral_class.get_sol_fitness(larvae_sol)    # Calcular el fitness de la larva

        # Intentar asentarse (como maximo k veces) en una posicion aleatoria que viene definida por un id de la lista
        n_try = 0
    
        while n_try < k:
            try_pose = int(random.sample(range(100),1)[0])
            
            # Si la posicion (id) esta vacia (todas las componentes a 0), si no se compara con la que está asentada
            if int(np.sum(coral_map[try_pose,:])) == 0:
                coral_map[try_pose,:] = larvae_sol
                # print(f'He puesto una larva en la pose {try_pose} en {n_try} intentos')
                break
            else:
                settled_fitness = coral_class.get_sol_fitness(coral_map[try_pose,:])    # Fitness de la solucion asentada
                if larvae_fitness > settled_fitness:
                    coral_map[try_pose,:] = larvae_sol
                    # print(f'He puesto una larva en la pose {try_pose} en {n_try} intentos')
                    break
                else:
                    n_try = n_try + 1

    coral_class.update_coral(coral_map)





def depredation(coral_class, Pd = 0.1):
    
    coral_ranking = coral_class.get_coral_fitness()
    filled_ranking = coral_ranking[coral_ranking['fitness']!= 0]

    n_depredated = int(np.floor(Pd*filled_ranking.shape[0]))                  # numero de depredados
    # filled_ranking['fitness'][-1-n_depredated:] = 0                         # borramos el fitness a los depredados

    ids = filled_ranking[-1-n_depredated:]['id']

    _, coral_map = coral_class.get_coral_data()

    # for id in ids:
    #     coral_map[id,:] = np.zeros((1,30))


    coral_map[ids,:] = np.zeros((1,30))
    coral_class.update_coral(coral_map)





def main():

    iter = 2   # Numero de iteraciones del algoritmo
    coral_class = CRO('Practica_Sist_Tec_Teleco.mat')

    phases_times = np.zeros((iter,4),dtype = np.float16)

    ## Fases de reproduccion
    for i in tqdm(range(iter)):
        poblated_id, coral_map = coral_class.get_coral_data()   # Poblated_id son los indices de coral_map que estan con corales
        larvae = np.array([])       # Aqui se van a almacenar las larvas que se generan

        # corals_left son los indices que se van a utilizar para brooding (los que sobran de broadcast_spawning)
        tic = time.time()
        larvae, corals_left = broadcast_spawning(poblated_id, 0.9, larvae, coral_map) #Fb define la cantidad de broadcast spawners respecto al total de corales
        phases_times[i,0] = time.time() - tic
        
        tic = time.time()
        larvae = brooding(corals_left ,larvae ,coral_map).astype(int)
        phases_times[i,1] = time.time() - tic
        # np.savetxt('larvas.csv',larvae,delimiter=',', fmt='%.0d')
        
        ## Fase de asentamiento
        tic = time.time()
        larvae_setting(larvae,coral_map,coral_class,3)
        phases_times[i,2] = time.time() - tic


        ###################### TODO Falta fase de asexual reproduction


        ## Fase de depredacion
        tic = time.time()
        depredation(coral_class)
        phases_times[i,3] = time.time() - tic
    
    print(np.mean(phases_times,axis=0))

    sol = coral_class.get_best_solution()
    print(sol)






if __name__ == '__main__':
    main()