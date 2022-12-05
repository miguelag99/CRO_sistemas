import os
import numpy as np
import pandas as pd
import random
import time
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.io import loadmat

from utils import crossover, mutation, calculate_euclidean_distance, print_bt, print_solution

SAVE_PATH = 'results_75_25/'

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
        
        # Parametros de la funcion de fitness normalizados
        self.alpha = 0.75/len(self.clients_df)   # Normalizamos entre el total de clientes
        self.beta = 0.25/sum(self.points_df['cost']) # Normalizamos entre el coste total de todas las estaciones
        # self.alpha = 1   # Normalizamos entre el total de clientes
        # self.beta = 0 # Normalizamos entre el coste total de todas las estaciones

        # Parametro de depredacion
        self.Pd = 0.1

        self.cro_init()

    def cro_init(self, verbose = False):
        # Rellena la lista inicial de individuos con soluciones aleatorias

        full_cells = int((1-self.rho)*self.N)   #Numero de soluciones (corales) llenas

        # Generamos las soluciones iniciales que se van a poblar, el resto son espacios vacios
        self.poblated_id = sorted(random.sample(range(100),full_cells))
        # print(f'Se han rellenado las siguientes id{sorted(id)}')
        
        for index in self.poblated_id[0:full_cells]:
            # Para cada solucion seleccionada rng, se rellena con posiciones aleatorias (de bt)
            solution = random.sample(range(100),self.M)
            self.coral_map[index,:] = solution

        if verbose:
            pd.DataFrame(self.coral_map).astype(int).to_csv('coral_map.csv',header=False,index=False)

    def get_coral_data(self):
        
        return self.poblated_id, self.coral_map

    def update_coral(self, new_coral_map):
        # actualiza el coral y poblated id (ver que soluciones estan libres y cuales no)
        self.coral_map = new_coral_map
        summary = np.sum(self.coral_map,axis = 1,dtype=int)     # Se suma cada fila para ver cuales de las pos estan vacias
        summary[summary > 0] = 1                                # Se convierte a binario
        self.poblated_id = list(np.array(range(100))[summary.astype(bool)])                           # Se actualiza en la clase

        

    def get_sol_fitness(self,sol):
        # Metodo para el calculo del fitness de una solucion de 30 elementos o id

        global_fitness = 0 #Fitness total de la solucion (30 elementos)

        ## Iteramos por cada punto de la solucion que se corresponde a un id del dataframe de 100 puntos
        # for index in range(len(sol)):
        #     pos_id = int(sol[index])
        #     C2 = float(self.points_df.iloc[pos_id]['cost'])
        #     C1 = np.sum(self.points_df.iloc[pos_id]['clients_in_range'][0])
        #     local_fitness = self.alpha*(C1) + self.beta*(1/C2)  # Fitness de un elemento (id) de la solucion
        #     global_fitness = global_fitness + local_fitness

        local_fitness = np.vectorize(lambda x: self.alpha*(np.sum(self.points_df.iloc[int(x)]['clients_in_range'][0])) + \
                                             self.beta*(1/float(self.points_df.iloc[int(x)]['cost'])))(sol)

        # print(local_fitness)

        value_sum = np.sum(local_fitness, dtype = np.float32)
        
        return value_sum


    def get_coral_fitness(self, ranked = True):

        coral_ranking = pd.DataFrame(data={'id':range(self.coral_map.shape[0]),\
            'fitness':np.zeros((self.coral_map.shape[0]),dtype=np.float32)})

        # coral_ranking_fitness = np.vectorize(lambda i:  self.get_sol_fitness(self.coral_map[i]) \
        #                                         if int(self.poblated_id[i]) == 1 \
        #                                         else coral_ranking['fitness'].iloc[i] \
        #                                         )(list(range(len(self.poblated_id))))
        # coral_ranking['fitness'] = [i for _,i in sorted(zip(coral_ranking.iloc[:,0],coral_ranking_fitness))]

        # for full_flag, solution, row in zip(self.poblated_id,self.coral_map,coral_ranking.iterrows()):      # Iteramos a la vez por el coral map y la lista en bin si esta ocupad@                                                                                            
        #    coral_ranking.iloc[row[0],1] = self.get_sol_fitness(solution)                                   # dataframe con fitness y id

        for full_id, solution, idx in zip(self.poblated_id,self.coral_map, range(len(self.coral_map))):        # Iteramos a la vez por el coral map y la lista en bin si esta ocupad@                                                                                  
            if np.sum(solution) > 0:
                coral_ranking.iloc[idx,1] = self.get_sol_fitness(solution)                            
                coral_ranking.iloc[idx,0] = full_id 
            else:
                coral_ranking.iloc[idx,1] = 0
                coral_ranking.iloc[idx,0] = full_id

        if ranked:
            return coral_ranking.sort_values(by='fitness',ascending=False)                                  # se ordenan por fitness
            
        else:
            return coral_ranking

    def get_best_solution(self):

        # Sacamos la mejor solución final
        sorted_sol = self.get_coral_fitness(ranked=True)
        # print(f'La mejor solución es la {sorted_sol.iloc[0,0]} con un fitness de {sorted_sol.iloc[0,1]}')

        positions = np.zeros((30,2))

        for id,i in zip(self.coral_map[sorted_sol['id'].iloc[0],:], range(30)):
            
            element = self.points_df.iloc[int(id)]
            positions[i,0] = element['x']
            positions[i,1] = element['y']

        return positions, sorted_sol['fitness'].iloc[0]

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

   
        # copy_ponts = self.points_df
        # copy_ponts['clients_in_range'] = copy_ponts['clients_in_range'].apply(lambda x: np.sum(x[0]))

        # copy_ponts.to_csv('bs_lowest_clients.csv',header=False,index=False)


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
        

def larvae_setting(larvae_list, coral_class, k):

    _, coral_map = coral_class.get_coral_data()
    
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
                # print(f'He puesto una larva en la pose {try_pose} no habia nada')
                break
            else:
                settled_fitness = coral_class.get_sol_fitness(coral_map[try_pose,:])    # Fitness de la solucion asentada
                if larvae_fitness > settled_fitness:
                    coral_map[try_pose,:] = larvae_sol
            
                    # print(f'He puesto una larva de fit {larvae_fitness} en la pose {try_pose} que tenia un fit de {settled_fitness}')
                    break
                else:
                    n_try = n_try + 1

    return coral_map



def depredation(coral_class, Pd = 0.1):
    
    coral_ranking = coral_class.get_coral_fitness()
    filled_ranking = coral_ranking[coral_ranking['fitness']!= 0]

    n_depredated = int(np.floor(Pd*filled_ranking.shape[0]))                  # numero de depredados
    # filled_ranking['fitness'][-1-n_depredated:] = 0                         # borramos el fitness a los depredados

    ids = filled_ranking[-1-n_depredated:]['id']
    
    _, coral_map = coral_class.get_coral_data()

    for id in ids:
        coral_map[id,:] = np.zeros((1,30))

    # coral_map[ids,:] = np.zeros((1,30))
    return coral_map





def main():

    iter = 1000   # Numero de iteraciones del algoritmo
    coral_class = CRO('Practica_Sist_Tec_Teleco.mat')

    # Create save folder
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # init cv2 image and video writer
    img = np.zeros((1000,1000,3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = cv2.VideoWriter(SAVE_PATH+'output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1000,1000))

    img = print_bt(coral_class.points_df, img)
    out.write(img)

    # cv2 video show
    # cv2.imshow('frame',img)
    # cv2.waitKey(1)

    # np array to store the best solution of each iteration
    fitness_solutions = []
    plot_points = []
    best_fit = 0



    ## Fases de reproduccion
    main_iterator = tqdm(range(iter))
    for i in main_iterator:
        
 
        poblated_id, coral_map = coral_class.get_coral_data()   # Poblated_id son los indices de coral_map que estan con corales
        larvae = np.array([])       # Aqui se van a almacenar las larvas que se generan

        # corals_left son los indices que se van a utilizar para brooding (los que sobran de broadcast_spawning)
        larvae, corals_left = broadcast_spawning(poblated_id, 0.9, larvae, coral_map) #Fb define la cantidad de broadcast spawners respecto al total de corales

        
        larvae = brooding(corals_left ,larvae ,coral_map).astype(int)
   

        ## Fase de asentamiento

        coral_class.update_coral(larvae_setting(larvae,coral_class,3))


        ## TODO Asexual reproduction


        ## Fase de depredacion

        coral_class.update_coral(depredation(coral_class))


        sol, loc_best_fit = coral_class.get_best_solution()
        fitness_solutions.append(loc_best_fit)

        if loc_best_fit > best_fit:
            best_fit = loc_best_fit
            plot_points.append(best_fit)
        else:
            plot_points.append(best_fit)
        

        img = print_solution(sol, img)
        out.write(img)
        # cv2.imshow('frame',img)
        # cv2.waitKey(1)

        img = print_bt(coral_class.points_df, img)
        main_iterator.set_description(f'Current best fitness value {best_fit}')
    
    out.release()

    sol,_ = coral_class.get_best_solution()
    print(sol)

    np.savetxt(SAVE_PATH+'best_sol.csv',sol,delimiter=',')
    np.savetxt(SAVE_PATH+'fitness_solutions.csv',fitness_solutions,delimiter=',')

    plt.plot(list(range(iter)),plot_points)
    plt.savefig(SAVE_PATH+'fitness.png')







if __name__ == '__main__':
    main()