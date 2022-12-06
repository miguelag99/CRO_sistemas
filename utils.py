import numpy as np
import random
import cv2
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import time


class CRO():
    def __init__(self,file_name,repeat_clients = False):
        extra_data = loadmat(file_name)
        self.points_df = pd.DataFrame(data=np.concatenate((np.array(extra_data['bt']),\
            np.array(extra_data['C'])),axis = 1),columns=['x','y','cost'])
        self.points_df['clients_in_range'] = 0
        self.clients_df = pd.DataFrame(data= np.array(extra_data['xp']),columns=['x','y'])
        self.clients_df['covered'] = False

        for index in range(len(self.points_df)):
           self.points_df.at[index,'clients_in_range'] = [np.zeros((1,500))]

        self.get_clients_in_coverage(repeated=repeat_clients) # Si se selecciona repeated, los clientes pueden estar en varias estaciones a la vez

        self.N = 100    #Se trabaja con 100 individuos
        self.M = 30     #Cada solucion se compone por 30 puntos
        self.coral_map = np.zeros((self.N,self.M))

        self.rho = 0.4  #Define la cantidad inicial (ratio) de individuos libres
        
        # Parametros de la funcion de fitness normalizados
        self.alpha = 0.25/len(self.clients_df)   # Normalizamos entre el total de clientes
        self.beta = 0.75/sum(self.points_df['cost']) # Normalizamos entre el coste total de todas las estaciones
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

    def get_dataframes(self):
        
        return self.points_df, self.clients_df

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

    def get_clients_in_coverage(self, repeated = True):
        # A partir de la posicion de un ap se calcula cuantos clientes estan dentro
        tic = time.time()
        print('Calculando los clientes en cada zona de cobertura')
        for index in tqdm(range(len(self.points_df))):

            ap_pose = (self.points_df.iloc[index]['x'],self.points_df.iloc[index]['y'])     #posicion de la estacion base
            clients_in_range = self.points_df.iloc[index]['clients_in_range'][0][0]         #lista de clientes en rango (vacia) de un AP
            
            for client_i in range(len(clients_in_range)):
                client_pose = (self.clients_df.iloc[client_i]['x'],self.clients_df.iloc[client_i]['y']) #posicion del cliente
                dist = calculate_euclidean_distance(ap_pose,client_pose)
                
                if repeated:
                    if dist < 0.35:
                        clients_in_range[client_i] = int(1)
                else:
                    if dist < 0.35 and self.clients_df.iloc[client_i]['covered'] == False:
                        clients_in_range[client_i] = int(1)
                        self.clients_df.at[client_i,'covered'] = True

            self.points_df.at[index,'clients_in_range'] = [clients_in_range]
        print(f'Tiempo de calculo distancias de clientes es {time.time()-tic}')

   
        # copy_ponts = self.points_df
        # copy_ponts['clients_in_range'] = copy_ponts['clients_in_range'].apply(lambda x: np.sum(x[0]))

        # copy_ponts.to_csv('bs_lowest_clients.csv',header=False,index=False)


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
    # Realiza mutación de una componente aleatoria de la solucion 

    mutation_index = random.sample(range(30),1)[0]
    new_val = random.sample(range(100),1)[0]

    list1[mutation_index] = new_val

    return test_duplicated_solution(np.array(list1))

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


