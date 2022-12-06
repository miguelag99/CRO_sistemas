import os
import pandas as pd
import numpy as np

from utils import CRO

SAVE_PATH = 'results_75_25/'
SOL_CSV = os.path.join(SAVE_PATH,'best_sol.csv')

def analyze_sol():

    assert os.path.exists(SOL_CSV), f'No solution found in {SAVE_PATH}'

    coral_class = CRO('Practica_Sist_Tec_Teleco.mat',repeat_clients=False)
    sol_data = pd.read_csv(SOL_CSV, header=None)
    
    points_df, client_df = coral_class.get_dataframes()

    print(client_df)

    coded_sol = []
    total_clients = 0
    total_cost = 0
       
    for row in sol_data.iterrows():
        
        match = points_df.loc[(round(points_df['x'],3) == round(row[1][0],3)) & \
            (round(points_df['y'],3) == round(row[1][1],3))]

        coded_sol.append(match.index.values[0])
        
        total_clients += np.sum(match['clients_in_range'].values[0][0])
        total_cost += match['cost'].values[0]
    
    print(f'Best solution for {SAVE_PATH}: {sorted(coded_sol)}\n with {int(total_clients)} clients and {total_cost} cost')
    

if __name__ == "__main__":
    analyze_sol()