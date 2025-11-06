# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:48:30 2025

@author: Capucine

Calcul des plots theta(V)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

folder_path = r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime"
volumes_dict = {1:4.507, 2:11.74, 3:9.45, 4:1.36, 5:0.92, 6:13.93}
#volumes_dict = {3:4.507, 4:11.74, 5:9.45}
# Récupérer toutes les valeurs uniques de B dans les fichiers CSV
all_B_values = set()
for i in range(1, 7):
    file_path = os.path.join(folder_path, f"B_theta_resultats{i}.csv")
    df = pd.read_csv(file_path, sep=None, engine='python')
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or col == 'BG'], errors='ignore')
    df['B'] = df['B'].astype(str).str.replace(',', '.').astype(float)
    all_B_values.update(df['B'].dropna().unique())

#all_B_values = sorted(all_B_values)
B_seuil = 580
all_B_values = sorted([B for B in all_B_values if B > B_seuil])

plt.figure(figsize=(10,7))
colors = plt.cm.viridis(np.linspace(0, 1, len(all_B_values)))  #palette auto

for B_val, color in zip(all_B_values, colors):
    angles = []
    valid_volumes = []
    
    for i in range(1, 7):
        file_path = os.path.join(folder_path, f"B_theta_resultats{i}.csv")
        df = pd.read_csv(file_path, sep=None, engine='python')
        df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or col == 'BG'], errors='ignore')
        df['B'] = df['B'].astype(str).str.replace(',', '.').astype(float)
        df['theta'] = df['theta'].astype(str).str.replace(',', '.').astype(float)
        
        df_B = df[['B','theta']].dropna()
        if not df_B.empty:
            idx = (df_B['B'] - B_val).abs().idxmin()
            theta_value = df_B.loc[idx, 'theta']
            angles.append(theta_value)
            valid_volumes.append(volumes_dict[i])
        else:
            print(f"Aucune valeur de B proche de {B_val} dans le fichier {file_path}")
    
    if angles:  # seulement si on a des données valides
        valid_volumes = np.array(valid_volumes)
        angles = np.array(angles)
        order = np.argsort(valid_volumes)
        valid_volumes = valid_volumes[order]
        angles = angles[order]
        
        plt.plot(valid_volumes, angles, linestyle='--', color=color, label=f"B = {B_val:.2f} mT")
        plt.scatter(valid_volumes, angles, color=color)

plt.xlabel("Volume (µL)")
#plt.yscale('log')
#plt.xscale('log' )
plt.ylabel("Angle de contact θ (°)")
plt.title("theta(V) pour 580mT<B<1400mT")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.00,1), loc='upper left')  # legende à côté
plt.tight_layout()
plt.savefig("angles_toutes_B.png", dpi=300)
plt.show()
