

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 2025

@author: Capucine
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf
from sklearn.preprocessing import StandardScaler


#volumes
volumes_dict = {1: 4.507, 2: 11.74, 3: 9.45, 4: 1.36, 5: 0.92, 6: 13.93}
base_path = r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\retour_polyfit"

#stocker les données
B_all, V_all, theta_all = [], [], []


for i in range(1, 7):
    # Lecture CSV en gardant seulement B et theta, lignes mal formées ignorées
    filepath = os.path.join(base_path, f"B_theta_resultats{i}.csv")#créé le filepath
    df = pd.read_csv(filepath, sep=";", usecols=["B", "theta"], on_bad_lines='skip')#ouvre le csv
    
    # Remplacer la virgule par un point et convertir en numérique
    df["B"] = pd.to_numeric(df["B"].astype(str).str.replace(",", "."), errors='coerce')
    df["theta"] = pd.to_numeric(df["theta"].astype(str).str.replace(",", "."), errors='coerce') 
    
    # Supprimer les lignes où B ou theta sont NaN
    df = df.dropna(subset=["B", "theta"])
    
    # Ajouter les données aux listes globales
    B_all.extend(df["B"])
    theta_all.extend(df["theta"])
    V_all.extend([volumes_dict[i]] * len(df))

#Convertit les listes en numpy arrays pour l'interpolation
B = np.array(B_all)
V = np.array(V_all)
theta = np.array(theta_all)


#normalisation des variables d'entrée
scaler = StandardScaler()
BV_scaled = scaler.fit_transform(np.column_stack((B, V)))
B_scaled, V_scaled = BV_scaled[:, 0], BV_scaled[:, 1]

#RBF : interpolation lissée
rbf = Rbf(B_scaled, V_scaled, theta,
          function='multiquadric',   
          epsilon=0.1,               # plus petit = plus proche des points et 1.0 =plus lisse
          smooth=0.5)               

#Grille
B_lin = np.linspace(B_scaled.min(), B_scaled.max(), 100)
V_lin = np.linspace(V_scaled.min(), V_scaled.max(), 100)
#génère 100 valeurs régulièrement espacées entre la valeur minimale et maximale
B_grid, V_grid = np.meshgrid(B_lin, V_lin)
#produit deux matrices 2D (B_grid, V_grid) représentant tous les couples (B,V) de la grille 100×100
theta_grid = rbf(B_grid, V_grid)
#évalue l’interpolant RBF sur toute la grille (résultat 100×100 de theta estimé).
#la grille est en variables normalisées (comme la construction de rbf)

#Revenir à l’échelle réelle pour affichage
B_plot = np.linspace(B.min(), B.max(), 100)
V_plot = np.linspace(V.min(), V.max(), 100)
B_grid_plot, V_grid_plot = np.meshgrid(B_plot, V_plot)

#affichage
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter points expérimentaux
cmap = cm.get_cmap("viridis")
volumes = sorted(list(set(V_all)))
handles = [mpatches.Patch(color=cmap((vol - min(volumes)) / (max(volumes)-min(volumes))),
                          label=f"V={vol:.2f} uL") for vol in volumes]
ax.legend(handles=handles)

ax.scatter(B, V, theta, c=V, cmap='viridis', s=40)
ax.plot_surface(B_grid_plot, V_grid_plot, theta_grid, cmap='viridis', alpha=0.65)

ax.set_xlabel("B (T)")
ax.set_ylabel("Volume (uL)")
ax.set_zlabel("Theta (°)")
ax.set_title("Surface RBF lissée (normalisation + ajustement epsilon)")
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()


# Valeurs de volume pour les coupes (en cm³)
B_coupes = [100, 200,300]
V_lin = np.linspace(V.min(), V.max(), 200)

plt.figure(figsize=(8,6))
for B_const in B_coupes:
    # Créer les couples (B, V_const)
    BV = np.column_stack((V_lin, np.full_like(V_lin, B_const)))
    BV_scaled = scaler.transform(BV)
    
    # Évaluer la surface RBF
    theta_pred = rbf(BV_scaled[:,0], BV_scaled[:,1])
    
    plt.plot(V_lin, theta_pred, label=f"V = {B_const} mT")

plt.xlabel("B (T)")
plt.ylabel("θ (°)")
plt.title("Coupes 2D : θ = f(B) pour différents volumes")
plt.legend()
plt.grid(True)
plt.show()