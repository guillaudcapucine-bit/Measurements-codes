
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 10:50:48 2025

@author: Capucine
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


#Dictionnaire des volumes
volumes_dict = {1: 4.46, 2: 11.79, 3: 9.41, 4: 1.39, 5: 0.92, 6: 13.95}

#Chemin du dossier contenant les fichiers CSV
base_path = r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\retour_polyfit"

#stocker les données
B_all, V_all, theta_all = [], [], []

#boucle sur les 6 fichiers test
for i in range(1, 7):
    filepath = os.path.join(base_path, f"B_theta_resultats{i}.csv")
    print(f"Lecture du fichier {i} :", filepath)
    
    # Lecture CSV en gardant seulement les colonnes B et theta+ignorer les lignes mal formées
    df = pd.read_csv(filepath, sep=";", usecols=["B", "theta"], on_bad_lines='skip')
    
    # Remplacer la virgule par un point et convertir en numérique
    df["B"] = pd.to_numeric(df["B"].astype(str).str.replace(",", "."), errors='coerce')
    df["theta"] = pd.to_numeric(df["theta"].astype(str).str.replace(",", "."), errors='coerce')
    
    # Supprimer les lignes où B ou theta sont NaN
    df = df.dropna(subset=["B", "theta"])
    print(f"Nombre de points valides dans le fichier {i} :", len(df))
    
    # Ajouter les données aux listes globales
    B_all.extend(df["B"])
    theta_all.extend(df["theta"])
    V_all.extend([volumes_dict[i]] * len(df))

#Tracé carte 3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

# Nuage de points coloré selon le volume
sc = ax.scatter(B_all, V_all, theta_all, c=V_all, cmap="viridis", s=40)
    


# Labels et titre
ax.set_xlabel("B (T)")
ax.set_ylabel("Volume (uL)")
ax.set_zlabel("Theta (°)")
ax.set_title("Nuage 3D : B, Volume, Theta")



#vue initiale
ax.view_init(elev=30, azim=45)  # changer elev et azim pour tourner la vue au départ
#légende pour les 6 volumes 
cmap = cm.get_cmap("viridis")
volumes = sorted(list(set(V_all)))
handles = [mpatches.Patch(color=cmap((vol - min(volumes)) / (max(volumes)-min(volumes))),
                          label=f"V={vol} uL") for vol in volumes]
ax.legend(handles=handles)

plt.tight_layout()
plt.show()


#Projection 2D de B et theta
projections = [
    ("B (T)", "Theta (°)", B_all, theta_all, "B vs Theta")]


for xlabel, ylabel, xdata, ydata, title in projections:
    plt.figure(figsize=(6,5))
    for vol in volumes:
        idx = [j for j, v in enumerate(V_all) if v == vol]
        plt.scatter([xdata[j] for j in idx],
                    [ydata[j] for j in idx],
                    color=cmap((vol - min(volumes)) / (max(volumes)-min(volumes))),
                    s=40,
                    label=f"V={vol} uL")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
#V et B
cmap = cm.get_cmap("viridis")
B_values = sorted(list(set(B_all)))  # toutes les valeurs uniques de B
B_values = [b for b in B_values if b >= 580]
plt.figure(figsize=(6,5))

for B in B_values:
    # Indices correspondant à ce B
    idx = [j for j, b in enumerate(B_all) if b == B]
    
    plt.scatter([V_all[j] for j in idx],   # x = Volume
                [theta_all[j] for j in idx],  # y = Theta
                color=cmap((B - min(B_values)) / (max(B_values)-min(B_values))),
                s=40,
                label=f"B={B} T")

plt.xlabel("Volume (uL)")
plt.ylabel("Theta (°)")
plt.title("Theta vs Volume, couleur selon B")
plt.legend()
plt.tight_layout()
plt.show()