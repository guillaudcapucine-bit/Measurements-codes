# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:19:58 2025

@author: Capucine

Permet de tracer Rg/2a avec les csv importés de ImageJ (mesures faites manuellement)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob



#lecture des fichiers de B
B = pd.read_csv(r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\20251003 POST\Rosensweig\B_values.csv")["B"].values
nB = len(B)  # nombre de valeurs de B (normalement 25)

# fichiers de test
fichiers = sorted(glob.glob(r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\20251003 POST\Rosensweig\Test*.csv"))
labels = ["4.46uL", "11.79uL", "9.41uL", "1.89uL", "0.92uL", "13.95uL"]

plt.figure(figsize=(8,5))

#boucle sur les csv
for path, label in zip(fichiers, labels):
    df = pd.read_csv(path, sep=",")
    values = df["Length"].values   # ou "Angle" selon ton besoin

    # Supprimer la première ligne s’il y a un en-tête ou un index bizarre
    if not np.issubdtype(values.dtype, np.number):
        values = values[1:]  # au cas où la première ligne serait texte

    # Tronquer pour un multiple de 3 (au cas ou)
    n = len(values) - (len(values) % 3)
    values = values[:n]

    # Reshape (3 colonnes : R1, R2, a)
    grouped = values.reshape(-1, 3)
    R1, R2, a = grouped[:, 0], grouped[:, 1], grouped[:, 2]

    # Calcul de (R1 + R2)/2a
    #Y = (R1*R2*4) /(3.1421*a*a) aire du carré de RG/aire du cercle de la goutte
    Y=(R1+R2)/(2*a)

    #compléter avec 0 quand ya plus d'instabilités
    if len(Y) < nB:
        Y = np.pad(Y, (0, nB - len(Y)), constant_values=0)

    # Tracé

    plt.plot(B, Y, marker='o', label=label)

#plot

plt.xlabel("B (mT)")
plt.ylabel("(R1 + R2) / 2a")
plt.title("Rg diameter vs 2a")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

