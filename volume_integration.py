
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 13 12:01:11 2025
@author: Capucine
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from points_depart import pt_depart
from scipy.integrate import quad
import os 
import re


def volume(img_path, 
           n_points=300, 
           spline_s=50.0,
           scale_factor=3.8e-3 / 308.002, 
           show=True):
    """
    Extrait le contour supérieur d'une goutte et calcule une spline 1D y(x),
    puis le volume par rotation autour de l’axe vertical.
    """

    #lecture dimage+recadrage
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image introuvable : " + img_path)
        

    crop_box = (100, 600, 530, 800)  # (x_min, x_max, y_min, y_max)
    #pour avoir seulement la silhouette de la goutte et pas la top view
    x_min, x_max, y_min, y_max = crop_box
    img = img[y_min:y_max, x_min:x_max]
    h, w = img.shape[:2]
    #parametre de l'image

    #correction gamma pour améliorer le contraste
    gamma = 0.3
    table = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype("uint8")
    img_eq = cv2.LUT(img, table)

    #conversion en niveau de gris et seuillage
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    
    
    #on extrait le contour principal
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Aucun contour trouvé.")
    contour = max(contours, key=cv2.contourArea)
    #plus grand countour = la goutte
    coords = contour[:, 0, :].astype(float)

    #origine en bas
    coords[:, 1] = (h - 1) - coords[:, 1]

    #point gauche et droits 
    left, right, _ = pt_depart(img_path, delta=10, persist=40, Nfit=40, show=False)
    left1 = (h - 1) - left[1]
    right1 = (h - 1) - right[1]

    #filtrage hauteur pour pas avoir les points du cadre en bas 
    mask = (coords[:, 1] > 10)
    coords_kept = coords[mask]
    if len(coords_kept) < 10:
        raise RuntimeError("Trop peu de points après filtrage.")

    #moyenne par arrondi, filtrage des points pour eviter les doublons 
    x_kept = coords_kept[:, 0]
    y_kept = coords_kept[:, 1]
    x_round = np.round(x_kept).astype(int)
    unique_x, inv = np.unique(x_round, return_inverse=True)
    y_mean = np.array([y_kept[inv == i].mean() for i in range(len(unique_x))])

    #tri
    order = np.argsort(unique_x)
    x_u = unique_x[order].astype(float)
    y_u = y_mean[order]

    #création de la spline 
    spl = UnivariateSpline(x_u, y_u, s=spline_s)
    x_new = np.linspace(x_u.min(), x_u.max(), n_points)
    y_new = spl(x_new)

    #on décale la spline a la base (y=0)
    y_base = min(right1, left1)
    y_shifted = y_new - y_base
    y_shifted[y_shifted < 0] = 0
    
    #centre de la goutte (axe de symetrie autour duquel la gouette tourne)
    x_center=(left[0]+right[0])/2

    x_left = left[0]
    x_right = right[0]
    
    #definition de la fonciotn du volume pour mettre l'integrale apres
    f = lambda x: (spl(x) - y_base) * abs(x - x_center)
    V_int = quad(f, x_left, x_right)
    volume_m3 = V_int[0] * np.pi * scale_factor**3

    #plots
    if show:
        plt.figure(figsize=(7, 7))
        plt.imshow(img, extent=[0, w, 0, h])
        plt.plot(x_u, y_u, 'o', markersize=3, label='Points extraits')
        plt.plot(x_new, y_shifted, '-', color='red', label='Spline')
        plt.plot(left[0], left1, 'go', label='Point gauche')
        plt.plot(right[0], right1, 'bo', label='Point droit')
        plt.title(f"{os.path.basename(img_path)}")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels, origine en bas)")
        plt.text(20, h * 0.8, f"Volume = {volume_m3:.2e} m³", color='yellow', fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.5))
        plt.legend()
        #plt.show()

    return volume_m3

#pour trier les fichiers dans le bon ordre si besoin

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]




