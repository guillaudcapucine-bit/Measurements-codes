

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from points_depart import pt_depart



def analyse_angles_PCA(image_path, left_pt, right_pt, Nfit=10, show=True, save_path=None, crop_box=(100,600,530,800)):
    """
    Calcule les angles gauche et droit sur l'image recadrée à partir du contour principal.
    Utilise la régression orthogonale (PCA/SVD) pour estimer les tangentes locales aux deux bords.

    Paramètres :
    image_path : str
        Chemin de l'image.
    left_pt, right_pt : tuple(x, y) des bords gauche et droit.
    Nfit : int : Nombre de points du contour utilisés pour le fit local.
    show : bool : Affiche l'image annotée si True.
    save_path : str ou None : sauvegarde l'image annotée si défini.
    crop_box : tuple (x_min, x_max, y_min, y_max) pour recadrer l'image.

    angle_gauche, angle_droit : float, float
        Angles en degrés (dans le repère image).
    """

    # --- Lecture et recadrage ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
    x_min, x_max, y_min, y_max = crop_box
    img = img[y_min:y_max, x_min:x_max].copy()
    h, w = img.shape

    # --- Correction gamma ---
    #gamma = 0.5
   # table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
   # img = cv2.LUT(img, table)
    
    #Correction LUT pour le constraste
    # Définir la fonction
    def f(x):
        return 255 * (1 - np.exp(-0.2 * x))

    # Créer la LUT
    x = np.arange(256)
    LUT = np.clip(f(x), 0, 255).astype(np.uint8)

    # Appliquer la LUT
    img = cv2.LUT(img, LUT)

    #contour
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Aucun contour trouvé.")
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)
    N = pts.shape[0]

    def fit_angle(pt_ref, direction='left'):
        """Sous-fonction : calcule la tangente locale via PCA autour du point donné"""
        distances = np.linalg.norm(pts - np.array(pt_ref), axis=1)
        idx = np.argmin(distances)

        if direction == 'left':
            fit_indices = [(idx - i) % N for i in range(min(Nfit, N))]
        else:  # direction == 'right'
            fit_indices = [(idx + i) % N for i in range(min(Nfit, N))]

        fit_pts = np.unique(pts[fit_indices], axis=0)
        if fit_pts.shape[0] < 2:
            raise RuntimeError(f"Points insuffisants pour le fit sur {direction}")

        pts_float = fit_pts.astype(float)
        mean_pt = pts_float.mean(axis=0)
        pts_centered = pts_float - mean_pt
        _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
        vx, vy = Vt[0]

        # On oriente le vecteur pour que la tangente pointe vers le haut
        if vy > 0:
            vx, vy = -vx, -vy

        angle_deg = np.degrees(np.arctan2(vy, vx))
        return angle_deg, (vx, vy)

    #Calcul des deux angles
    angle_gauche, dir_g = fit_angle(left_pt, 'left')
    angle_droi, dir_d = fit_angle(right_pt, 'right')
    angle_droit=-180-angle_droi
    
    #Moyenne des deux angles
    angle_moyen = round((angle_gauche + angle_droit) / 2, 2)

    #Image annotée
    imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    # Affichage de la moyenne
    cv2.putText(imgc, f"Angle moyen = {-angle_moyen:.1f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    # Marqueurs
    cv2.circle(imgc, left_pt, 6, (0,0,255), -1)
    cv2.circle(imgc, right_pt, 6, (255,0,0), -1)
    cv2.line(imgc, left_pt, right_pt, (0,255,0), 1)

    # Tangentes
    L = 100
    for (pt, (vx, vy), angle, color) in [
        (left_pt, dir_g, angle_gauche, (255,0,255)),
        (right_pt, dir_d, angle_droit, (0,255,255))
    ]:
        x0, y0 = pt
        x1 = int(x0 + L * vx)
        y1 = int(y0 + L * vy)
        cv2.line(imgc, (x0, y0), (x1, y1), color, 2)
        cv2.putText(imgc, f"{angle:.1f}", (x0+10, y0-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if save_path is not None:
        cv2.imwrite(save_path, imgc)
    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB))
        plt.title(f"Angle gauche = {angle_gauche:.1f}°, Angle droit = {angle_droit:.1f}°")
        plt.axis('off')
        plt.show()

    return angle_gauche, angle_droit, angle_moyen





def natural_sort_key(s):
    """Trie les fichiers en ordre naturel (goutte_1, goutte_2, … goutte_10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]
