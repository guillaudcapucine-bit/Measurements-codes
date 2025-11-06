# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 14:11:34 2025

@author: Capucine
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def pt_depart(image_path, delta=13, persist=20, Nfit=40, show=False, save_path=None, offset=0):
    """Calcule l’angle gauche et affiche points gauche, droit et le point le plus bas à droite."""
    """"image_path : chemin de l’image à analyser.
    delta : différence verticale pour détecter un changement de pente.
    persist : nombre de points consécutifs pour confirmer un commencement de goutte.
    Nfit : nombre de points utilisés pour ajuster la tangente gauche.
    show : si True, affiche l’image annotée.
    save_path : si défini, sauvegarde l’image annotée."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    crop_box = (100, 600, 530, 800) 
    x_min, x_max, y_min, y_max = crop_box
    img = img[y_min:y_max, x_min:x_max]
    
    # correction gamma moins bien que la LUT
    #gamma= 0.3
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

    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
    h, w = img.shape
    "charge l'image en niveaux de gris, si elle existe + h et w sont les dimensionns de l'image"


    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    "THRESH_BINARY_INV : inverse les couleurs (goutte = blanc, fond = noir) "
    "THRESH_OTSU : calcule automatiquement le seuil optimal pour séparer le fond et la goutte."



    kernel = np.ones((3,3), np.uint8)
    "sorte de pinceau pour le nettoyage"
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1) 
    "supprime le bruit"
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1) 
    "ferme les petits trous"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    "on garde tout les points du contour (pas d'approximation"
    
    if not contours:
        raise RuntimeError("Aucun contour trouvé.")
    contour = max(contours, key=cv2.contourArea)
    "contour est un tableau contenant tout les points du contours, de forme (150,1,2)"
    pts = contour.reshape(-1, 2)
    "on fait un tableau plus simple, de forme (nombre_de_points, 2)."
    N = pts.shape[0]


    inside = (pts[:,0] > 0) & (pts[:,0] < w-1) & (pts[:,1] > 0) & (pts[:,1] < h-1)
    kept_idx = np.where(inside)[0]
    if kept_idx.size == 0:
        kept_idx = np.arange(N)

    ys_kept = pts[kept_idx, 1]
    y_max = int(ys_kept.max())
    candidates = kept_idx[ys_kept == y_max]
    start_idx = int(candidates[len(candidates)//2])

    #trouver la pente
    def walk_and_detect(start, direction, y_limit):
        "y_limit = référence verticale"
        idx = start
        base_pts = [start]
        consecutive = 0
        steps = 0
        while steps < N:
            idx = (idx + direction) % N
            y = int(pts[idx, 1])
            if y <= y_limit - delta:
                consecutive += 1
                if consecutive >= persist:
                    return min(base_pts, key=lambda i: pts[i,1])
            else:
                consecutive = 0
                base_pts.append(idx)
            steps += 1
        return min(base_pts, key=lambda i: pts[i,1])
    
    #point gauche de depart
    left_idx = walk_and_detect(start_idx, -1, y_max)
    left_idx = (left_idx +offset)    # recule un peu vers la goutte
    left_pt = tuple(map(int, pts[left_idx]))
    


    #point le plus bas, mais du coté droit 
    xs = pts[:,0]
    ys = pts[:,1]
    mask_right = (xs > left_pt[0]) & (ys > 0) & (ys < h-1) & (xs > 0) & (xs < w-1)
    
    if np.any(mask_right):
        ys_right = ys[mask_right]
        idx_candidates = np.where(mask_right)[0]
        rightmost_low_idx = idx_candidates[np.argmax(ys_right)]
     #   rightmost_low_pt = tuple(map(int, pts[rightmost_low_idx]))
    else:
        rightmost_low_idx = left_idx
       # rightmost_low_pt = left_pt

    #point droit de depart
    right_start_idx = walk_and_detect(rightmost_low_idx, 1, y_max)
    right_start_idx = (right_start_idx - offset)    # recule aussi vers la goutte
    right_start_pt = tuple(map(int, pts[right_start_idx]))
    
    #point le plus haut de la goutte, utile pour la PCA pour la determination de NFIT
    top_idx = np.argmin(pts[:, 1])  # indice du plus petit y
    top_pt = tuple(map(int, pts[top_idx]))  # coordonnées (x, y)

    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(img, cmap='gray')
        plt.scatter(*left_pt, color='red', label='Point gauche')
        plt.scatter(*top_pt, color='blue', s=60, label='Point le plus haut')
        plt.scatter(*right_start_pt, color='lime', label='Point droit')
        plt.scatter(pts[:,0], pts[:,1], s=2, c='yellow', label='Contour')
        plt.legend()
        plt.show()


    return left_pt, right_start_pt, top_pt

