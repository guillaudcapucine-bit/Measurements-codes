# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:52:02 2025

@author: Capucine
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os


def spline_goutte(img_path, n_points=300, spline_s=50.0,
                  save_tab_sep="\t", scale_factor=1000.0*10**(-6)/122.0,
                  show=True, save_txt=True, Nfit=10):
    """
    Extrait le contour supérieur d'une goutte et calcule une spline 1D y(x).
    Cherche les max de dérivée (angle) dans deux zones rectangulaires définies en dur.
    Affiche les deux rectangles, les deux angles (signés pour l'affichage de la tangente)
    et la moyenne des deux angles en valeur absolue.
    """

    # lecture et recadrage 
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image introuvable : " + img_path)

    crop_box = (100, 600, 530, 800)  # (x_min, x_max, y_min, y_max) -> adapte si besoin
    x_min, x_max, y_min, y_max = crop_box
    img = img[y_min:y_max, x_min:x_max]
    h, w = img.shape[:2]
    
    #correction gamma moins bien que la LUT
    #gamma = 0.1
    #table = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype("uint8")
    #img_eq = cv2.LUT(img, table)
    
    #Correction LUT pour le constraste
    # Définir la fonction
    def f(x):
        return 255 * (1 - np.exp(-0.2 * x))

    # Créer la LUT
    x = np.arange(256)
    LUT = np.clip(f(x), 0, 255).astype(np.uint8)

    # Appliquer la LUT
    img_eq = cv2.LUT(img, LUT)

    # --- Contour ---
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("Aucun contour trouvé.")
    contour = max(contours, key=cv2.contourArea)
    coords = contour[:, 0, :].astype(float)

    #origine en bas
    coords[:, 1] = (h - 1) - coords[:, 1]

    #filtrage hauteur
    mask = (coords[:, 1] > 10)
    coords_kept = coords[mask]
    if len(coords_kept) < 10:
        raise RuntimeError("Trop peu de points après filtrage.")

    #moyenne par x arrondi
    x_kept = coords_kept[:, 0]
    y_kept = coords_kept[:, 1]
    x_round = np.round(x_kept).astype(int)
    unique_x, inv = np.unique(x_round, return_inverse=True)
    y_mean = np.array([y_kept[inv == i].mean() for i in range(len(unique_x))])

    #tri
    order = np.argsort(unique_x)
    x_u = unique_x[order].astype(float)
    y_u = y_mean[order]
    valid = np.isfinite(x_u) & np.isfinite(y_u)
    x_u = x_u[valid]
    y_u = y_u[valid]

    #spline
    spl = UnivariateSpline(x_u, y_u, s=spline_s)
    x_new = np.linspace(x_u.min(), x_u.max(), n_points)
    y_new = spl(x_new)

    #dérivée de la spline
    spl_der = spl.derivative()
    dy_dx_all = spl_der(x_new)

    #RECTANGLES DE RECHERCHE
    rects = [
        (20, 200, 20, 100),   # gauche
        (300, 480, 20, 100)   # droite
    ]

    angles_abs = []        # angles en valeur absolue (pour la moyenne)
    angles_signed = []     # angles signés (pour le tracé et affichage signés)
    points_angle = []

    img_display = img_eq.copy()
    cv2.drawContours(img_display, [contour], -1, (0, 255, 0), 1)

    for i, (xmin_rect, xmax_rect, ymin_rect, ymax_rect) in enumerate(rects):
        # masque des points de la spline dans le rectangle
        mask_zone = (x_new >= xmin_rect) & (x_new <= xmax_rect) & \
                    (y_new >= ymin_rect) & (y_new <= ymax_rect)

        if np.any(mask_zone):
            dy_dx_sel = dy_dx_all[mask_zone]
            x_sel = x_new[mask_zone]
        else:
            print(f"[AVERTISSEMENT] Aucun point dans la zone {i+1}, max global utilisé.")
            dy_dx_sel = dy_dx_all
            x_sel = x_new

        # on prend le max à gauche (pente ascendante), la min à droite (pente descendante)
        if i == 0:
            idx_local = np.argmax(dy_dx_sel)
        else:
            idx_local = np.argmin(dy_dx_sel)

        dy_dx_val = dy_dx_sel[idx_local]
        x_val = x_sel[idx_local]
        y_val = spl(x_val)

        # angle signé (en degrés) : utile pour orienter la droite correctement et l'afficher avec signe
        angle_signed = np.degrees(np.arctan(dy_dx_val))
        # angle absolu pour la moyenne et si on veut l'afficher sans signe
        angle_abs = abs(angle_signed)

        angles_signed.append(angle_signed)
        angles_abs.append(angle_abs)
        points_angle.append((x_val, y_val))

        print(f"Angle signé (rectangle {i+1}) = {angle_signed:.2f}° ; |angle| = {angle_abs:.2f}°")

        #tracé du rectangle sur l'image
        color_rect = (0, 255, 255) if i == 0 else (255, 165, 0)  # jaune / orange
        cv2.rectangle(
            img_display,
            (int(xmin_rect), int(h - 1 - ymax_rect)),
            (int(xmax_rect), int(h - 1 - ymin_rect)),
            color_rect, 2
        )

        #tracé de la tangente
        line_length = 150
        half_len = line_length / 2
        angle_rad_signed = np.radians(angle_signed)

        x_center = int(round(x_val))
        y_center = int(round(h - 1 - y_val))

        x1 = int(round(x_center - half_len * np.cos(angle_rad_signed)))
        y1 = int(round(y_center + half_len * np.sin(angle_rad_signed)))
        x2 = int(round(x_center + half_len * np.cos(angle_rad_signed)))
        y2 = int(round(y_center - half_len * np.sin(angle_rad_signed)))

        color_line = (255, 0, 255) if i == 0 else (0, 0, 255)
        cv2.line(img_display, (x1, y1), (x2, y2), color_line, 2)

        # point central, et texte : on affiche l'angle signé (avec - si pente négative)
        cv2.circle(img_display, (x_center, y_center), 5, color_line, -1)
        cv2.putText(
            img_display, f"{angle_signed:+.1f}°",  # + force le signe affiché
            (x_center + 10, y_center - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_line, 2
        )

    #moyenne des deux angles
    angle_moyen_abs = float(np.mean(angles_abs))
    print(f"Angle moyen (valeurs absolues) = {angle_moyen_abs:.2f}°")

    # afficher la moyenne sur l'image (texte blanc)
    cv2.putText(
        img_display, f"Angle moyen = {angle_moyen_abs:.2f}°",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    #sauvegarde
    if save_txt:
        t_new = np.linspace(0.0, 1.0, len(x_new))
        out_x = "goutte_xf.txt"
        out_y = "goutte_yf.txt"
        np.savetxt(out_x, np.column_stack((t_new, x_new * scale_factor)),
                   delimiter=save_tab_sep, fmt="%.6f", header="t x", comments='')
        np.savetxt(out_y, np.column_stack((t_new, y_new * scale_factor)),
                   delimiter=save_tab_sep, fmt="%.6f", header="t y", comments='')
        print("Fichiers sauvegardés :", os.path.abspath(out_x), os.path.abspath(out_y))

    #image+dérivée
    if show:
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(img_path))
        plt.axis('off')
        plt.show()

        #tracé de la dérivée
        plt.figure(figsize=(6, 4))
        plt.plot(x_new, dy_dx_all, lw=1.8, label="dérivée spline")
        # tracer les verticales aux positions détectées
        for j, (x_val, _) in enumerate(points_angle):
            plt.axvline(x=x_val, color="crimson" if j == 0 else "orange",
                        linestyle="--", label=f"point {j+1}")
        plt.xlabel("x (pixels)")
        plt.ylabel("dy/dx")
        plt.title("Dérivée de la spline le long du contour")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return x_new * scale_factor, y_new * scale_factor,  angle_moyen_abs
