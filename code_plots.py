
"""
Created on Thu Oct  9 15:02:49 2025

@author: Capucine
"""

"""
code pour plots toutes les courbes theta(B) en se servant des deux différentes méthodes.
soit angles et méthode PCA
soit spline_goutte et methode maximum de la dérivée
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from methodemaxder2 import spline_goutte
from points_depart import pt_depart
from Angle_PCA import analyse_angles_PCA
import re
import itertools
from scipy.stats import linregress



def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]




def analyse_dossier2(
    folder_path,
    output_folder,
    n_points=300,
    spline_s=30.0,
    delta=10,
    persist=20,
    Nfit=10,
    scale_factor=1000.0 * 10**(-6) / 122.0,
    fichier_B=None
):
    """
    Analyse toutes les images d’un dossier, remplit la colonne 'theta' dans le fichier CSV (BG, B, theta),
    et trace la courbe theta (°) en fonction de B (mT).
    """
    import matplotlib.pyplot as plt

    # --- Vérif et préparation ---
    if fichier_B is None:
        raise ValueError("Il faut fournir un fichier CSV avec les colonnes BG, B, theta.")
    os.makedirs(output_folder, exist_ok=True)

    # --- Charger le fichier CSV existant ---
    df = pd.read_csv(fichier_B, sep=";", decimal=",")
    if "theta" not in df.columns:
        df["theta"] = np.nan

    files = sorted(os.listdir(folder_path), key=natural_sort_key)
    image_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))]

    if len(image_files) != len(df):
        print(f"[AVERTISSEMENT] {len(image_files)} images trouvées mais {len(df)} lignes dans le CSV.")

    # --- Boucle sur les images ---
    for i, file in enumerate(image_files):
        path = os.path.join(folder_path, file)
        try:
            
            _, _, angle_spline = spline_goutte(
                path,
                n_points=n_points,
                spline_s=spline_s,
                show=True,
                Nfit=Nfit,
                scale_factor=scale_factor)
           
          
            #left, right, _ = pt_depart(path, delta=10, persist=40, Nfit=40, show=False)
            #angle=analyse_angle(path, left, right, Nfit=20, show=True, save_path=None)
            df.loc[i, "theta"] = angle_spline
            
            print(f"[OK] {file} -> θ = {angle_spline:.2f}°")

        except Exception as e:
            print(f"[ERREUR] {file} : {e}")
            df.loc[i, "theta"] = np.nan

    # --- Sauvegarde du CSV mis à jour ---
    out_csv = os.path.join(output_folder, "B_theta_resultats6.csv")
    df.to_csv(out_csv, sep=";", decimal=",", index=False)
    print(f"Fichier CSV mis à jour sauvegardé : {out_csv}")

    # --- Tracé θ en fonction de B ---
    plt.figure(figsize=(7, 5))
    plt.plot(df["B"], df["theta"], "o-", color="royalblue", markersize=5)
    plt.xlabel("Champ magnétique B (mT)")
    plt.ylabel("Angle θ (°)")
    plt.title("θ(B) TEST 6")
    plt.grid(True)
    plt.tight_layout()

    # --- Sauvegarde du graphique ---
    out_fig = os.path.join(output_folder, "theta_vs_B_1.png")
    plt.savefig(out_fig, dpi=300)
    plt.show()
    print(f"Graphique sauvegardé : {out_fig}")
    
def comparer_tests(liste_csv,dots, labels=None, titre="Comparaison des test θ(B)", save_path=None):
    """
    Trace plusieurs courbes θ(B) à partir de plusieurs fichiers CSV.
    
    Paramètres :
        - liste_csv : liste de chemins vers les CSV (issus de analyse_dossier2)
        - labels : liste de noms à afficher dans la légende (ex: ["Test 1", "Test 2", ...])
        - titre : titre du graphique
        - save_path : chemin complet du fichier image à sauvegarder (facultatif)
    """

    plt.figure(figsize=(8, 6))
    

    for i, (fichier, dot) in enumerate(zip(liste_csv, itertools.cycle(dots))):
        try:
            df = pd.read_csv(fichier, sep=";", decimal=",")
            label = labels[i] if labels and i < len(labels) else os.path.basename(fichier)
    
            # Définir le seuil
            B_min = 0  # par exemple, on ne garde que B >= 580 mT
    
            # Filtrer le DataFrame et supprimer les NaN
            dft = df[df["B"] >= B_min].dropna(subset=["theta"])
            B2 = dft["B"]
    
            if len(B2) < 2:
                print(f"[ATTENTION] Pas assez de points pour le fit pour {fichier}")
                continue
    
            # Tracé des données
            plt.plot(B2, dft["theta"], marker=dot, lw=1.8, label=label)
    
            # Fit linéaire avec p-value
           # slope, intercept, r_value, p_value, std_err = linregress(B2, dft["theta"])
           # r=r_value**2
            # Tracé du fit
           # B_fit = np.linspace(B2.min(), B2.max(), 100)
           # theta_fit = slope * B_fit + intercept
           # plt.plot(B_fit, theta_fit, "--", lw=1.5, color='gray', alpha=0.7)
    
        except Exception as e:
            print(f"[ERREUR] Impossible de lire {fichier} : {e}")


    plt.xlabel("Champ magnétique B(mT)")
    plt.ylabel("Angle θ (°)")
    plt.title(titre)
    plt.legend(
        loc='right',
        bbox_to_anchor=(1.14, 0.80),
        ncol= 1,
        frameon=True      # optionnel : pas de cadre autour de la légende
    )
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"→ Graphique sauvegardé dans : {save_path}")

    plt.show()

def analyse_dossier2_angle(
    folder_path,
    output_folder,
    fichier_B,
    n_points=300,
    delta=10,
    persist=20,
    scale_factor=1000.0 * 10**(-6) / 122.0
):
    """
    Analyse toutes les images d’un dossier avec la méthode analyse_angle()
    (basée sur les points trouvés par pt_depart),
    puis met à jour le CSV et trace la courbe θ(B).

    Paramètres :
    ------------
    folder_path : dossier contenant les images (ordre du dossier conservé)
    output_folder : dossier de sortie
    fichier_B : CSV avec colonnes BG, B, theta
    Nfit : nombre de points pour le fit de la tangente gauche
    """

    # --- Vérif et préparation ---
    if fichier_B is None:
        raise ValueError("Il faut fournir un fichier CSV avec les colonnes BG, B, theta.")
    os.makedirs(output_folder, exist_ok=True)

    # --- Charger le fichier CSV existant ---
    df = pd.read_csv(fichier_B, sep=";", decimal=",")
    if "theta" not in df.columns:
        df["theta"] = np.nan

    # --- Récupération des fichiers dans l’ordre du dossier ---
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))]
    image_files = sorted(image_files, key=natural_sort_key)
    if len(image_files) != len(df):
        print(f"[AVERTISSEMENT] {len(image_files)} images trouvées mais {len(df)} lignes dans le CSV.")

    #Boucle principale 
    for i, file in enumerate(image_files):
        path = os.path.join(folder_path, file)
        try:
            #Détermination des points gauche et droit
            left_pt, right_pt, top_pt = pt_depart(path, delta=delta, persist=persist, Nfit=10, show=False)
            hauteur_goutte=right_pt[1]


            #Calcul de l’angle avec la méthode tangente gauche
            _,_, angle_deg = analyse_angles_PCA(path, left_pt, right_pt, Nfit=20, show=True, save_path=None)

            # Mise à jour du tableau
            df.loc[i, "theta"] = -angle_deg
            print(f"[OK] {file} -> θ = {angle_deg:.2f}°")

        except Exception as e:
            print(f"[ERREUR] {file} : {e}")
            df.loc[i, "theta"] = np.nan

    #Sauvegarde du CSV mis à jour 
    out_csv = os.path.join(output_folder, "B_theta_resultats5.csv")
    df.to_csv(out_csv, sep=";", decimal=",", index=False)
    print(f"Fichier CSV mis à jour sauvegardé : {out_csv}")

    # --- Tracé θ en fonction de B ---
    plt.figure(figsize=(7, 5))
    plt.plot(df["B"], df["theta"], "o-", color="royalblue", markersize=5)
    plt.xlabel("Champ magnétique B (mT)")
    
    plt.ylabel("Angle θ (°)")
    plt.title("θ(B) – méthode polyfit")
    plt.grid(True)
    plt.tight_layout()

    out_fig = os.path.join(output_folder, "theta_vs_B_anglem21.png")
    plt.savefig(out_fig, dpi=300)
    plt.show()
    print(f"Graphique sauvegardé : {out_fig}")
    

# =======================
# Exemple d’appel :
# =======================

#pour tracer chaque courbe individuellement, quand c'est avec la methodes de PCA
'''
analyse_dossier2_angle(
    folder_path=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\Test 5 POST\Field increase\Silhouette",
    output_folder=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime",
    fichier_B=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Python\Book2.csv",
    delta=10,
    persist=40
)


'''

#pour mettre toutes les courbes sur le meme plot
comparer_tests(
    liste_csv=[
        r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime\B_theta_resultats1.csv",
        r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime\B_theta_resultats2.csv",
        r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime\B_theta_resultats3.csv",
        r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime\B_theta_resultats4.csv",
        r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime\B_theta_resultats5.csv",
        r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\complete regime\B_theta_resultats6.csv",
    ],dots = ['o', 's', 'x', '^', 'd', 'v'],
    labels=["4.46μL", "11.79μL", "9.41μL", "1.39μL", "0.92μL", "13.95μL"],
    titre="theta(B) - méthode:PCA - retour ",
    save_path=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\comparaison_testB.png"
)
#²

#pour tracer chaque courbe individuellement, quand cest avec la methode de la maximum de la derivee
'''
# Exemple d'appel
analyse_dossier2(
    folder_path=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\Test 1 POST\Retour\Silhouette",
    output_folder=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Experiences\20251003 POST\pasmoyLUTR",
    fichier_B=r"E:\Documents\cours\ENSE3\2A\SIMAP - Stage\Python\B_retour.csv")
'''