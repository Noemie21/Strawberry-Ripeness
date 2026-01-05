import argparse
import os
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from classifier import StrawberryClassifier
from utils import evaluate_clustering, plot_kmeans_result, load_images_from_csv

def main():
    
    parser = argparse.ArgumentParser(description="Clustering des fraises")
    parser.add_argument('--color_space', type=str, choices=['Lab', 'HSV', 'RGB'], default='Lab', help="Espace colorimétrique à utiliser")
    parser.add_argument('--descriptor', type=str, choices=['mean', 'histogram2d'], default='mean', help="Descripteur à utiliser")
    args = parser.parse_args()
    
    color_space = args.color_space
    descriptor = args.descriptor
    
    #------------------DEBUT------------------#
    print("Bienvenue au clustering")
    print("Clustering Training and Evaluation.")
    print(f"Espace de couleur : {color_space}, descripteur : {descriptor}")
    random.seed(42)
    
    # Importer les images
    image_folder = "./Fraises"
    csv_path  = "./train_labels.csv"
    images, paths = load_images_from_csv(csv_path, image_folder)
    print(f"Total des images d'entrainement : {len(images)}.")
    
    #------------------TROUVE NOMBRE DE CLUSTER IDEAL ------------------#
    print("On cherche d'abord le meilleur nombre de clusters")
    # Évaluer le nombre optimal de clusters avec les paramètres choisis
    optimal_k_silhouette, optimal_k_inertia = evaluate_clustering(images, 
                                                                  color_space=color_space, 
                                                                  descriptor=descriptor, 
                                                                  max_clusters=10)
    
    
    print(f"Meilleur nombre de clusters (Silhouette): {optimal_k_silhouette}")
    print(f"Meilleur nombre de clusters (Inertie): {optimal_k_inertia}")
    
    
    num_clusters = 4
    
    #------------------ENTARINEMEMENT AVEC 4 CLUSTERS------------------#
    print("Entrainemenent avec 4 clusters correspondant aux labels réels")
    model = StrawberryClassifier(color_space=color_space, 
                                 descriptor=descriptor, 
                                 num_clusters=num_clusters)
    
    from scipy.spatial.distance import cdist

    # Train the model and get the descriptors, labels, and cluster centers
    descriptors, labels, centers = model.train(images)
    
    
    plot_kmeans_result(descriptors, labels, centers, color_space, descriptor)

    # Création du DataFrame avec les centres des clusters
    df = pd.DataFrame(centers, columns=[f"Feature {i+1}" for i in range(centers.shape[1])])
    df.insert(0, "Cluster Label", np.arange(len(centers)))  # Ajout des labels des clusters

    # Compter le nombre d'éléments par cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    df["Count"] = counts

    # Calcul de la distance euclidienne entre tous les centres des clusters
    distances = cdist(centers, centers, metric='euclidean')

    # Afficher la matrice des distances entre les centres
    #print("Matrice des distances Euclidiennes entre les centres des clusters :")
    #print(distances)

    # Ajouter les distances entre chaque paire de clusters au DataFrame
    # Chaque ligne contient les distances entre le centre du cluster et tous les autres clusters
    distance_df = pd.DataFrame(distances, columns=[f"Distance to Cluster {i}" for i in range(len(centers))])
    df = pd.concat([df, distance_df], axis=1)
    # Arrondir toutes les valeurs numériques à 3 décimales
    df = df.round(3)
    #------------------RESULTATS------------------#
    print("Dataframe des résultats et distances entre chaque cluster")
    print(df)
    df.to_csv(f"{color_space}-{descriptor}-Clustering.csv", index=False,float_format="%.3f")
    
    
    # Supprimer le dossier si il existe déjà
    dossier_sortie = os.path.join(f"{color_space}-{descriptor}-Clustering")
    if os.path.exists(dossier_sortie):
        shutil.rmtree(dossier_sortie)  # Supprimer le dossier et son contenu
    
    # Créer le dossier principal Lab-Clusters
    os.makedirs(dossier_sortie, exist_ok=True)
    
    # Créer des dossiers pour chaque cluster
    for cluster_num in range(num_clusters):
        cluster_dir = os.path.join(dossier_sortie, f'Cluster_{cluster_num}')
        os.makedirs(cluster_dir, exist_ok=True)

    # Sauvegarder chaque image dans le dossier correspondant
    for i, label in enumerate(labels):
        img = images[i]
        image_name = os.path.basename(paths[i])
        save_path = os.path.join(dossier_sortie, f'Cluster_{label}', image_name)
        cv.imwrite(save_path, img)
    
    # Déterminer le nombre de colonnes (max 10 images par cluster)
    n_cols = 10  
    n_rows = num_clusters  # Une ligne par cluster

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))

    # Si un seul cluster, convertir axes en tableau 2D
    if num_clusters == 1:
        axes = np.array([axes])

    for cluster_num in range(num_clusters):
        cluster_dir = os.path.join(dossier_sortie, f'Cluster_{cluster_num}')
        image_files = os.listdir(cluster_dir)
    
        # Sélectionner aléatoirement jusqu'à 10 images
        sample_images = random.sample(image_files, min(n_cols, len(image_files)))
    
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(cluster_dir, img_name)
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
            ax = axes[cluster_num, j]  # Sélectionner l'axe correspondant
            ax.imshow(img, aspect='auto')
            ax.axis('off')
            ax.set_title(img_name, fontsize=8)  # Afficher le nom sous l'image
            
        # Ajouter un titre unique pour chaque ligne (cluster)
        fig.text(0.02, 1 - (cluster_num + 0.5) / n_rows, f'Cluster {cluster_num}', 
             va='center', ha='left', fontsize=12, fontweight='bold')

    # Ajuster la mise en page et sauvegarder l'image finale
    plt.tight_layout(rect=[0.1, 0, 1, 1])  # Laisser de la place pour les titres à gauche
    plt.savefig(f"{color_space}-{descriptor}-Clustering.png")  # Enregistrer l'image
    #plt.show()
    
    
    #-------------------- EVALUATION ADJUST AREA INDEX -----------------#
    print("Évaluation avec la méthode ADJUST AREA INDEX (ARI)")
    # Importer les images
    image_folder = "./Fraises"
    csv_path  = "./test_labels.csv"
    images, paths = load_images_from_csv(csv_path, image_folder)
    print(f"Total des images d'entrainement : {len(images)}.")
    # Appeler la méthode evaluate pour afficher les images et calculer les métriques
    ARI = model.evaluate(csv_path, images)
    print(f"Adjusted Rand Index (ARI): {ARI:.3f}")
    
if __name__=="__main__":
    main()