import os
from classifier import StrawberryClassifier

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import pandas as pd

np.random.seed(42)

def load_images_from_csv(csv_path, image_folder):
    df = pd.read_csv(csv_path)
    images = []
    paths = []
    for fname in df['image']:
        img_path = os.path.join(image_folder, fname)
        img = cv.imread(img_path)
        if img is not None:
            images.append(img)
            paths.append(img_path)
    return images, paths

def plot_results(k_range, 
                 silhouette_scores, 
                 inertias,
                 color_space='RGB', 
                 descriptor='mean'):
    # Courbe du score de silhouette
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title("Evolution du Score de Silhouette")
    plt.xlabel("Nombre de Clusters")
    plt.ylabel("Score de Silhouette")
    plt.xticks(np.arange(1, 11, 1))  # Afficher de 1 à 10 avec un pas de 1


    # Courbe de l'Elbow (Inertie)
    plt.subplot(1, 2, 2)
    plt.plot(k_range, inertias, marker='o')
    plt.title("Courbe de l'Elbow (Inertie)")
    plt.xlabel("Nombre de Clusters")
    plt.ylabel("Inertie (SSE)")
    plt.xticks(np.arange(1, 11, 1))  # Afficher de 1 à 10 avec un pas de 1
    plt.tight_layout()
    plt.savefig(f"Choosing-k-values-in-{color_space}-with-{descriptor}-values'.png")
    #plt.show()

def plot_kmeans_result(descriptors_pca, 
                       labels, 
                       centers,
                       color_space='RGB', 
                       descriptor='mean'):
    """Plot the PCA reduced original descriptors"""
    
    # Mapping des noms d’axes en fonction de l’espace de couleur
    axis_labels = {
        'RGB': ('R', 'G'),
        'HSV': ('H', 'S'),
        'Lab': ('a', 'b')
    }
    
    xlabel, ylabel = axis_labels.get(color_space, ('PC1', 'PC2'))  # fallback si couleur inconnue

    plt.figure(figsize=(8, 8))
    plt.scatter(descriptors_pca[:, 0], descriptors_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)  
    
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], s=300, c='black', marker='o')
        plt.text(center[0], center[1], str(i), color='red', fontsize=12, ha='center', va='center', fontweight='bold')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Clustering results in {color_space} with {descriptor} values')
    plt.savefig(f"Clustering-in-{color_space}-with-{descriptor}-values.png")
    
def plot_descriptors(descriptors,
                     color_space='RGB', 
                     descriptor='mean'):
    """Plot the 2D descriptors (e.g. PCA reduced)"""
    
    # Définition des noms d’axes en fonction de l’espace de couleur
    axis_labels = {
        'RGB': ('R', 'G'),
        'HSV': ('H', 'S'),
        'Lab': ('a', 'b')
    }

    xlabel, ylabel = axis_labels.get(color_space, ('Principal Component 1', 'Principal Component 2'))

    plt.figure(figsize=(4, 4))
    plt.scatter(descriptors[:, 0], descriptors[:, 1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('2D Visualization of Image Descriptors without Clustering')
    plt.savefig(f'Initial-in-{color_space}-with-{descriptor}-values.png')
    #plt.show()

def evaluate_clustering(images, color_space='HSV', descriptor='mean', max_clusters:int=10):
    classifier = StrawberryClassifier(color_space=color_space, descriptor=descriptor)
    descriptors = [classifier.preprocess(image) for image in images]
    
    # Si l'histogramme 2D est utilisé, aplatir les descripteurs 2D
    if descriptor =='histogram2d':
        descriptors = [d.flatten() for d in descriptors]
    descriptors = np.array(descriptors)
    
    # Appliquer le StandardScaler pour normaliser les descripteurs
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)

    silhouette_scores = []
    inertias = []
    k_range = range(2, max_clusters+1)

    for k in k_range:
        classifier.kmeans = KMeans(init="k-means++", 
                                   n_clusters=k, 
                                   random_state=42)
        
        classifier.kmeans.fit(descriptors_scaled)
        
        #print(descriptors_scaled)
        
        silhouette_scores.append(silhouette_score(descriptors_scaled, classifier.kmeans.labels_))
        inertias.append(classifier.kmeans.inertia_)
    
    
    # Trouver l'index correspondant à k=4 (pour plus de clarté et flexibilité)
    k_value = 4
    index_for_k = k_value - 2  # Puisque la plage commence à 2

    print(
        f"Pour {k_value} clusters."
        f"Score de silhouette moyen: {silhouette_scores[index_for_k]:.3f},."
        f"Inertie: {inertias[index_for_k]:.3f}"
    )
    plot_results(k_range, silhouette_scores, inertias, color_space, descriptor)
    plot_descriptors(descriptors_scaled,color_space, descriptor)
    
    # Déterminer le k optimal basé sur les scores de silhouette et d'inertie
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_inertia = k_range[np.argmin(inertias)]  # Inertia doit être minimisée

    return optimal_k_silhouette, optimal_k_inertia
