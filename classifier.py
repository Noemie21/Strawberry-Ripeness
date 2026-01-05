import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

np.random.seed(42)

#---- Cette classe est destinée au prétraitement sur les images ----#

class StrawberryDescriptor:
    """ Image Preprocessing and compute descriptors for k-means 
        clustering
    """
    def __init__(self, color_space: str='RGB', descriptor: str='mean'):
        self.color_space = color_space
        self.descriptor = descriptor
        
    def change_space(self, img):
        if self.color_space == 'HSV':
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            H, S, V = cv.split(img)  
            # Create mask to ignore 0 pixel-value
            mask = (H > 0) & (S > 0) & (V > 0)
            S, V = [(channel[mask] / 255.0).astype(np.float32) for channel in (S, V)]
            S = S / V # Normaliser par la valeur pour être robuste au changement de luminosité
            H = (H[mask] / 180.0).astype(np.float32)
            return H, S
        
        elif self.color_space == 'Lab':
            img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
            _, a, b = cv.split(img)  # a and b
            # Create mask to ignore 128 pixel-value at same time in a and b
            mask = (a != 128) & (b != 128)
            a, b = [(channel[mask] / 255.0).astype(np.float32) for channel in (a,b)]
            return a, b
        
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            R, G, _ = cv.split(img)
            mask = (R > 0) & (G > 0)
            R, G = [(channel[mask] / 255.0).astype(np.float32) for channel in (R,G)]
            return R, G

    def compute_descriptors(self, 
                            channel1: np.ndarray, 
                            channel2: np.ndarray,
                          ) -> np.ndarray:
        if self.descriptor == 'mean':
            mean_channel1 = channel1.mean()
            mean_channel2 = channel2.mean()
            return np.array([mean_channel1, mean_channel2], dtype=np.float64) 
            
        if self.descriptor == 'histogram2d':
            # Calcul du minimum et maximum global parmi les deux canaux
            min_val = np.min([np.min(channel1), np.min(channel2)])
            max_val = np.max([np.max(channel1), np.max(channel2)])
            hist = cv.calcHist([channel1, channel2], 
                               [0, 1],
                               None,
                               [4, 4],
                               [min_val, max_val+1, min_val, max_val+1])
            # Noramlise par le total de pixels valides
            hist = hist / hist.sum()
            return hist.astype(np.float64)


class StrawberryClassifier:
    def __init__(self, color_space='HSV', descriptor='mean', num_clusters=10):
        # Passer à StrawberryDescriptor les deux paramètres : color_space et descriptor
        self.color_descriptor = StrawberryDescriptor(color_space, descriptor)
        self.num_clusters = num_clusters
        self.kmeans = None
        self.scaler = None  # Ajouter un attribut pour le scaler

    def preprocess(self, image):
        # Utiliser le descripteur selon la configuration choisie
        channel1, channel2 = self.color_descriptor.change_space(image)
        descriptor = self.color_descriptor.compute_descriptors(channel1, channel2)
        #Si le descripteur est un tuple (cas de 'mean'), on le convertit en ndarray
        #if isinstance(descriptor, tuple):
            #descriptor = np.array(descriptor, dtype=np.float64)
        #print("Forme du descripteur :", descriptor.shape, "type: ", type(descriptor))
        return descriptor

    def train(self, images):
        descriptors = [self.preprocess(image) for image in images]
        # Si l'histogramme 2D est utilisé, aplatir les descripteurs 2D
        if self.color_descriptor.descriptor=='histogram2d':
            descriptors = [d.flatten() for d in descriptors]
        
        descriptors = np.array(descriptors)
    
        # Vérification de la forme des descripteurs avant de les normaliser
        # print(f"Forme des descripteurs avant normalisation: {descriptors.shape}")
    
        # Appliquer le StandardScaler pour normaliser les descripteurs
        self.scaler = StandardScaler()
        descriptors_scaled = self.scaler.fit_transform(descriptors)

        # KMeans clustering sur les descripteurs mis à l'échelle
        self.kmeans = KMeans(init="k-means++", n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(descriptors_scaled)
        return descriptors_scaled, self.kmeans.labels_, self.kmeans.cluster_centers_

    def test(self, image):
        if self.kmeans is None:
            raise ValueError("Le modèle n'a pas été entraîné. Veuillez entraîner le modèle d'abord.")
    
        descriptor = self.preprocess(image)
    
        # Normaliser le descripteur avant de le prédire
        descriptor_scaled = self.scaler.transform([descriptor])
    
        # Prédire le label de l'image
        label = self.kmeans.predict(descriptor_scaled)
        return label
    
    def evaluate(self, csv_file, images):
        if self.kmeans is None or self.scaler is None:
            raise ValueError("Le modèle n'a pas été chargé. Veuillez charger le modèle d'abord.")
        # Charger les labels réels à partir du fichier CSV
        # Chargement des vraies étiquettes depuis le fichier CSV
        df = pd.read_csv(csv_file)
        true_labels = df['label'].values

        descriptors = []
    
        # Extraire les descripteurs pour chaque image
        for img in images:
            desc = self.preprocess(img)
            if self.color_descriptor.descriptor == 'histogram2d':
                desc = desc.flatten()
            descriptors.append(desc)

        # Conversion en tableau numpy de type float64 (exigé par KMeans)
        descriptors = np.asarray(descriptors, dtype=np.float64)

        # Appliquer le scaler sur tous les descripteurs
        descriptors_scaled = self.scaler.transform(descriptors)
        descriptors_scaled = descriptors_scaled.astype(np.float64)
        # Prédire les labels des clusters
        predicted_labels = self.kmeans.predict(descriptors_scaled)

        # Calculer l'Adjusted Rand Index (ARI)
        ARI = adjusted_rand_score(true_labels, predicted_labels)

        return ARI

    
"""if __name__ == '__main__':
    img = cv.imread("./Images/s3375.png")
    import matplotlib.pyplot as plt
    plt.imshow(img[:,:,[2, 1, 0]])
    plt.show()
    processor = StrawberryDescriptor(color_space='RGB', descriptor='histogram2d')
    H, S = processor.change_space(img)
    descriptor = processor.compute_descriptors(H, S)
    import matplotlib.pyplot as plt
    plt.imshow(descriptor)
    plt.xlabel("R") 
    plt.ylabel("G") 
    plt.colorbar(plt.imshow(descriptor))
    plt.show()
    print(np.sum(descriptor))"""
            