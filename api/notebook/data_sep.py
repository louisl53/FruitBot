import os
import shutil
import numpy as np
import kaggle

try:
    # Create the folder to contain the dataset
    dossier_cible = 'api/notebook/dataset'
    os.makedirs(dossier_cible, exist_ok=True)

    # Authenticate and download the dataset from Kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten', path='api/notebook/dataset', unzip=True)

    # Path to the folder containing the fruit dataset directories
    chemin_dataset = 'api/notebook/dataset/Fruit And Vegetable Diseases Dataset'

    # Ensure that the dataset folder exists
    if not os.path.exists(chemin_dataset):
        raise Exception("The dataset folder was not found after download and extraction.")

    # Set the seed for reproducibility of random operations
    np.random.seed(42)

    # Create directories for training and testing datasets
    os.makedirs(os.path.join(chemin_dataset, 'entrainement'), exist_ok=True)
    os.makedirs(os.path.join(chemin_dataset, 'test'), exist_ok=True)

    # Iterate over each fruit folder
    for dossier in os.listdir(chemin_dataset):
        chemin_dossier = os.path.join(chemin_dataset, dossier)
        
        # Check if it is a directory and not the 'entrainement' or 'test' directory
        if os.path.isdir(chemin_dossier) and dossier not in ['entrainement', 'test']:
            # Create subdirectories for training and testing within each fruit directory
            os.makedirs(os.path.join(chemin_dataset, 'entrainement', dossier), exist_ok=True)
            os.makedirs(os.path.join(chemin_dataset, 'test', dossier), exist_ok=True)
            
            # List all image files in the directory
            fichiers = os.listdir(chemin_dossier)
            np.random.shuffle(fichiers)  # Shuffle the file list for random selection
            
            # Calculate the split index for 80% training data
            index_split = int(0.8 * len(fichiers))
            
            # Divide the files into training and testing sets
            fichiers_entrainement = fichiers[:index_split]
            fichiers_test = fichiers[index_split:]
            
            # Copy files to the new training and testing directories
            for fichier in fichiers_entrainement:
                shutil.copy(os.path.join(chemin_dossier, fichier), os.path.join(chemin_dataset, 'entrainement', dossier))
            
            for fichier in fichiers_test:
                shutil.copy(os.path.join(chemin_dossier, fichier), os.path.join(chemin_dataset, 'test', dossier))

            # Remove the original directory to free up space
            shutil.rmtree(chemin_dossier)

    print("Dataset split into training and testing sets completed! All original folders have been removed.")

except Exception as e:
    # Handle exceptions and report errors
    print(f"An error occurred: {e}")
