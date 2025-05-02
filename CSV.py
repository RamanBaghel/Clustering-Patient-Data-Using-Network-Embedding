import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# Directory containing the CSV files
csv_directory = '/home/project/matching/output/csv'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
print(csv_files)

# Select the encounter, careplans, and allergies files by checking the name
encounter_file = [f for f in csv_files if 'encounter' in f.lower()]
careplans_file = [f for f in csv_files if 'careplans' in f.lower()]
allergies_file = [f for f in csv_files if 'allergies' in f.lower()]
patients_file = [f for f in csv_files if 'patients' in f.lower()]

# Load the encounter file into a dataframe
if encounter_file:
    encounter_file = encounter_file[0]  # Assuming there is one encounter file
    encounter_file_path = os.path.join(csv_directory, encounter_file)
    encounters_df = pd.read_csv(encounter_file_path)
    print(f"Loaded {encounter_file} into dataframe 'encounters_df'")

# Load the careplans file into a dataframe
if careplans_file:
    careplans_file = careplans_file[0]  # Assuming there is one careplans file
    careplans_file_path = os.path.join(csv_directory, careplans_file)
    careplans_df = pd.read_csv(careplans_file_path)
    print(f"Loaded {careplans_file} into dataframe 'careplans_df'")

# Load the allergies file into a dataframe
if allergies_file:
    allergies_file = allergies_file[0]  # Assuming there is one allergies file
    allergies_file_path = os.path.join(csv_directory, allergies_file)
    allergies_df = pd.read_csv(allergies_file_path)
    print(f"Loaded {allergies_file} into dataframe 'allergies_df'")

# Load the patients file into a dataframe
if patients_file:
    patients_file = patients_file[0]  # Assuming there is one patients file
    patients_file_path = os.path.join(csv_directory, patients_file)
    patients_df = pd.read_csv(patients_file_path)
    print(f"Loaded {patients_file} into dataframe 'patients_df'")

    # Rename 'Id' column to 'PATIENT' for consistency with other dataframes
    patients_df = patients_df.rename(columns={'Id': 'PATIENT'})

from datetime import datetime

# Ensure BIRTHDATE and DEATHDATE are in datetime format
#patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'], errors='coerce')
#patients_df['DEATHDATE'] = pd.to_datetime(patients_df['DEATHDATE'], errors='coerce')

# Calculate age at the time of death or today if alive
#today = datetime.today()
#patients_df['AGE'] = (patients_df['DEATHDATE'] - patients_df['BIRTHDATE']).dt.days // 365
#patients_df['AGE'] = patients_df['AGE'].fillna((today - patients_df['BIRTHDATE']).dt.days // 365)

# Add a binary column for deceased status
#patients_df['IS_DECEASED'] = patients_df['DEATHDATE'].notna().astype(int)

# Optional: Calculate time since death for deceased patients
#patients_df['TIME_SINCE_DEATH'] = (today - patients_df['DEATHDATE']).dt.days
#patients_df['TIME_SINCE_DEATH'] = patients_df['TIME_SINCE_DEATH'].fillna(0)  # 0 for alive patients


# Select the necessary columns from all dataframes
encounters_df = encounters_df[['PATIENT', 'Id', 'DESCRIPTION', 'ENCOUNTERCLASS', 'CODE', 'REASONDESCRIPTION']]
careplans_df = careplans_df[['PATIENT', 'Id', 'DESCRIPTION', 'REASONDESCRIPTION']]
allergies_df = allergies_df[['PATIENT', 'DESCRIPTION', 'DESCRIPTION1', 'SEVERITY1', 'DESCRIPTION2']]
patients_df = patients_df[['PATIENT', 'RACE', 'ETHNICITY', 'GENDER', 'HEALTHCARE_EXPENSES']]


# Combine the three dataframes
combined_df = pd.concat([encounters_df, careplans_df, allergies_df, patients_df], ignore_index=True)

# Initialize the StandardScaler
scaler = StandardScaler()

# Columns to scale
#columns_to_scale = ['AGE', 'TIME_SINCE_DEATH']

# Apply Standard Scaling
#combined_df[columns_to_scale] = scaler.fit_transform(combined_df[columns_to_scale])

# Verify the scaled values
#print("Standardized AGE and TIME_SINCE_DEATH:")
#print(combined_df[columns_to_scale].head())

# Save the combined dataframe to a CSV file
#combined_csv_file = '/home/project/matching//output/csv/combined_encoun                                                                                                                                         ters_careplans_allergies_patients.csv'
#combined_df.to_csv(combined_csv_file, index=False)
#print(f"Combined dataframe saved to: {combined_csv_file}")

# List of columns to be one-hot encoded
columns_to_encode = ['ENCOUNTERCLASS', 'DESCRIPTION', 'REASONDESCRIPTION', 
                     'DESCRIPTION1', 'SEVERITY1', 'DESCRIPTION2', 
                     'RACE', 'ETHNICITY', 'GENDER']

# Initialize OneHotEncoder with sparse matrix option
encoder = OneHotEncoder(sparse_output=True, drop='first')  # drop='first' is optional, drops the first category for each column to avoid multicollinearity

# Fit and transform the specified columns into a sparse matrix
encoded_sparse = encoder.fit_transform(combined_df[columns_to_encode])

# Convert the sparse matrix to a DataFrame for easy inspection (if needed)
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_sparse, columns=encoder.get_feature_names_out(columns_to_encode))


# Optionally, concatenate the encoded data with the rest of the DataFrame
combined_df_encoded = pd.concat([combined_df.drop(columns=columns_to_encode), encoded_df], axis=1)

# Show the resulting sparse DataFrame (will be large, so use head() for preview)
print("DataFrame after one-hot encoding with sparse matrix:")
print(combined_df_encoded.head())

# Run DBSCAN clustering
dbscan_model = DBSCAN(eps=2, min_samples=3)

# Exclude identifiers but include numerical columns
X = combined_df_encoded.drop(columns=['PATIENT', 'Id'])

# Include numerical columns for clustering
#X['AGE'] = combined_df['AGE']
#X['IS_DECEASED'] = combined_df['IS_DECEASED']
#X['TIME_SINCE_DEATH'] = combined_df['TIME_SINCE_DEATH']

# Fill NaN values with 0 for clustering
X = X.fillna(0)

# Apply DBSCAN
dbscan_labels = dbscan_model.fit_predict(X)

# Add the cluster labels to the encoded dataframe
combined_df_encoded['Cluster'] = dbscan_labels

# # Save the clustered dataframe to a new CSV file
clustered_output_csv_file = '/home/project/matching/clustered_combined_allergies.csv'
combined_df_encoded.to_csv(clustered_output_csv_file, index=False)
print(f"Clustering complete. Results saved to: {clustered_output_csv_file}")

# Show the number of DBSCAN clusters (unique cluster labels)
unique_clusters_dbscan = np.unique(dbscan_labels)
print(f"Number of DBSCAN clusters: {len(unique_clusters_dbscan)} (including noise)")

# Calculate total number of points
total_points = len(dbscan_labels)

# Calculate DBSCAN noise points
dbscan_noise_points = np.sum(dbscan_labels == -1)

# Print results
print(f"Total number of points: {total_points}")
print(f"Number of noise points in DBSCAN: {dbscan_noise_points}")

# Apply KMeans clustering
kmeans_model = KMeans(n_clusters=50, random_state=42)  # You can change the number of clusters
kmeans_labels = kmeans_model.fit_predict(X)

# Add the KMeans cluster labels to the encoded dataframe
combined_df_encoded['KMeans_Cluster'] = kmeans_labels

# Show the number of KMeans clusters
unique_clusters_kmeans = np.unique(kmeans_labels)
print(f"Number of KMeans clusters: {len(unique_clusters_kmeans)}")

# Function to compute Jaccard coefficient (same as in your original code)
def compute_jaccard_coefficient(set1, set2):
    """Compute Jaccard coefficient between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def compute_jaccard_matrix(cluster_df):
    """
    Compute a pairwise Jaccard similarity matrix for a given cluster.
    
    Args:
    - cluster_df: DataFrame containing one-hot encoded encounter classes for patients in the cluster.
    
    Returns:
    - A condensed matrix of pairwise Jaccard distances (1 - similarity).
    """
    # Convert the DataFrame to a binary matrix (1/0 values)
    binary_matrix = cluster_df.values
    
    # Compute pairwise Jaccard distances
    jaccard_distances = pdist(binary_matrix, metric='jaccard')
    
    # Convert distances to similarities
    jaccard_similarities = 1 - jaccard_distances
    return jaccard_similarities

def cluster_score_optimized(df, labels):
    """
    Calculate the Jaccard similarity score for each cluster using an optimized approach.
    
    Args:
    - df: DataFrame with one-hot encoded data.
    - labels: cluster labels.
    
    Returns:
    - A dictionary with cluster labels as keys and their Jaccard score as values.
    """
    # Add the cluster labels to the DataFrame
    df['Cluster'] = labels
    
    cluster_scores = {}
    
    # Loop through each unique cluster
    for cluster in np.unique(labels):
        if cluster == -1:  # Skip noise points
            continue
        
        # Select patients in the current cluster
        cluster_df = df[df['Cluster'] == cluster].drop(columns=['PATIENT', 'Id', 'Cluster'])
        
        # Skip if the cluster has fewer than 2 patients
        if len(cluster_df) < 2:
            cluster_scores[cluster] = 0
            continue
        
        # Compute the pairwise Jaccard similarity matrix
        jaccard_similarities = compute_jaccard_matrix(cluster_df)
        
        # Calculate the average Jaccard similarity for the cluster
        cluster_scores[cluster] = jaccard_similarities.mean() if len(jaccard_similarities) > 0 else 0
    
    return cluster_scores


# Calculate points per cluster in KMeans (since KMeans has no noise points)
kmeans_points_per_cluster = pd.Series(kmeans_labels).value_counts()
print("Number of points per KMeans cluster:")
print(kmeans_points_per_cluster)

# Compute Jaccard scores for DBSCAN and KMeans clusters using the optimized method
dbscan_scores_optimized = cluster_score_optimized(combined_df_encoded, dbscan_labels)
kmeans_scores_optimized = cluster_score_optimized(combined_df_encoded, kmeans_labels)

# Print the optimized cluster scores
print("Optimized DBSCAN Cluster scores:", dbscan_scores_optimized)
print("Optimized KMeans Cluster scores:", kmeans_scores_optimized)
