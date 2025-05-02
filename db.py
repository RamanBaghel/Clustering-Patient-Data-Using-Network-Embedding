import psycopg2 
import time
import logging
import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
from networkx.algorithms.community import label_propagation_communities


# Setup logging
logging.basicConfig(filename='script_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting the script")

# Connect to the database
try:
    conn = psycopg2.connect(host="localhost", dbname="test", user="project", password="birne", port=5432)
    cur = conn.cursor()
    logging.info("Connected to the database")
except Exception as ex:
    logging.error(f"Error connecting to the database: {ex}")
    raise

# Initialize graph and patient list
G = nx.DiGraph()
all_patient_ids = []  # Use a list to allow duplicates

# Utility functions for fetching and adding data to the graph
def fetch_data(query):
    try:
        cur.execute(query)
        return cur.fetchall()
    except Exception as ex:
        logging.error(f"Error fetching data: {ex}")
        raise

def add_to_graph(G, data, data_type):
    for row in data:
        if data_type == 'careplans':
            G.add_node(row[0], label=f"ID: {row[0]}")
            G.add_node(row[1], label=f"PATIENT: {row[1]}")
            all_patient_ids.append(row[1])
            G.add_node(row[4], label=f"TREATMENT: {row[4]}")
            if row[5] is not None:
                G.add_node(row[5], label=f"DIAGNOSE: {row[5]}")
                G.add_edge(row[0], row[5], label="diagnose")
            G.add_edge(row[1], row[0], label="uniqueid")
            G.add_edge(row[0], row[4], label="treatment")
        elif data_type == 'encounters':
            G.add_node(row[0], label=f"ID: {row[0]}")
            G.add_node(row[1], label=f"PATIENT: {row[1]}")
            all_patient_ids.append(row[1])
            G.add_node(row[4], label=f"TREATMENTCODE: {row[4]}")
            G.add_node(row[5], label=f"TREATMENT: {row[5]}")
            if row[6] is not None:
                G.add_node(row[6], label=f"DIAGNOSE: {row[4]}")
                G.add_edge(row[0], row[6], label="diagnose")
            G.add_node(row[7], label=f"ENCOUNTERCLASS: {row[6]}")
            G.add_edge(row[0], row[1], label="uniqueid")
            G.add_edge(row[0], row[5], label="treatment")
            G.add_edge(row[0], row[4], label="treatmentcode")
            G.add_edge(row[0], row[7], label="encounterclass")
        elif data_type == 'patients':
            G.add_node(row[0], label=f"PATIENT: {row[0]}")
            G.add_node(row[1], label=f"BIRTHDATE: {row[1]}")
            if row[2] is not None:
                G.add_node(row[2], label=f"DEATHDATE: {row[2]}")
            G.add_node(row[3], label=f"RACE: {row[3]}")
            G.add_node(row[4], label=f"ETHNICITY: {row[4]}")
            G.add_node(row[5], label=f"GENDER: {row[5]}")
            G.add_node(row[6], label=f"HEALTHCARE_EXPENSES: {row[6]}")
            G.add_edge(row[0], row[1], label="birthdate")
            if row[2] is not None:
                G.add_edge(row[0], row[2], label="deathdate")
            G.add_edge(row[0], row[3], label="race")
            G.add_edge(row[0], row[4], label="ethnicity")
            G.add_edge(row[0], row[5], label="gender")
            G.add_edge(row[0], row[6], label="healthcareexpenses")
        elif data_type == 'allergies':
            patient_id = row[2]
            G.add_node(patient_id, label=f"PATIENT: {patient_id}")
            all_patient_ids.append(patient_id)
            allergystart_label = f"ALLERGYSTART: {row[0]}"
            allergystop_label = f"ALLERGYSTOP: {row[1]}" if row[1] is not None else None
            description_label = f"DESCRIPTION: {row[3]}"
            description1_label = f"DESCRIPTION1: {row[4]}" if row[4] is not None else None
            severity1_label = f"SEVERITY1: {row[5]}" if row[5] is not None else None
            description2_label = f"DESCRIPTION2: {row[6]}" if row[6] is not None else None
            G.add_node(allergystart_label)
            if allergystop_label:
                G.add_node(allergystop_label)
            G.add_node(description_label)
            if description1_label:
                G.add_node(description1_label)
            if severity1_label:
                G.add_node(severity1_label)
            if description2_label:
                G.add_node(description2_label)
            G.add_edge(patient_id, allergystart_label, label="allergystart")
            if allergystop_label:
                G.add_edge(patient_id, allergystop_label, label="allergystop")
            G.add_edge(patient_id, description_label, label="description")
            if description1_label:
                G.add_edge(patient_id, description1_label, label="description1")
            if severity1_label:
                G.add_edge(patient_id, severity1_label, label="severity1")
            if description2_label:
                G.add_edge(patient_id, description2_label, label="description2")

# Fetch data and add to graph
careplans_data = fetch_data('SELECT Id, PATIENT, START, STOP, DESCRIPTION, REASONDESCRIPTION FROM careplans')
add_to_graph(G, careplans_data, 'careplans')
logging.info(f"Fetched {len(careplans_data)} careplans records")

encounters_data = fetch_data('SELECT Id, PATIENT, START, STOP, CODE, DESCRIPTION, REASONDESCRIPTION, ENCOUNTERCLASS FROM encounters')
add_to_graph(G, encounters_data, 'encounters')
logging.info(f"Fetched {len(encounters_data)} encounters records")

patients_data = fetch_data('SELECT Id, birth_date, death_date, race, ethnicity, gender, healthcare_expenses FROM patients')
add_to_graph(G, patients_data, 'patients')
logging.info(f"Fetched {len(patients_data)} patients records")

allergies_data = fetch_data('SELECT START, STOP, PATIENT, DESCRIPTION, DESCRIPTION_1, SEVERITY_1, DESCRIPTION_2 FROM allergies')
add_to_graph(G, allergies_data, 'allergies')
logging.info(f"Fetched {len(allergies_data)} allergies records")

number_of_patient_ids = len(all_patient_ids)
print(f"Number of unique patient IDs: {number_of_patient_ids}")

partition = community_louvain.best_partition(G)
print("Louvain clusters (patient nodes only):")

# Filter patient nodes and their clusters clearly
patient_partition = {node: cluster for node, cluster in partition.items() 
                     if G.nodes[node].get('label', '').startswith('PATIENT:')}

for patient_node, cluster_id in patient_partition.items():
    patient_id = G.nodes[patient_node]['label'].split('PATIENT: ')[1]
    print(f"Patient ID: {patient_id} -> Cluster: {cluster_id}")

# Apply Label Propagation
lp_communities = label_propagation_communities(G)

# Convert communities to a patient-cluster mapping
patient_clusters_lp = {}
for cluster_id, community in enumerate(lp_communities):
    for node in community:
        node_label = G.nodes[node].get('label', '')
        if node_label.startswith('PATIENT:'):
            patient_id = node_label.replace('PATIENT: ', '')
            patient_clusters_lp[patient_id] = cluster_id

# Show the total number of clusters detected
print(f"Total Label Propagation Clusters: {cluster_id + 1}")


# Jaccard index function
def jaccard_index(a, b):
    return len(a.intersection(b)) / len(a.union(b))

def cluster_score(patient_ids):
    if len(patient_ids) < 2:
        return 0

    encounter_classes = []
    for pid in patient_ids:  # Do not filter out duplicates here
        patient_encounters = set()
        cur.execute('SELECT ENCOUNTERCLASS FROM encounters WHERE PATIENT = %s', (pid,))
        for row in cur:
            patient_encounters.add(row[0])
        encounter_classes.append(patient_encounters)

    scores = []
    for i in range(len(encounter_classes)):
        for j in range(i+1, len(encounter_classes)):
            scores.append(jaccard_index(encounter_classes[i], encounter_classes[j]))

    return sum(scores) / len(scores)    

# Node2vec embedding
def generate_node2vec_embeddings(graph):
    node2vec = Node2Vec(graph, dimensions=128, walk_length=10, num_walks=10, workers=2)
    model = node2vec.fit(window=10, min_count=1)
    return {node: model.wv[str(node)] for node in graph.nodes() if str(node) in model.wv}

start = time.time()
embeddings = generate_node2vec_embeddings(G)
logging.info(f"Node2Vec embedding generation time: {time.time() - start}")

# Dictionary of Embedding (Prepare data for clustering)
X = np.array([val for idx, val in embeddings.items()])

# Number of nodes embedded
number_of_nodes = X.shape[0]
print("Number of nodes embedded:", number_of_nodes)

# DBSCAN Clustering
for eps in [1.2]:
    for min_samples in [5]:
        start = time.time()
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
        logging.info(f"DBSCAN fit time (eps={eps}, min_samples={min_samples}): {time.time() - start}")
        labels = db.labels_
        core_indices = db.core_sample_indices_
        core_points = X[core_indices]
        noise_points = X[labels == -1]
        unique_labels = np.unique(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        logging.info(f'Parameters: eps={eps}, min_samples={min_samples}')
        logging.info(f'Number of clusters: {n_clusters_}')
        logging.info(f'Number of noise points: {n_noise_}')
        logging.info(f'Unique labels: {unique_labels}')
        logging.info(f'Labels: {labels}')

# Calculate cluster scores after DBSCAN clustering
def calculate_cluster_scores(labels):
    unique_clusters = set(labels)
    cluster_scores = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            # Skip noise points (label -1)
            continue

        # Get the patient IDs for the current cluster
        cluster_patient_ids = [patient_id for patient_id, label in zip(all_patient_ids, labels) if label == cluster_id]

        # Compute the cluster score using the Jaccard similarity
        score = cluster_score(cluster_patient_ids)
        cluster_scores[cluster_id] = score
        logging.info(f"Cluster {cluster_id} score: {score}")

    return cluster_scores

# Call this function to calculate scores after DBSCAN
cluster_scores = calculate_cluster_scores(labels)

# Log or print the cluster scores
for cluster_id, score in cluster_scores.items():
    print(f"Cluster {cluster_id} Jaccard Score: {score}")
    logging.info(f"Cluster {cluster_id} Jaccard Score: {score}")

 # After your existing code:
louvain_cluster_scores = {}

for cluster_id in set(partition.values()):
    patient_ids = [node for node, cid in partition.items()
                   if G.nodes[node].get('label', '').startswith('PATIENT:') and cid == cluster_id]

    score = cluster_score(patient_ids)
    louvain_cluster_scores[cluster_id] = score
    print(f"Louvain Cluster {cluster_id} has {len(patient_ids)} patients and Jaccard Score: {score:.3f}")
   
# Evaluate Label Propagation clusters using Jaccard similarity
lp_cluster_scores = {}
for cluster_id in set(patient_clusters_lp.values()):
    patient_ids = [pid for pid, cid in patient_clusters_lp.items() if cid == cluster_id]
    score = cluster_score(patient_ids)
    lp_cluster_scores[cluster_id] = score
    print(f"LP Cluster {cluster_id}: Patients = {len(patient_ids)}, Jaccard Score = {score:.3f}")

# Map DBSCAN cluster labels to the nodes in the graph (node_labels is a mapping of node -> cluster)
node_labels = {node: label for node, label in zip(G.nodes(), labels)}


# Convert each dataset into DataFrame and map the cluster labels
# Careplans DataFrame
df_careplans = pd.DataFrame(careplans_data, columns=['Id', 'PATIENT', 'START', 'STOP', 'DESCRIPTION', 'REASONDESCRIPTION'])
df_careplans['Node'] = df_careplans['Id']  # Assuming 'Id' is the node identifier
df_careplans['Cluster'] = df_careplans['Node'].map(node_labels)

# Encounters DataFrame
df_encounters = pd.DataFrame(encounters_data, columns=['Id', 'PATIENT', 'START', 'STOP', 'CODE', 'DESCRIPTION', 'REASONDESCRIPTION', 'ENCOUNTERCLASS'])
df_encounters['Node'] = df_encounters['Id']  # Assuming 'Id' is the node identifier
df_encounters['Cluster'] = df_encounters['Node'].map(node_labels)

# Patients DataFrame
df_patients = pd.DataFrame(patients_data, columns=['Id', 'birth_date', 'death_date', 'race', 'ethnicity', 'gender', 'healthcare_expenses'])
df_patients['Node'] = df_patients['Id']  # Assuming 'Id' is the node identifier
df_patients['Cluster'] = df_patients['Node'].map(node_labels)

# Allergies DataFrame
df_allergies = pd.DataFrame(allergies_data, columns=['START', 'STOP', 'PATIENT', 'DESCRIPTION', 'DESCRIPTION_1', 'SEVERITY_1', 'DESCRIPTION_2'])
df_allergies['Node'] = df_allergies['PATIENT']  # Assuming 'PATIENT' is the node identifier in allergies
df_allergies['Cluster'] = df_allergies['Node'].map(node_labels)

# Combine all the data into a single DataFrame and export to CSV
combined_df = pd.concat([df_careplans, df_encounters, df_patients, df_allergies], ignore_index=True)
combined_df.to_csv('clustered_healthcare_data.csv', index=False)

print("Data has been exported to clustered_healthcare_data.csv")

conn.commit()
cur.close()
conn.close()