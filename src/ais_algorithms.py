import numpy as np
from preprocess import load_kdd_data, load_iotnid_data, apply_pca
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import pickle

def generate_detectors(self_data, num_detectors, threshold, data_min, data_max):
    detectors = []
    for i in range(num_detectors):
        detector = np.random.uniform(data_min, data_max)
        matches_self = False
        for self_sample in self_data:
            distance = np.linalg.norm(detector - self_sample)
            if distance < threshold:
                matches_self = True
                break
        if not matches_self:
            detectors.append(detector)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_detectors} detectors...")
    return np.array(detectors)

def clonal_selection(detectors, anomaly_data, affinity_threshold, clone_rate, mutation_rate, max_detectors, data_min, data_max):
    new_detectors = detectors.copy()
    for sample in anomaly_data:
        for detector in new_detectors:
            affinity = np.linalg.norm(sample - detector)
            if affinity < affinity_threshold:
                num_clones = int(clone_rate * (1 / (affinity + 1e-10)))
                for _ in range(num_clones):
                    mutated_detector = detector + np.random.normal(0, mutation_rate, detector.shape)
                    mutated_detector = np.clip(mutated_detector, data_min, data_max)
                    new_detectors = np.vstack([new_detectors, mutated_detector])
    
    if len(new_detectors) > max_detectors:
        affinities = [min(np.linalg.norm(anomaly_data - d, axis=1)) for d in new_detectors]
        indices = np.argsort(affinities)[:max_detectors]
        new_detectors = new_detectors[indices]
    
    return new_detectors

def artificial_immune_network(detectors, anomaly_data, suppression_threshold, stimulation_factor, max_detectors):
    network = detectors.copy()
    # Stimulate detectors based on affinity to anomalies
    for i, detector in enumerate(network):
        min_distance = min(np.linalg.norm(anomaly_data - detector, axis=1))
        if min_distance < suppression_threshold:
            network[i] *= (1 + stimulation_factor)
    
    # Suppress similar detectors
    for i, d1 in enumerate(network):
        for j, d2 in enumerate(network[i+1:], start=i+1):
            distance = np.linalg.norm(d1 - d2)
            if distance < suppression_threshold:
                aff1 = min(np.linalg.norm(anomaly_data - d1, axis=1))
                aff2 = min(np.linalg.norm(anomaly_data - d2, axis=1))
                if aff1 > aff2:
                    network[i] *= (1 - stimulation_factor)
                else:
                    network[j] *= (1 - stimulation_factor)
    
    # Select top detectors based on affinity to anomalies
    if len(network) > max_detectors:
        affinities = [min(np.linalg.norm(anomaly_data - d, axis=1)) for d in network]
        indices = np.argsort(affinities)[:max_detectors]
        network = network[indices]
    
    return network

def detect_anomalies(data, detectors, threshold):
    predictions = []
    distances = []
    for i, sample in enumerate(data):
        min_distance = float('inf')
        for detector in detectors:
            distance = np.linalg.norm(sample - detector)
            min_distance = min(min_distance, distance)
            if distance < threshold:
                predictions.append(1)
                break
        else:
            predictions.append(0)
        distances.append(min_distance)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...")
    return predictions, distances

def evaluate_predictions(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    fpr = np.sum((np.array(predictions) == 1) & (np.array(true_labels) == 0)) / np.sum(np.array(true_labels) == 0) if np.sum(np.array(true_labels) == 0) > 0 else 0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fpr': fpr
    }

def save_detectors(detectors, filename):
    with open(filename, 'wb') as f:
        pickle.dump(detectors, f)

if __name__ == "__main__":
    # Load and preprocess KDD data
    print("Processing KDD Cup 1999...")
    try:
        kdd_self, kdd_non_self = load_kdd_data('data/kddcup.data_10_percent')
        kdd_self_reduced, pca = apply_pca(kdd_self, n_components=10)
        kdd_non_self_reduced = pca.transform(kdd_non_self)
        
        # Normalize PCA output to [0,1]
        scaler = MinMaxScaler()
        kdd_all_reduced = np.vstack([kdd_self_reduced, kdd_non_self_reduced])
        kdd_all_scaled = scaler.fit_transform(kdd_all_reduced)
        kdd_self_scaled = kdd_all_scaled[:len(kdd_self_reduced)]
        kdd_non_self_scaled = kdd_all_scaled[len(kdd_self_reduced):]
        data_min, data_max = kdd_all_scaled.min(axis=0), kdd_all_scaled.max(axis=0)
        
        # Generate detectors
        detectors_kdd = generate_detectors(kdd_self_scaled, num_detectors=1000, threshold=0.2, data_min=data_min, data_max=data_max)
        print("KDD Detectors Generated:", len(detectors_kdd))
        
        # Refine with CSA
        detectors_kdd = clonal_selection(
            detectors_kdd, kdd_non_self_scaled[:500],
            affinity_threshold=0.2, clone_rate=10, mutation_rate=0.1, max_detectors=1000,
            data_min=data_min, data_max=data_max
        )
        print("KDD Detectors After CSA:", len(detectors_kdd))
        
        # Refine with AIN
        detectors_kdd = artificial_immune_network(
            detectors_kdd, kdd_non_self_scaled[:500], suppression_threshold=0.3, stimulation_factor=0.05, max_detectors=1000
        )
        print("KDD Detectors After AIN:", len(detectors_kdd))
        save_detectors(detectors_kdd, 'data/detectors_kdd.pkl')
        
        # Test with different thresholds
        thresholds = [0.5, 0.6, 0.7]
        test_data_kdd = np.vstack([kdd_self_scaled[:200], kdd_non_self_scaled[:200]])
        true_labels_kdd = [0] * 200 + [1] * 200
        for threshold in thresholds:
            print(f"\nTesting KDD with threshold {threshold}...")
            predictions_kdd, distances_kdd = detect_anomalies(test_data_kdd, detectors_kdd, threshold)
            metrics_kdd = evaluate_predictions(true_labels_kdd, predictions_kdd)
            print("KDD Metrics:", metrics_kdd)
            print("Average Distance to Nearest Detector:", np.mean(distances_kdd))
    except Exception as e:
        print("Error processing KDD:", e)
    
    # Load and preprocess IoTNID data
    print("\nProcessing IoTNID...")
    try:
        iot_self, iot_non_self = load_iotnid_data('data/IoTNID.csv')
        # Skip PCA for IoTNID to retain more information
        iot_self_scaled = scaler.fit_transform(iot_self)
        iot_non_self_scaled = scaler.transform(iot_non_self)
        data_min, data_max = iot_self_scaled.min(axis=0), iot_self_scaled.max(axis=0)
        
        # Balance attack data for CSA
        np.random.seed(42)
        indices = np.random.choice(len(iot_non_self_scaled), size=len(iot_self_scaled), replace=False)
        balanced_non_self_scaled = iot_non_self_scaled[indices]
        
        # Generate detectors
        detectors_iot = generate_detectors(iot_self_scaled, num_detectors=1000, threshold=0.2, data_min=data_min, data_max=data_max)
        print("IoTNID Detectors Generated:", len(detectors_iot))
        
        # Refine with CSA
        detectors_iot = clonal_selection(
            detectors_iot, balanced_non_self_scaled[:500],
            affinity_threshold=0.2, clone_rate=10, mutation_rate=0.1, max_detectors=1000,
            data_min=data_min, data_max=data_max
        )
        print("IoTNID Detectors After CSA:", len(detectors_iot))
        
        # Refine with AIN
        detectors_iot = artificial_immune_network(
            detectors_iot, balanced_non_self_scaled[:500], suppression_threshold=0.3, stimulation_factor=0.05, max_detectors=1000
        )
        print("IoTNID Detectors After AIN:", len(detectors_iot))
        save_detectors(detectors_iot, 'data/detectors_iot.pkl')
        
        # Test with different thresholds
        thresholds = [0.5, 0.6, 0.7]
        test_data_iot = np.vstack([iot_self_scaled[:200], iot_non_self_scaled[:200]])
        true_labels_iot = [0] * 200 + [1] * 200
        for threshold in thresholds:
            print(f"\nTesting IoTNID with threshold {threshold}...")
            predictions_iot, distances_iot = detect_anomalies(test_data_iot, detectors_iot, threshold)
            metrics_iot = evaluate_predictions(true_labels_iot, predictions_iot)
            print("IoTNID Metrics:", metrics_iot)
            print("Average Distance to Nearest Detector:", np.mean(distances_iot))
    except Exception as e:
        print("Error processing IoTNID:", e)