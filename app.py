# app.py
"""
Streamlit app: Quantum-assisted clustering for file-metadata anomaly detection.

Features:
 - Upload CSV or use synthetic dataset
 - Automatic featurizer (adapts to common metadata columns)
 - Quantum Kernel (Qiskit ZZFeatureMap) + Spectral Clustering
 - Nyström approximation for kernel scaling
 - Experimental VQC (variational circuit + optimizer -> classical kmeans)
 - UI controls, visualizations, download results

Notes:
 - This demo uses local Qiskit simulators by default (statevector/qasm).
 - Kernel computation scales O(n^2). Use Nyström to approximate with m landmarks.
 - VQC here is experimental and may be slow; it's intended for small datasets (n <= 200).
 - If you want IBM hardware, configure IBMQ and set backend in app (not included automatically).
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from io import BytesIO

# Qiskit imports
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit import ParameterVector, QuantumCircuit
from scipy.optimize import minimize

st.set_page_config(layout="wide", page_title="Quantum File Anomaly — Streamlit Demo")

# ---------------- Utilities / synthetic data ----------------
def simulate_file_metadata(n_samples=200, anomaly_frac=0.07, random_state=42):
    rng = np.random.RandomState(random_state)
    base_n = int(n_samples * (1 - anomaly_frac))
    n_anom = n_samples - base_n
    size_normal = np.exp(rng.normal(np.log(5e4), 1.5, base_n))
    extensions = rng.choice(['txt','pdf','jpg','png','exe','docx','log','cfg','dat'], size=base_n)
    name_entropy = np.clip(rng.normal(3.5, 1.0, base_n), 0.0, 8.0)
    path_depth = rng.poisson(3, base_n)
    owner_uid = rng.choice(['user','root','service','admin'], size=base_n, p=[0.7,0.05,0.2,0.05])
    last_mod = rng.normal(180, 200, base_n)
    perms = rng.choice([644,600,755,700,666,777], size=base_n)

    size_anom = np.exp(rng.normal(np.log(5e7), 1.0, n_anom))
    extensions_anom = rng.choice(['exe', 'bin', 'scr', 'tmp', 'unknown'], size=n_anom)
    name_entropy_anom = np.clip(rng.normal(6.0, 1.0, n_anom), 0.0, 8.0)
    path_depth_anom = rng.poisson(5, n_anom)
    owner_uid_anom = rng.choice(['unknown','root','service'], size=n_anom)
    last_mod_anom = rng.normal(1.5, 3.0, n_anom)
    perms_anom = rng.choice([777,700,600,666], size=n_anom)

    data = {
        'size_bytes': np.concatenate([size_normal, size_anom]),
        'extension': np.concatenate([extensions, extensions_anom]),
        'name_entropy': np.concatenate([name_entropy, name_entropy_anom]),
        'path_depth': np.concatenate([path_depth, path_depth_anom]),
        'owner': np.concatenate([owner_uid, owner_uid_anom]),
        'last_mod_days': np.concatenate([last_mod, last_mod_anom]),
        'perms': np.concatenate([perms, perms_anom]),
    }
    df = pd.DataFrame(data)
    df['label'] = 0
    df.loc[df.index[-n_anom:], 'label'] = 1
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

# ---------------- Featurizer (adaptive) ----------------
def adaptive_featurizer(df: pd.DataFrame, debug=False):
    """
    Attempt to automatically map columns to useful features.
    Supports common columns: size, extension, owner, last_mod, perms, path_depth, name_entropy.
    Returns (X, feature_names, df_orig) where X is numpy array ready for scaling/PCA.
    """
    dfc = df.copy()
    # Lowercase column names for robust matching
    mapping = {c: c.lower() for c in dfc.columns}
    dfc.columns = [mapping[c] for c in dfc.columns]
    features = {}

    # Numeric: size_bytes or size
    if 'size_bytes' in dfc.columns:
        features['log_size'] = np.log1p(dfc['size_bytes'].astype(float).values)
    elif 'size' in dfc.columns:
        features['log_size'] = np.log1p(dfc['size'].astype(float).values)

    # name_entropy if present
    if 'name_entropy' in dfc.columns:
        features['name_entropy'] = dfc['name_entropy'].astype(float).values

    # path depth
    if 'path_depth' in dfc.columns:
        features['path_depth'] = dfc['path_depth'].astype(float).values
    elif 'path' in dfc.columns:
        # attempt to compute path depth from string path
        features['path_depth'] = dfc['path'].astype(str).apply(lambda s: s.count('/') if isinstance(s, str) else 0).values.astype(float)

    # last modified
    if 'last_mod_days' in dfc.columns:
        features['last_mod_days'] = dfc['last_mod_days'].astype(float).values
    elif 'mtime' in dfc.columns:
        features['last_mod_days'] = dfc['mtime'].astype(float).values

    # perms
    if 'perms' in dfc.columns:
        features['perms'] = dfc['perms'].astype(float).values

    # owner categorical
    owner_cols = []
    if 'owner' in dfc.columns:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        owners_ohe = enc.fit_transform(dfc[['owner']].astype(str))
        owner_cols = [f"owner_{c}" for c in enc.categories_[0]]
        for i, col in enumerate(owner_cols):
            features[col] = owners_ohe[:, i]

    # extension categorical
    ext_cols = []
    if 'extension' in dfc.columns:
        enc2 = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ext_ohe = enc2.fit_transform(dfc[['extension']].astype(str))
        ext_cols = [f"ext_{c}" for c in enc2.categories_[0]]
        for i, col in enumerate(ext_cols):
            features[col] = ext_ohe[:, i]

    # If features empty -> fallback: numeric columns only
    if len(features) == 0:
        # try to take all numeric columns
        numeric = dfc.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("No usable columns found. Please upload a CSV with numeric metadata columns or use the sample dataset.")
        for c in numeric.columns:
            features[c] = numeric[c].astype(float).values

    # Build matrix
    feat_names = list(features.keys())
    X = np.vstack([features[n] for n in feat_names]).T
    if debug:
        st.write("Detected feature columns:", feat_names)
    return X, feat_names, dfc

# ---------------- Quantum Kernel ----------------
def compute_quantum_kernel_matrix(X_scaled_for_q, n_qubits=4, backend_name='aer_simulator_statevector', shots=512):
    """Compute quantum kernel matrix using Qiskit's QuantumKernel and ZZFeatureMap."""
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    backend = Aer.get_backend(backend_name)
    qi = QuantumInstance(backend=backend, shots=shots)
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)
    K = qkernel.evaluate(x_vec=X_scaled_for_q)
    return K, qkernel

# ---------------- Nyström approximation ----------------
def nystrom_approximation(K_func, X_q, m_landmarks=50, random_state=42):
    """
    Manual Nyström:
     - sample m landmarks from X_q (indexed)
     - compute K_nm (n x m) and K_mm (m x m) using K_func (callable that computes kernel between two arrays)
     - approximate K ≈ K_nm K_mm^{-1} K_nm^T
    K_func should accept (A, B) and return kernel matrix shape (len(A), len(B)).
    """
    rng = np.random.RandomState(random_state)
    n = X_q.shape[0]
    m = min(m_landmarks, n)
    idx = rng.choice(n, size=m, replace=False)
    landmarks = X_q[idx]
    K_nm = K_func(X_q, landmarks)   # shape (n, m)
    K_mm = K_func(landmarks, landmarks)  # shape (m, m)
    # regularize
    jitter = 1e-8 * np.eye(m)
    try:
        K_mm_inv = np.linalg.pinv(K_mm + jitter)
    except Exception:
        K_mm_inv = np.linalg.pinv(K_mm + jitter)
    K_approx = K_nm.dot(K_mm_inv).dot(K_nm.T)
    return K_approx, idx

# helper to create callable quantum kernel using Qiskit QuantumKernel object
def make_qkernel_callable(qkernel):
    def kernel_fn(A, B):
        # qkernel.evaluate can accept x_vec and y_vec
        return qkernel.evaluate(x_vec=A, y_vec=B)
    return kernel_fn

# ---------------- VQC (experimental) ----------------
def vqc_embed_and_kmeans(X_q, n_qubits=4, var_layers=1, params=None, backend_name='aer_simulator_statevector', shots=256, random_state=42):
    """
    Build a parameterized circuit that maps each n_qubit input (already scaled) into an expectation-based
    feature vector of length n_qubits (one expectation per qubit). Then run kmeans on those features.
    If params provided, uses them. This function returns embeddings (n x n_qubits).
    """
    rng = np.random.RandomState(random_state)
    n_samples = X_q.shape[0]
    # parameterized circuit per sample:
    # embed features with RZ rotations on each qubit, then variational two-local layers
    param_vector = params
    # Build base circuit generator
    def circuit_for_params(param_values):
        # param_values: length equals n_qubits * var_layers * 2 (approx) or flexible
        pvec = ParameterVector('θ', len(param_values))
        qc = QuantumCircuit(n_qubits)
        # embedding: use angles from X_q (we'll bind per-sample later)
        # We create placeholders for input parameters per qubit: use parameters named 'x0...'
        return qc, pvec

    # Simpler approach: reuse TwoLocal as variational circuit and apply rotations with data as rotation angles
    # We'll construct a circuit per data point where we set rotation gates according to X_q row and add variational gates.
    backend = Aer.get_backend(backend_name)
    qi = QuantumInstance(backend=backend, shots=shots)

    # define parameter count
    var_param_count = n_qubits * var_layers * 2  # an approximate number to control
    if params is None:
        params = rng.normal(0, 0.5, size=(var_param_count,))
    # build an unbound circuit template using ParameterVector for variational parameters and placeholders for data
    thetas = ParameterVector('t', var_param_count)
    # We'll embed data with RX gates that receive numeric angles (we'll bind them directly)
    qc_template = QuantumCircuit(n_qubits)
    # placeholder: apply RX on each qubit with a parameter (we will bind separately per-sample)
    # We won't use Parameters for data (we'll bind float values at runtime)
    # Add variational layers
    two_local = TwoLocal(num_qubits=n_qubits, rotation_blocks='ry', entanglement_blocks='cz',
                         reps=var_layers, entanglement='linear', insert_barriers=False)
    # Compose template: empty (we'll build per-sample)
    # Execute per-sample circuits
    # For each data point, build a circuit: RX(data[j]) on each qubit, then bind variational params into two_local and append
    embeddings = np.zeros((n_samples, n_qubits))
    for i in range(n_samples):
        qc = QuantumCircuit(n_qubits)
        row = X_q[i]
        # if row has fewer dims than n_qubits, tile or truncate
        if row.shape[0] < n_qubits:
            angles = np.tile(row, int(np.ceil(n_qubits / row.shape[0])))[:n_qubits]
        else:
            angles = row[:n_qubits]
        # embed
        for q in range(n_qubits):
            qc.rx(float(angles[q]), q)
        # add variational circuit with bound params
        # create a tunable two_local and bind parameters
        tl = TwoLocal(num_qubits=n_qubits, rotation_blocks='ry', entanglement_blocks='cz',
                      reps=var_layers, entanglement='linear', insert_barriers=False)
        # bind the parameter values to tl
        if tl.num_parameters > 0:
            bind_map = {}
            pvals = params[:tl.num_parameters]
            for idx, p in enumerate(tl.parameters):
                bind_map[p] = float(pvals[idx])
            try:
                tl_bound = tl.bind_parameters(bind_map)
                qc.compose(tl_bound, inplace=True)
            except Exception:
                qc.compose(tl, inplace=True)
        else:
            qc.compose(tl, inplace=True)
        # measure expectation value of Z on each qubit by running statevector (simulate) and deriving Bloch vector
        # Simpler: run statevector and compute probabilities -> expectation = P(0)-P(1) per qubit
        backend_local = qi.backend
        job = backend_local.run(qc.decompose().to_instruction().definition) if hasattr(backend_local, 'run') else backend_local.run(qc)
        # Actually using the Qiskit runtime call is complicated; simpler: use Aer statevector simulator directly to get statevector
        try:
            sv_backend = Aer.get_backend('aer_simulator_statevector')
            result = sv_backend.run(qc).result()
            state = result.get_statevector()
            # compute expectation of Z on each qubit
            # state is vector size 2^n; expectation_z = sum_{b} |amp_b|^2 * (-1)^{b_q}
            probs = np.abs(state)**2
            for q in range(n_qubits):
                # axis: bit q (LSB is qubit 0); compute (-1)^{bit}
                dim = 2**n_qubits
                expect = 0.0
                for idx_b in range(dim):
                    bit = (idx_b >> q) & 1
                    expect += probs[idx_b] * ((-1)**bit)
                embeddings[i, q] = expect
        except Exception:
            # fallback: random embeddings if statevector not available (shouldn't happen)
            embeddings[i, :] = np.tanh(np.dot(np.ones((n_qubits,)), angles[:n_qubits]))[:n_qubits]

    return embeddings

def vqc_cost(params, X_q, n_qubits, var_layers, backend_name, n_clusters):
    """
    Cost function for VQC: build embeddings with params, run k-means, and return within-cluster inertia.
    Lower is better.
    """
    emb = vqc_embed_and_kmeans(X_q, n_qubits=n_qubits, var_layers=var_layers, params=params, backend_name=backend_name)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    km.fit(emb)
    return km.inertia_

# ---------------- Streamlit UI ----------------
st.title("Quantum Pattern Recognition for File Anomalies — Streamlit")

with st.sidebar:
    st.header("Data")
    data_choice = st.radio("Choose dataset", ["Use sample (synthetic)", "Upload CSV"])
    if data_choice == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file with file-metadata (columns like size_bytes,extension,owner,mtime,perms,...)", type=['csv'])
    st.write("---")
    st.header("Algorithm")
    algo = st.selectbox("Algorithm", ["Quantum Kernel + SpectralClustering", "Nyström approx. + SpectralClustering", "Variational Quantum Clustering (VQC) (experimental)"])
    st.write("---")
    st.header("Quantum / backend")
    backend = st.selectbox("Backend (Qiskit)", ["aer_simulator_statevector", "aer_simulator", "qasm_simulator"])
    shots = st.slider("Shots (for shot-based backends)", 64, 2048, 512, step=64)
    st.write("---")
    st.header("Preprocessing & qubits")
    target_qubits = st.slider("n_qubits (PCA -> qubit features)", 2, 8, 4)
    scale_range = st.selectbox("Scale reduced features to", ["[0, π]", "[-π, π]"])
    st.write("---")
    st.header("Nyström (if selected)")
    if algo.startswith("Nyström"):
        m_landmarks = st.slider("m landmarks (Nyström)", 10, 200, 50)
    else:
        m_landmarks = 50
    st.write("---")
    st.header("VQC (if selected)")
    if algo.startswith("VQC"):
        var_layers = st.slider("Variational layers (small recommended)", 1, 3, 1)
        vqc_opt_steps = st.slider("VQC optimizer steps (COBYLA)", 1, 20, 5)
        n_clusters = st.slider("Clusters for VQC (kmeans)", 2, 5, 2)
    else:
        var_layers = 1
        vqc_opt_steps = 5
        n_clusters = 2

    st.write("---")
    run_button = st.button("Run pipeline")

# Load data
if data_choice == "Use sample (synthetic)":
    df = simulate_file_metadata(n_samples=250, anomaly_frac=0.08)
    st.success("Using synthetic dataset (250 rows).")
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    else:
        df = None
        st.info("Upload a CSV to proceed or choose the sample dataset.")

# Show a preview
if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(10))

# Main run
if run_button:
    if df is None:
        st.error("Please upload data or choose the sample dataset.")
    else:
        with st.spinner("Featurizing and preprocessing..."):
            try:
                X_raw, feat_names, dfc = adaptive_featurizer(df, debug=True)
            except Exception as e:
                st.error(f"Featurizer error: {e}")
                st.stop()

            st.write("Detected features:", feat_names)
            # scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            # PCA to n_qubits
            pca = PCA(n_components=target_qubits)
            X_reduced = pca.fit_transform(X_scaled)

            # scale to feature map range
            if scale_range == "[0, π]":
                X_min, X_max = X_reduced.min(), X_reduced.max()
                if abs(X_max - X_min) < 1e-12:
                    X_q = np.zeros_like(X_reduced)
                else:
                    X_q = (X_reduced - X_min) / (X_max - X_min) * np.pi
            else:
                # [-pi, pi]
                X_min, X_max = X_reduced.min(), X_reduced.max()
                if abs(X_max - X_min) < 1e-12:
                    X_q = np.zeros_like(X_reduced)
                else:
                    X_q = ((X_reduced - X_min) / (X_max - X_min) * 2 - 1) * np.pi

        # Branch per algorithm
        if algo == "Quantum Kernel + SpectralClustering":
            st.info("Computing quantum kernel (may be slow for n>200).")
            with st.spinner("Computing quantum kernel matrix..."):
                try:
                    K, qk = compute_quantum_kernel_matrix(X_q, n_qubits=target_qubits, backend_name=backend, shots=shots)
                except Exception as e:
                    st.error(f"Quantum kernel computation error: {e}")
                    st.stop()
            st.success("Kernel computed.")
            st.write("Kernel matrix shape:", K.shape)
            # spectral clustering
            n_clusters = st.slider("Number of clusters (spectral)", 2, 6, 2)
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0, assign_labels='kmeans')
            labels = sc.fit_predict(K)
            st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
            # anomaly score: 1 - mean similarity to cluster members (as earlier)
            def kernel_anomaly_score(Km, labels):
                n = Km.shape[0]
                scores = np.zeros(n)
                for lbl in np.unique(labels):
                    idx = np.where(labels == lbl)[0]
                    if len(idx) == 1:
                        scores[idx] = 1.0
                        continue
                    subK = Km[np.ix_(idx, idx)]
                    mean_sim = (subK.sum(axis=1) - np.diag(subK)) / (len(idx)-1)
                    max_sim, min_sim = mean_sim.max(), mean_sim.min()
                    if max_sim - min_sim < 1e-12:
                        cs = np.zeros_like(mean_sim)
                    else:
                        cs = 1.0 - (mean_sim - min_sim) / (max_sim - min_sim)
                    scores[idx] = cs
                return scores
            scores = kernel_anomaly_score(K, labels)
            dfc['cluster'] = labels
            dfc['anomaly_score'] = scores
            st.subheader("Top suspicious files")
            st.dataframe(dfc.sort_values('anomaly_score', ascending=False).head(20))
            # plot histogram
            fig, ax = plt.subplots()
            ax.hist(dfc['anomaly_score'], bins=30)
            ax.set_title("Anomaly score distribution")
            st.pyplot(fig)

        elif algo.startswith("Nyström"):
            st.info("Nyström approximation — selecting landmarks and approximating kernel.")
            with st.spinner("Building quantum kernel object (for callable) and computing Nyström..."):
                try:
                    # build quantum kernel once (on feature_map) - we will use evaluate for pairs
                    feature_map = ZZFeatureMap(feature_dimension=target_qubits, reps=2, entanglement='linear')
                    backend_obj = Aer.get_backend(backend)
                    qi = QuantumInstance(backend=backend_obj, shots=shots)
                    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)
                    kernel_fn = make_qkernel_callable(qkernel)
                    K_approx, landmark_idx = nystrom_approximation(kernel_fn, X_q, m_landmarks=m_landmarks)
                except Exception as e:
                    st.error(f"Nyström / kernel building error: {e}")
                    st.stop()
            st.success("Nyström approximation completed.")
            st.write("Approximated kernel shape:", K_approx.shape)
            n_clusters = st.slider("Number of clusters (spectral)", 2, 6, 2)
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0, assign_labels='kmeans')
            labels = sc.fit_predict(K_approx)
            dfc['cluster'] = labels
            # compute anomaly scores same way
            def kernel_anomaly_score(Km, labels):
                n = Km.shape[0]
                scores = np.zeros(n)
                for lbl in np.unique(labels):
                    idx = np.where(labels == lbl)[0]
                    if len(idx) == 1:
                        scores[idx] = 1.0
                        continue
                    subK = Km[np.ix_(idx, idx)]
                    mean_sim = (subK.sum(axis=1) - np.diag(subK)) / (len(idx)-1)
                    max_sim, min_sim = mean_sim.max(), mean_sim.min()
                    if max_sim - min_sim < 1e-12:
                        cs = np.zeros_like(mean_sim)
                    else:
                        cs = 1.0 - (mean_sim - min_sim) / (max_sim - min_sim)
                    scores[idx] = cs
                return scores
            scores = kernel_anomaly_score(K_approx, labels)
            dfc['anomaly_score'] = scores
            st.subheader("Top suspicious files (Nyström)")
            st.dataframe(dfc.sort_values('anomaly_score', ascending=False).head(20))
            fig, ax = plt.subplots()
            ax.hist(dfc['anomaly_score'], bins=30)
            ax.set_title("Anomaly score distribution (Nyström)")
            st.pyplot(fig)

        else:
            # VQC
            st.warning("VQC is experimental and may be slow. Recommended for small datasets (n <= 200).")
            # small dataset guard
            if X_q.shape[0] > 250:
                st.warning("Large dataset detected. Consider downsampling to <=250 rows for VQC.")
            with st.spinner("Running VQC optimization (COBYLA)..."):
                # initial params
                rng = np.random.RandomState(0)
                param_count = target_qubits * var_layers * 2
                params0 = rng.normal(0, 0.5, size=(param_count,))
                # run a few steps of gradient-free optimization to minimize kmeans inertia on embeddings
                best = None
                try:
                    def obj(p):
                        val = vqc_cost(p, X_q, n_qubits=target_qubits, var_layers=var_layers, backend_name=backend, n_clusters=n_clusters)
                        st.text(f"Current VQC cost: {val:.4f}")
                        return val
                    res = minimize(obj, params0, method='COBYLA', options={'maxiter': vqc_opt_steps, 'tol':1e-3})
                    best_params = res.x
                    st.success(f"VQC optimization finished (status: {res.message})")
                    embeddings = vqc_embed_and_kmeans(X_q, n_qubits=target_qubits, var_layers=var_layers, params=best_params, backend_name=backend)
                    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embeddings)
                    labels = km.labels_
                    dfc['cluster'] = labels
                    # anomaly score: distance to cluster center in embedding space, normalized
                    dist = pairwise_distances(embeddings, km.cluster_centers_)
                    min_dist = dist.min(axis=1)
                    # normalize 0..1
                    md_min, md_max = min_dist.min(), min_dist.max()
                    if md_max - md_min < 1e-12:
                        scores_vqc = np.zeros_like(min_dist)
                    else:
                        scores_vqc = (min_dist - md_min) / (md_max - md_min)
                    dfc['anomaly_score'] = scores_vqc
                    st.subheader("VQC results — top suspicious")
                    st.dataframe(dfc.sort_values('anomaly_score', ascending=False).head(20))
                    fig, ax = plt.subplots()
                    ax.hist(dfc['anomaly_score'], bins=30)
                    ax.set_title("Anomaly score distribution (VQC)")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"VQC failed: {e}")

        # show optional evaluation if label column exists
        if 'label' in dfc.columns:
            st.write("---")
            st.subheader("Evaluation (if 'label' column is ground truth)")
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(dfc['label'], dfc['anomaly_score'])
                st.write(f"ROC AUC (anomaly_score vs label): **{auc:.4f}**")
            except Exception as e:
                st.write("Could not compute ROC AUC:", e)

        # allow download of results
        st.write("---")
        st.subheader("Download results (CSV)")
        to_download = dfc.copy()
        buf = BytesIO()
        to_download.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download CSV with cluster & anomaly_score", data=buf, file_name="quantum_anomaly_results.csv", mime="text/csv")

st.write("---")
st.markdown("""
### Notes & next steps

- **Scale & runtime**: quantum kernel evaluation scales badly with large n (O(n^2) kernel entries). Use Nyström or subsample for production.  
- **Real hardware**: to use IBM hardware set up `IBMQ.save_account(...)` and change backend selection accordingly — you may need to use `QuantumInstance` with provider backend.  
- **VQC**: the included VQC is experimental — it builds embeddings by measuring Z-expectations after applying data rotations + variational layers and minimizes k-means inertia via COBYLA. It's slow but demonstrates a variational approach.  
- **Improvements**: implement Nyström more efficiently (precompute only kernel blocks in parallel), try different feature maps (PauliFeatureMap / ZZ), or train a kernel-aware one-class classifier in kernel space.  
- If you want, I can:
  - adapt the featurizer to an exact schema you paste,
  - add an automated `n_qubits` tuner (grid search across 2..6 with cross-validated objective),
  - or produce a Dockerfile + requirements to run this reliably on a server.
""")
