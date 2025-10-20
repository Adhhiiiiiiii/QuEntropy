# app.py
import streamlit as st
import numpy as np
import sqlite3
import json
import os
from datetime import datetime
from quantum_entropy import profile_file, sha256_hex, classical_entropy_normalized

DB_PATH = "profiles.db"

# initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            sha256 TEXT,
            created_at TEXT,
            params TEXT,
            profile_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

st.set_page_config(page_title="Quantum Entropy Profiler", layout="wide")
st.title("Quantum Entropy Profiler — File Tampering Detection (Simulated)")

st.markdown("""
Upload a file, split it into chunks, and compute chunk-level entropy.
- **Classical entropy** uses byte-frequency Shannon entropy.
- **Quantum entropy proxy** simulates simple quantum circuits (Qiskit Aer) and computes measurement distribution entropy.
If Qiskit isn't installed the quantum mode will show an error and you can use classical mode alone.
""")

uploaded = st.file_uploader("Upload file (any type)", type=None)
col1, col2, col3 = st.columns(3)
with col1:
    chunk_size = st.number_input("Chunk size (bytes)", min_value=8, max_value=65536, value=256, step=8)
with col2:
    mode = st.selectbox("Mode", options=['both', 'classical', 'quantum'])
with col3:
    st.write("Quantum params (ignored if classical only)")
    n_qubits = st.slider("n_qubits", min_value=2, max_value=6, value=4)
    shots = st.slider("shots", min_value=256, max_value=8192, value=1024, step=256)

if uploaded:
    data = uploaded.read()
    st.write(f"File: **{uploaded.name}** — {len(data)} bytes")
    sha = sha256_hex(data)
    st.write(f"SHA-256: `{sha}`")

    if st.button("Profile file"):
        with st.spinner("Profiling... (quantum runs may take time)"):
            profiles = profile_file(data, chunk_size=chunk_size, mode=mode, n_qubits=n_qubits, shots=shots)

        # extract arrays for plotting
        classical_vals = []
        quantum_vals = []
        offsets = []
        for p in profiles:
            offsets.append(p['offset'])
            classical_vals.append(p.get('classical', {}).get('entropy_normalized', None))
            qv = p.get('quantum', {}).get('entropy_normalized', None)
            # if quantum errored put None
            quantum_vals.append(qv)

        st.subheader("Entropy profile")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        x = np.arange(len(offsets))
        if any(v is not None for v in classical_vals):
            ax.plot(x, [v if v is not None else np.nan for v in classical_vals],
                    label='Classical entropy (normalized)', color='tab:blue', marker='o')
        if any(v is not None for v in quantum_vals):
            ax.plot(x, [v if v is not None else np.nan for v in quantum_vals],
                    label='Quantum entropy proxy (normalized)', color='tab:orange', marker='x')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Chunk index')
        ax.set_ylabel('Normalized entropy (0..1)')
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # table of suspicious chunks
        st.subheader("Suspicious chunks (high entropy areas)")
        # define threshold sliders
        col_a, col_b = st.columns(2)
        with col_a:
            classical_thresh = st.slider("Classical threshold", 0.0, 1.0, 0.8)
        with col_b:
            quantum_thresh = st.slider("Quantum threshold", 0.0, 1.0, 0.8)

        suspicious = []
        for p in profiles:
            idx = p['index']
            off = p['offset']
            cs = p.get('classical', {})
            qs = p.get('quantum', {})
            cs_val = cs.get('entropy_normalized') if cs else None
            qs_val = qs.get('entropy_normalized') if qs and 'entropy_normalized' in qs else None
            flag = False
            reason = []
            if cs_val is not None and cs_val >= classical_thresh:
                flag = True
                reason.append(f"classical={cs_val:.3f}")
            if qs_val is not None and qs_val >= quantum_thresh:
                flag = True
                reason.append(f"quantum={qs_val:.3f}")
            if flag:
                suspicious.append({
                    'index': idx,
                    'offset': off,
                    'size': p['size'],
                    'reasons': ', '.join(reason)
                })
        if suspicious:
            import pandas as pd
            st.table(pd.DataFrame(suspicious))
        else:
            st.info("No chunks exceed the chosen thresholds.")

        # Save profile to DB
        st.subheader("Save profile")
        case_name = st.text_input("Case / label (optional)", value=os.path.splitext(uploaded.name)[0])
        if st.button("Save profile to DB"):
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            metadata = {
                'file_name': uploaded.name,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'chunk_size': chunk_size,
                'mode': mode,
                'n_qubits': n_qubits,
                'shots': shots,
                'case_name': case_name
            }
            params_json = json.dumps(metadata)
            profile_json = json.dumps({'profiles': profiles})
            c.execute('INSERT INTO profiles (filename, sha256, created_at, params, profile_json) VALUES (?, ?, ?, ?, ?)',
                      (uploaded.name, sha, datetime.utcnow().isoformat() + 'Z', params_json, profile_json))
            conn.commit()
            conn.close()
            st.success("Saved profile to local DB (profiles.db)")

        # Export JSON
        st.subheader("Export / Download")
        from io import BytesIO
        export_obj = {
            'file_name': uploaded.name,
            'sha256': sha,
            'chunk_size': chunk_size,
            'mode': mode,
            'n_qubits': n_qubits,
            'shots': shots,
            'profiles': profiles,
            'exported_at': datetime.utcnow().isoformat() + 'Z'
        }
        export_bytes = json.dumps(export_obj, indent=2).encode('utf-8')
        st.download_button("Download profile JSON", data=export_bytes, file_name=f"{uploaded.name}_entropy_profile.json")

# View saved profiles
st.sidebar.header("Saved profiles")
if st.sidebar.button("List saved profiles"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, filename, sha256, created_at, params FROM profiles ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=['id', 'filename', 'sha256', 'created_at', 'params'])
        st.sidebar.table(df)
    else:
        st.sidebar.info("No saved profiles found.")
