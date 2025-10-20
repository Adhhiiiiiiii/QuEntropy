# quantum_entropy.py
import numpy as np
import math
import hashlib
import json

# Qiskit imports are optional (quantum mode). Import lazily to allow classical-only mode to run.
def try_import_qiskit():
    try:
        from qiskit import QuantumCircuit, Aer, transpile
        from qiskit.circuit.library import CSwapGate
        return QuantumCircuit, Aer, transpile, None
    except Exception as e:
        return None, None, None, e

QISKIT_IMPORTS = try_import_qiskit()

# --------------------
# File chunking utils
# --------------------
def chunk_bytes(data_bytes: bytes, chunk_size: int):
    """Yield successive chunks (bytes). Last chunk may be shorter."""
    for i in range(0, len(data_bytes), chunk_size):
        yield data_bytes[i:i+chunk_size]

# --------------------
# Classical Shannon entropy (byte-level)
# --------------------
def classical_shannon_entropy(chunk: bytes):
    """Compute Shannon entropy of byte distribution in chunk. Returns value in bits (0..8)."""
    if len(chunk) == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log2(probs))
    return float(H)

def classical_entropy_normalized(chunk: bytes):
    """Normalized to 0..1 where 1 means maximum entropy (8 bits)"""
    H = classical_shannon_entropy(chunk)
    return H / 8.0

# --------------------
# Simple angle encoding
# --------------------
def bytes_to_angles(chunk: bytes, n_qubits: int):
    """
    Map a chunk to n_qubits rotation angles in [0, pi].
    Strategy: sample or repeat bytes to length n_qubits and scale 0..255 -> 0..pi
    """
    if len(chunk) == 0:
        chunk = b'\x00'
    arr = list(chunk)
    if len(arr) < n_qubits:
        times = (n_qubits + len(arr) - 1) // len(arr)
        arr = (arr * times)[:n_qubits]
    else:
        # pick representative bytes spaced across chunk for more coverage
        step = max(1, len(arr) // n_qubits)
        arr = [arr[i*step] for i in range(n_qubits)]
    angles = [(b / 255.0) * math.pi for b in arr]
    return angles

# --------------------
# Quantum circuit based entropy proxy
# --------------------
def quantum_entropy_proxy(chunk: bytes, n_qubits: int = 4, shots: int = 1024):
    """
    Using Qiskit Aer (simulator), build a product state via RY rotations per qubit, then apply Hadamard to each qubit,
    measure all qubits many times. Compute Shannon entropy of measurement probs and normalize to [0,1].
    Returns dict with results or error if Qiskit not available.
    """
    QC, AerModule, transpile, qiskit_error = QISKIT_IMPORTS
    if QC is None:
        raise RuntimeError(f"Qiskit not available: {qiskit_error}")

    # build angles
    angles = bytes_to_angles(chunk, n_qubits)

    # Prepare circuit: n_qubits, measure all
    qc = QC(n_qubits, n_qubits)
    # prepare product state via RY on each qubit
    for i, ang in enumerate(angles):
        qc.ry(ang, i)
    # apply Hadamard to all qubits to create interference
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))

    backend = AerModule.get_backend('aer_simulator')
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # compute normalized Shannon entropy of measurement distribution
    # counts keys are bitstrings; convert to probabilities
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    H = -np.sum([p * math.log2(p) for p in probs if p > 0])
    # Maximum possible entropy is n_qubits bits (log2(2^n) = n)
    H_norm = float(H / n_qubits)  # normalized 0..1
    return {
        'n_qubits': n_qubits,
        'shots': shots,
        'counts': counts,
        'entropy_bits': H,
        'entropy_normalized': H_norm
    }

# --------------------
# High-level profiling
# --------------------
def profile_file(data_bytes: bytes, chunk_size: int = 256, mode: str = 'both',
                 n_qubits: int = 4, shots: int = 1024):
    """
    mode: 'quantum', 'classical', or 'both'
    Returns a list of chunk profiles.
    Each profile: {index, offset, size, classical: {entropy_bits, entropy_norm}, quantum: {...}}
    """
    profiles = []
    for idx, chunk in enumerate(chunk_bytes(data_bytes, chunk_size)):
        p = {
            'index': idx,
            'offset': idx * chunk_size,
            'size': len(chunk)
        }
        if mode in ('classical', 'both'):
            H_bits = classical_shannon_entropy(chunk)
            p['classical'] = {
                'entropy_bits': H_bits,
                'entropy_normalized': H_bits / 8.0
            }
        if mode in ('quantum', 'both'):
            try:
                qres = quantum_entropy_proxy(chunk, n_qubits=n_qubits, shots=shots)
                p['quantum'] = qres
            except Exception as e:
                p['quantum'] = {'error': str(e)}
        profiles.append(p)
    return profiles

# --------------------
# Utilities
# --------------------
def sha256_hex(data_bytes: bytes):
    import hashlib
    h = hashlib.sha256()
    h.update(data_bytes)
    return h.hexdigest()

def profiles_to_json(profiles, metadata=None):
    out = {
        'metadata': metadata or {},
        'profiles': profiles
    }
    return json.dumps(out, indent=2)
