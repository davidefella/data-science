"""
NumPy — Laboratorio 03: Avanzato
==================================
Implementazioni applicate: normalizzazione, distanze, operazioni su dati reali.
Tutti i concetti usati qui (array, broadcasting, prodotto matriciale, statistiche)
sono già stati trattati nella demo e negli esercizi precedenti.
Niente framework ML — solo NumPy puro.

Esegui con: python 03_advanced.py

Setup ambiente (venv + dipendenze): vedi ../../SETUP.md
"""

import numpy as np

print("=" * 50)
print("SEZIONE 1 — Normalizzazione da zero")
print("=" * 50)

# Dataset: 5 campioni, 3 feature con scale molto diverse.
# Tipico problema pre-ML: le feature su scale diverse fanno sì che
# alcuni algoritmi (es. k-NN) siano dominati dalle feature con range ampi.
X = np.array([[100.0, 0.001, 50.0],
              [200.0, 0.002, 30.0],
              [150.0, 0.003, 70.0],
              [300.0, 0.001, 20.0],
              [250.0, 0.004, 60.0]])

# --- Es. 1 ---
# Implementa Z-score standardization da zero (senza sklearn).
# Formula per ogni colonna j: (x - mean_j) / std_j
# Dopo la standardizzazione ogni colonna deve avere media ≈ 0 e std ≈ 1.
def z_score(X: np.ndarray) -> np.ndarray:
    """
    Standardizza X per colonna: media 0, std 1.
    Args:
        X: array (n_samples, n_features)
    Returns:
        X_scaled: array (n_samples, n_features)
    """
    # TODO
    pass

X_scaled = z_score(X)
assert X_scaled is not None and X_scaled.shape == X.shape
assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
print("✓ Es.1 — Z-score (medie):", np.round(X_scaled.mean(axis=0), 6))
print("         Z-score (std):  ", np.round(X_scaled.std(axis=0), 6))


# --- Es. 2 ---
# Implementa Min-Max scaling da zero.
# Formula: (x - min) / (max - min) → porta ogni colonna in [0, 1].
def min_max_scale(X: np.ndarray) -> np.ndarray:
    """
    Scala X per colonna in [0, 1].
    Args:
        X: array (n_samples, n_features)
    Returns:
        X_scaled: array (n_samples, n_features)
    """
    # TODO
    pass

X_mm = min_max_scale(X)
assert X_mm is not None and X_mm.shape == X.shape
assert np.allclose(X_mm.min(axis=0), 0.0)
assert np.allclose(X_mm.max(axis=0), 1.0)
print("✓ Es.2 — Min-Max (min): ", X_mm.min(axis=0))
print("         Min-Max (max): ", X_mm.max(axis=0))


print("\n" + "=" * 50)
print("SEZIONE 2 — Distanze")
print("=" * 50)

# --- Es. 3 ---
# Implementa la distanza euclidea tra due vettori.
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distanza euclidea tra i vettori a e b.
    Formula: sqrt(sum((a - b)^2))
    """
    # TODO
    pass

a = np.array([0.0, 0.0])
b = np.array([3.0, 4.0])
dist = euclidean_distance(a, b)
assert np.isclose(dist, 5.0)   # triangolo 3-4-5
print("✓ Es.3 — distanza euclidea [0,0]-[3,4]:", dist)


# --- Es. 4 ---
# Calcola la matrice delle distanze euclidee tra tutti i punti
# di un dataset. Dati n punti, il risultato è una matrice (n, n)
# dove result[i, j] = distanza tra punto i e punto j.
# La diagonale deve essere 0 (distanza di ogni punto da se stesso).
# La matrice deve essere simmetrica.
#
# NON usare loop doppi — usa broadcasting.
# Hint: ||a - b||² = sum((a - b)²) su axis=-1

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Matrice delle distanze euclidee tra tutti i punti di X.
    Args:
        X: array (n_points, n_features)
    Returns:
        D: array (n_points, n_points)
    """
    # Hint: X[:, np.newaxis, :] ha shape (n, 1, d)
    #       X[np.newaxis, :, :] ha shape (1, n, d)
    #       la differenza ha shape (n, n, d) per broadcasting
    # TODO
    pass

points = np.array([[0.0, 0.0],
                   [3.0, 4.0],
                   [6.0, 0.0]])

D = pairwise_distances(points)
assert D is not None and D.shape == (3, 3)
assert np.allclose(np.diag(D), 0.0)      # distanza con se stesso = 0
assert np.allclose(D, D.T)               # simmetrica
assert np.isclose(D[0, 1], 5.0)          # [0,0] → [3,4] = 5
assert np.isclose(D[0, 2], 6.0)          # [0,0] → [6,0] = 6
print("✓ Es.4 — matrice distanze:\n", np.round(D, 3))


print("\n" + "=" * 50)
print("SEZIONE 3 — Operazioni su dati reali")
print("=" * 50)

# Simuliamo un piccolo dataset: 100 campioni, 4 feature, 2 classi
rng = np.random.default_rng(seed=42)

n_samples = 100
X_class0 = rng.normal(loc=0.0, scale=1.0, size=(n_samples // 2, 4))
X_class1 = rng.normal(loc=2.0, scale=1.0, size=(n_samples // 2, 4))
X_full    = np.vstack([X_class0, X_class1])   # (100, 4)
y_full    = np.array([0] * 50 + [1] * 50)     # etichette

# --- Es. 5 ---
# Calcola la media di ogni feature separatamente per classe 0 e classe 1.
# Risultato: due array di shape (4,), uno per classe.
mean_class0 = None  # TODO: media feature su campioni dove y == 0
mean_class1 = None  # TODO: media feature su campioni dove y == 1

assert mean_class0 is not None and mean_class0.shape == (4,)
assert mean_class1 is not None and mean_class1.shape == (4,)
# classe 0 centrata attorno a 0, classe 1 attorno a 2
assert all(mean_class0 < 1.0)
assert all(mean_class1 > 1.0)
print("✓ Es.5 — media classe 0:", np.round(mean_class0, 3))
print("         media classe 1:", np.round(mean_class1, 3))


# --- Es. 6 ---
# Train/Test split manuale: dividi X_full e y_full in
# 80% training e 20% test, mantenendo l'ordine originale (no shuffle).
split_idx = int(len(X_full) * 0.8)

X_train, X_test = None, None  # TODO
y_train, y_test = None, None  # TODO

assert X_train is not None and X_train.shape == (80, 4)
assert X_test  is not None and X_test.shape  == (20, 4)
assert len(y_train) == 80
assert len(y_test)  == 20
print("✓ Es.6 — train shape:", X_train.shape, "| test shape:", X_test.shape)


# --- Es. 7 ---
# Standardizza X_train con Z-score, poi applica la STESSA trasformazione
# (stessa media e std del training set) a X_test.
# IMPORTANTE: non calcolare media/std sul test set — è un data leakage.
train_mean = None  # TODO: media per colonna su X_train
train_std  = None  # TODO: std per colonna su X_train

X_train_scaled = None  # TODO: standardizza X_train
X_test_scaled  = None  # TODO: applica la stessa trasformazione a X_test

assert X_train_scaled is not None
assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-10)
assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-10)
# Il test set NON avrà media esattamente 0 — è corretto
print("✓ Es.7 — train scaled mean:", np.round(X_train_scaled.mean(axis=0), 3))
print("         test  scaled mean:", np.round(X_test_scaled.mean(axis=0), 3))


print("\n" + "=" * 50)
print("SEZIONE 4 — Algebra lineare")
print("=" * 50)

# --- Es. 8 ---
# Data una matrice quadrata A, verifica che A @ inv(A) ≈ identità.
A = np.array([[2.0, 1.0],
              [5.0, 3.0]])

A_inv    = None  # TODO: inversa di A con np.linalg.inv
identity = None  # TODO: A @ A_inv

assert A_inv is not None
assert identity is not None
assert np.allclose(identity, np.eye(2), atol=1e-10)
print("✓ Es.8 — A @ A_inv ≈ I:\n", np.round(identity, 6))


# --- Es. 9 ---
# Calcola autovalori e autovettori di una matrice simmetrica.
# Le matrici simmetriche hanno autovalori reali — proprietà usata in PCA.
M = np.array([[4.0, 2.0],
              [2.0, 3.0]])

eigenvalues  = None  # TODO
eigenvectors = None  # TODO
# Hint: np.linalg.eig o np.linalg.eigh (più stabile per simmetriche)

assert eigenvalues  is not None and len(eigenvalues) == 2
assert eigenvectors is not None and eigenvectors.shape == (2, 2)
# Verifica: M @ v = λ * v per ogni coppia (λ, v)
for i in range(2):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]
    assert np.allclose(M @ v, lam * v, atol=1e-10)
print("✓ Es.9 — autovalori:", np.round(eigenvalues, 4))
print("         autovettori:\n", np.round(eigenvectors, 4))


# --- Es. 10 ---
# La norma L2 di un vettore è la sua "lunghezza" nello spazio euclideo.
# Un vettore normalizzato (norma = 1) è detto versore.
# Normalizza ogni riga di questa matrice a norma 1.
# Hint: np.linalg.norm con keepdims=True

V = np.array([[3.0, 4.0],
              [1.0, 0.0],
              [0.0, 2.0]])

V_normalized = None  # TODO
assert V_normalized is not None and V_normalized.shape == (3, 2)
norms = np.linalg.norm(V_normalized, axis=1)
assert np.allclose(norms, 1.0, atol=1e-10)
print("✓ Es.10 — righe normalizzate a norma 1:\n", np.round(V_normalized, 4))
print("          norme:", np.round(norms, 6))


print("\n✅ Tutti gli esercizi avanzati completati!")