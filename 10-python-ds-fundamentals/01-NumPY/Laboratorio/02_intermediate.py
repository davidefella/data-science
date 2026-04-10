"""
NumPy — Laboratorio 02: Intermedio
====================================
Broadcasting, operazioni statistiche per asse, prodotto matriciale.
Completa ogni TODO e verifica con gli assert.
Esegui con: python 02_intermediate.py

Setup ambiente (venv + dipendenze): vedi ../../SETUP.md
"""

import numpy as np

print("=" * 50)
print("SEZIONE 1 — Statistiche per asse")
print("=" * 50)

# Dataset: 4 campioni, 3 feature
# Ogni riga è un campione, ogni colonna è una feature.
data = np.array([[2.0,  4.0,  1.0],
                 [6.0,  8.0,  3.0],
                 [10.0, 2.0,  5.0],
                 [4.0,  6.0,  7.0]])

# --- Es. 1 ---
# Calcola la media di ogni feature (colonna).
# Risultato atteso: shape (3,)
col_means = None  # TODO
assert col_means is not None and col_means.shape == (3,)
assert np.allclose(col_means, [5.5, 5.0, 4.0])
print("✓ Es.1 — media per colonna:", col_means)

# --- Es. 2 ---
# Calcola la deviazione standard di ogni campione (riga).
# Risultato atteso: shape (4,)
row_stds = None  # TODO
assert row_stds is not None and row_stds.shape == (4,)
print("✓ Es.2 — std per riga:", np.round(row_stds, 4))

# --- Es. 3 ---
# Trova l'indice della feature con valore massimo per ogni campione.
# Hint: argmax sull'asse giusto.
# Risultato atteso: shape (4,) con indici 0-2
argmax_per_row = None  # TODO
assert argmax_per_row is not None and argmax_per_row.shape == (4,)
assert list(argmax_per_row) == [1, 1, 0, 2]
print("✓ Es.3 — argmax per riga:", argmax_per_row)

print("\n" + "=" * 50)
print("SEZIONE 2 — Broadcasting")
print("=" * 50)

# --- Es. 4 ---
# Sottrai la media di ogni colonna da data (centratura dei dati).
# Non usare loop. Usa broadcasting.
centered = None  # TODO
assert centered is not None and centered.shape == data.shape
assert np.allclose(centered.mean(axis=0), [0.0, 0.0, 0.0])
print("✓ Es.4 — data centrata:\n", centered)

# --- Es. 5 ---
# Normalizza data con Min-Max scaling (porta ogni colonna in [0, 1]).
# Formula: (x - min) / (max - min)
# Ogni colonna deve avere min=0 e max=1 dopo la normalizzazione.
normalized = None  # TODO
assert normalized is not None and normalized.shape == data.shape
assert np.allclose(normalized.min(axis=0), [0.0, 0.0, 0.0])
assert np.allclose(normalized.max(axis=0), [1.0, 1.0, 1.0])
print("✓ Es.5 — Min-Max normalized:\n", np.round(normalized, 3))

# --- Es. 6 ---
# Aggiungi il vettore bias b a ogni riga della matrice A.
# Non usare loop.
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
b = np.array([10.0, 20.0, 30.0])

result = None  # TODO
assert result is not None and result.shape == (2, 3)
assert np.allclose(result, [[11, 22, 33], [14, 25, 36]])
print("✓ Es.6 — A + b broadcast:\n", result)

print("\n" + "=" * 50)
print("SEZIONE 3 — Prodotto matriciale")
print("=" * 50)

# --- Es. 7 ---
# Dato un batch X di 3 campioni con 4 feature ciascuno,
# e una matrice W di pesi (4 input → 2 output),
# calcola l'output del layer: X @ W
# Risultato atteso: shape (3, 2)
X = np.array([[1.0, 0.0, 2.0, 1.0],
              [0.0, 1.0, 1.0, 2.0],
              [2.0, 2.0, 0.0, 1.0]])

W = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0],
              [0.0, 1.0]])

output = None  # TODO
assert output is not None and output.shape == (3, 2)
assert np.allclose(output, [[3.0, 3.0], [1.0, 4.0], [2.0, 3.0]])
print("✓ Es.7 — X @ W:\n", output)

# --- Es. 8 ---
# Aggiungi il bias vector b all'output precedente.
# b ha un valore per ogni output (shape: (2,))
b = np.array([0.5, -0.5])
output_with_bias = None  # TODO
assert output_with_bias is not None and output_with_bias.shape == (3, 2)
assert np.allclose(output_with_bias, [[3.5, 2.5], [1.5, 3.5], [2.5, 2.5]])
print("✓ Es.8 — output + bias:\n", output_with_bias)

# --- Es. 9 ---
# Calcola X @ X.T (gram matrix): misura la similarità tra ogni coppia
# di campioni. Risultato atteso: matrice (3, 3) simmetrica.
gram = None  # TODO
assert gram is not None and gram.shape == (3, 3)
assert np.allclose(gram, gram.T)   # deve essere simmetrica
print("✓ Es.9 — Gram matrix (X @ X.T):\n", gram)

print("\n" + "=" * 50)
print("SEZIONE 4 — Boolean indexing avanzato")
print("=" * 50)

scores = np.array([0.92, 0.45, 0.88, 0.31, 0.76, 0.55, 0.93, 0.22])
labels = np.array([1,    0,    1,    0,    1,    0,    1,    0   ])
# scores = probabilità predette da un classificatore binario
# labels = etichette reali (1 = positivo, 0 = negativo)

# --- Es. 10 ---
# Estrai i valori di scores dove il label è 1 (campioni positivi).
positive_scores = None  # TODO
assert list(positive_scores) == [0.92, 0.88, 0.76, 0.93]
print("✓ Es.10 — scores positivi:", positive_scores)

# --- Es. 11 ---
# Conta quanti campioni hanno score > 0.5 E label == 0
# (falsi positivi ad alta confidenza).
fp_count = None  # TODO
assert fp_count == 1   # solo 0.55 ha label=0 e score>0.5
print("✓ Es.11 — falsi positivi ad alta confidenza:", fp_count)

# --- Es. 12 ---
# Applica una soglia di 0.5: predici 1 se score > 0.5, altrimenti 0.
# Poi conta quante predizioni coincidono con i label reali (accuracy).
predictions = None   # TODO: array di 0/1 basato sulla soglia
n_correct   = None   # TODO: numero di predizioni corrette
accuracy    = None   # TODO: n_correct / len(labels)

assert list(predictions) == [1, 0, 1, 0, 1, 1, 1, 0]
assert n_correct == 7
assert np.isclose(accuracy, 0.875)
print("✓ Es.12 — accuracy:", accuracy)

print("\n" + "=" * 50)
print("SEZIONE 5 — Operazioni su array")
print("=" * 50)

# --- Es. 13 ---
# Stack verticale: combina due array riga in una matrice.
r1 = np.array([1.0, 2.0, 3.0])
r2 = np.array([4.0, 5.0, 6.0])
stacked = None  # TODO: shape (2, 3)
assert stacked is not None and stacked.shape == (2, 3)
print("✓ Es.13 — vstack:\n", stacked)

# --- Es. 14 ---
# Dato un array con valori NaN, sostituisci i NaN con la media
# degli altri valori dell'array.
arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
# Hint: np.isnan(), np.nanmean()
filled = arr.copy()  # lavora su una copia
# TODO: sostituisci i NaN
assert not np.any(np.isnan(filled))
assert np.isclose(filled[1], 3.0)   # media di [1, 3, 5]
assert np.isclose(filled[3], 3.0)
print("✓ Es.14 — NaN riempiti:", filled)

print("\n✅ Tutti gli esercizi intermedi completati!")