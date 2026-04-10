"""
NumPy — Laboratorio 01: Basi
=============================
Esercizi di livello base: creazione array, attributi, operazioni semplici.
Completa ogni TODO e verifica con gli assert.
Esegui con: python 01_basics.py

Setup ambiente (venv + dipendenze): vedi ../../SETUP.md
"""

import numpy as np

print("=" * 50)
print("SEZIONE 1 — Creazione array")
print("=" * 50)

# --- Es. 1 ---
# Crea un array di interi da 0 a 9 incluso.
arr = None  # TODO
assert arr is not None and list(arr) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "Usa np.arange"
print("✓ Es.1 — array 0..9:", arr)

# --- Es. 2 ---
# Crea un array di 5 zeri con dtype float32.
zeros = None  # TODO
assert zeros is not None and zeros.dtype == np.float32 and len(zeros) == 5, "Usa np.zeros con dtype"
print("✓ Es.2 — zeros float32:", zeros)

# --- Es. 3 ---
# Crea una matrice identità 4×4.
eye = None  # TODO
assert eye is not None and eye.shape == (4, 4) and eye[0, 0] == 1 and eye[0, 1] == 0
print("✓ Es.3 — eye 4x4:\n", eye)

# --- Es. 4 ---
# Crea 6 punti equispaziati tra 0 e 1 inclusi.
pts = None  # TODO
assert pts is not None and len(pts) == 6 and pts[0] == 0.0 and pts[-1] == 1.0
print("✓ Es.4 — linspace:", pts)

print("\n" + "=" * 50)
print("SEZIONE 2 — Attributi")
print("=" * 50)

mat = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]], dtype=np.float64)

# --- Es. 5 ---
# Quante righe ha mat?
n_rows = None  # TODO
assert n_rows == 3
print("✓ Es.5 — righe:", n_rows)

# --- Es. 6 ---
# Quanti elementi totali ha mat?
n_elements = None  # TODO
assert n_elements == 12
print("✓ Es.6 — elementi:", n_elements)

# --- Es. 7 ---
# Quanti byte occupa mat in memoria?
# float64 = 8 byte/elemento
n_bytes = None  # TODO
assert n_bytes == 96  # 12 * 8
print("✓ Es.7 — nbytes:", n_bytes)

print("\n" + "=" * 50)
print("SEZIONE 3 — Operazioni element-wise")
print("=" * 50)

a = np.array([1.0, 4.0, 9.0, 16.0])

# --- Es. 8 ---
# Calcola la radice quadrata di ogni elemento.
result = None  # TODO
assert result is not None and np.allclose(result, [1.0, 2.0, 3.0, 4.0])
print("✓ Es.8 — sqrt:", result)

# --- Es. 9 ---
# Moltiplica ogni elemento per 3 e poi sottrai 1.
result = None  # TODO
assert result is not None and np.allclose(result, [2.0, 11.0, 26.0, 47.0])
print("✓ Es.9 — *3 -1:", result)

# --- Es. 10 ---
# Dati questi due array, calcola il loro prodotto scalare (dot product).
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
dot = None  # TODO
assert dot == 32.0  # 1*4 + 2*5 + 3*6
print("✓ Es.10 — dot product:", dot)

print("\n" + "=" * 50)
print("SEZIONE 4 — Indexing e slicing")
print("=" * 50)

arr = np.arange(20)   # [0..19]

# --- Es. 11 ---
# Estrai gli elementi dall'indice 5 al 10 incluso.
sliced = None  # TODO
assert list(sliced) == [5, 6, 7, 8, 9, 10]
print("✓ Es.11 — slice [5:11]:", sliced)

# --- Es. 12 ---
# Estrai tutti gli elementi pari dell'array.
evens = None  # TODO
assert list(evens) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
print("✓ Es.12 — pari:", evens)

# --- Es. 13 ---
# Da questa matrice, estrai la seconda colonna come array 1D.
mat = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])
col = None  # TODO
assert list(col) == [20, 50, 80]
print("✓ Es.13 — colonna 1:", col)

# --- Es. 14 ---
# Usa boolean indexing per estrarre i valori > 50 dalla matrice sopra.
big = None  # TODO
assert list(big) == [60, 70, 80, 90]
print("✓ Es.14 — valori > 50:", big)

print("\n" + "=" * 50)
print("SEZIONE 5 — Reshape")
print("=" * 50)

flat = np.arange(24)

# --- Es. 15 ---
# Trasforma flat in una matrice 4×6.
mat = None  # TODO
assert mat is not None and mat.shape == (4, 6)
print("✓ Es.15 — reshape (4,6):\n", mat)

# --- Es. 16 ---
# Trasforma flat in una matrice con 3 righe, lasciando che NumPy
# calcoli il numero di colonne automaticamente.
mat2 = None  # TODO
assert mat2 is not None and mat2.shape == (3, 8)
print("✓ Es.16 — reshape (3,-1):", mat2.shape)

# --- Es. 17 ---
# Trasponi mat (4×6) → dovrebbe diventare (6×4).
transposed = None  # TODO
assert transposed is not None and transposed.shape == (6, 4)
print("✓ Es.17 — transpose:", transposed.shape)

print("\n✅ Tutti gli esercizi base completati!")