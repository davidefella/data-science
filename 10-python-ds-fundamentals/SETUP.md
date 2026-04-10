# Setup ambiente — Python DS Fundamentals

Istruzioni per preparare l'ambiente Python necessario a eseguire gli script
e gli esercizi dei moduli in `10-python-ds-fundamentals/`.

## 1. Creare il virtual environment

Dalla cartella radice del progetto (`data-science/`):

```bash
python -m venv .venv
```

Questo crea una cartella `.venv/` isolata che conterrà la versione di Python
e i pacchetti installati per questo progetto, senza interferire con il sistema.

## 2. Attivare il venv

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell o cmd):**
```powershell
.venv\Scripts\activate
```

Quando il venv è attivo vedrai `(.venv)` davanti al prompt del terminale.
Per uscire dal venv in qualsiasi momento:

```bash
deactivate
```

## 3. Installare le dipendenze

Per il modulo **01-NumPY** serve solo NumPy:

```bash
pip install numpy
```

Moduli successivi aggiungeranno altre librerie (pandas, matplotlib, scikit-learn,
ecc.). Aggiornare questa sezione mano a mano.

## 4. (Opzionale) Congelare le dipendenze

Per rendere l'ambiente riproducibile:

```bash
pip freeze > requirements.txt
```

E in futuro, su una macchina nuova, dopo aver creato e attivato il venv:

```bash
pip install -r requirements.txt
```

## Note

- I moduli `csv`, `collections`, `datetime`, `pathlib`, `timeit` fanno parte
  della **standard library** di Python: non vanno installati.
- I dataset usati dagli script vivono in `00-datasets/` alla radice del
  progetto. I file `*.csv` reali sono esclusi da git (vedi `.gitignore`);
  i `*.sample.csv` versionati servono come dati di esempio.
