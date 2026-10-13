# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:08:24 2026

@author: Chiara Clima
"""

"""Salvataggio: cartella di output sicura, Excel multi-foglio, risultati intermedi (pickle)."""

import pickle
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

FALLBACK_DIR = Path.home() / "eeg_quant_output"     ## path.home= home directory dell'utente Windows utenti\chiara clima




def _can_write(folder, names):              ## se il programma ha effettivamente il permesso di scrivere nella cartella.
                                            ## names=lista di nomi dei file che il programma prevede di salvare, usata solo per verificare che possa crearli.
                                            ## es : excel_results_bands ...
    """Prova a scrivere (e cancellare) un file temporaneo per ogni nome di output."""
    try:
        folder.mkdir(parents=True, exist_ok=True)           ##parents=True → se mancano cartelle superiori, le crea
                                                            ##exist_ok=True → se la cartella esiste già, non dà errori
        for n in names:                         ## per tutti gli n nomi
            t = folder / (n + ".tmp")           ##tmp temporary file che verra cancellato subito
            t.write_bytes(b"test")
            t.unlink()                      ##cancella
        return True
    except OSError as e:
        print(f"ATTENZIONE: non riesco a scrivere in {folder}\n  -> {e}")   ##stampa errore se non riesce
        return False


def check_output_dir(folder, names, fallback=FALLBACK_DIR):     ##folder → cartella che vuoi usare per salvare i risultati
                                                                ##names → nomi dei file di output che vuoi creare (results_bands...)
                                                                ##fallback → cartella alternativa, che non devi necessariamente impostare nel main, perché ha già come default FALLBACK_DIR
    """Ritorna una cartella scrivibile (folder, altrimenti fallback) PRIMA dell'analisi."""
    folder = Path(folder)
    if _can_write(folder, names):
        return folder
    print("Provo la cartella alternativa:", fallback)
    if _can_write(Path(fallback), names):
        return Path(fallback)
    raise SystemExit("Nessuna cartella scrivibile: imposta OUT_DIR a un percorso valido.")
    
    """
    
    cartella scelta
     ↓
  scrivibile?
   /      \
 YES       NO
  ↓         ↓
return    fallback
            ↓
        scrivibile?
         /      \
       YES       NO
        ↓         ↓
     return    errore
 
   """




def save_tables_excel(tables, path, sheet_order=()):
    """Un solo .xlsx, un foglio per tabella (dict nome -> DataFrame), colonne separate."""
    names = [n for n in sheet_order if n in tables] + \
            [n for n in tables if n not in sheet_order]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in names:
            df = tables[name]
            sheet = name[:31]               ##I nomi dei fogli Excel non possono essere più lunghi di 31 caratteri. (non da errore, taglia il nome)
            df.to_excel(xw, sheet_name=sheet, index=False)   ##scrive realmente la tabella nell'Excel.
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"          ##ws.freeze_panes = "A2" tipo titolo delle colonne . es:subject | alpha | beta | gamma
            for i, col in enumerate(df.columns, start=1):
                widths = [len(str(col))] + [len(str(v)) for v in df[col].head(200)]    #!!PERCHE??     ##considerando al massimo le prime 200 righe.
                ws.column_dimensions[get_column_letter(i)].width = min(max(widths) + 2, 40)


def save_results(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)