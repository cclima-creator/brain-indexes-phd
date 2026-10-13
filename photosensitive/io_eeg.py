# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:02:11 2026

@author: Chiara Clima
"""

"""Importazione dati EEGLAB (.set/.fdt) -> mne.Epochs. Nessun calcolo qui.


files
  ↓
lista dei file .set

ep
  ↓
OGGETTO MNE EPOCHS
  ├── dati EEG
  ├── informazioni sulle epoche
  ├── eventi
  ├── tempi
  ├── canali
  └── metadata

labels
  ↓
nome/tipo dell'evento di ogni epoca

data_uv
  ↓
array numerico EEG in µV



"""


from pathlib import Path
import mne    ##libreria per leggere dati eeg


def discover_files(folder, pattern="*.set"):    ##input: tutti i file nella cartella che voglio, che hanno estensione .set(fissa)
    """Lista ordinata dei file .set in una cartella (il .fdt deve stare accanto)."""
    
    return sorted(Path(folder).glob(pattern))  ## Path(folder)= trasforma la stringa della cartella in un oggetto che si chiama Path: ogni volta che chiamo path sto parlando della cartella dove ho preso i files
                                                ## .glob(pattern)= sta considerando i file nella cartella che corrispondono al pattern, che negli input della funzione il pattern e' fissato a .set
                                                ## sorted= li mette in ordine alfabetico
                                                ##!!! RESTITUISCE LISTA FILE NEL PATH ORDINATA!!!
                                                

def load_epochs(path, fixed_length_s=None, window=None, picks="eeg"):    ##picks sono i cnaali che si possono mantenre, e'  variabile fissa e prende tutti gli eeg che sono classificati da MNE come eeg (nel mio caso tutti)
                                                                          ##volendo posso prendere solo alcuni canali ep.pick(["O1", "O2", "Pz"])                             
    """                                                                 
    Carica un .set come mne.Epochs.

    - Se il file contiene epoche: le usa cosi' come sono.
    - Se e' un registrato continuo: serve fixed_length_s (es. 4.0) per tagliarlo in
      epoche di durata fissa; se e' None solleva l'errore originale.
    - window=(tmin, tmax) ritaglia ogni epoca (secondi, tmax escluso).
    - posso prendere solo alcuni canali ep.pick(["O1", "O2", "Pz"]. la impostaz predefinita li prende tutti
    """
    try:
        ep = mne.io.read_epochs_eeglab(str(path), verbose="ERROR")  ## con la libr mne per leggere eeg prova a vedere se ci sono epoche gia definite
    except Exception:                           ##se non ci sono epoche le crea
        if fixed_length_s is None:
            raise                                ##raise serve per fare vedere che ce  errore se non ci sono epoche E non ho definito input della window per crearle
        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
        ep = mne.make_fixed_length_epochs(raw, duration=fixed_length_s, ##creazione delle epoche se non ci sono nei dati originali e do corretti input
                                          preload=True, verbose="ERROR")
    ep.pick(picks)                              ##ep e' il nome della variabile multidimensionale che contiene info su epoche
    if window is not None:
        ep.crop(window[0], window[1], include_tmax=False)
    return ep

    """
                    .set
                 │
                 ▼
         read_epochs_eeglab()
                 │
          ┌──────┴──────┐
          │             │
      ha epoche?      continuo?
          │             │
         YES            NO
          │             │
          ▼             ▼
      usa quelle    read_raw_eeglab
      esistenti          │
                         ▼
               make_fixed_length_epochs
                         │
                         ▼
                     Epochs
          └──────────────┬──────────────┘
                         │
                         ▼
                   ep.pick(picks)
                         │
                         ▼
                   ep.crop(window)
                         │
                         ▼
                   return ep
               
                
               
                
               
                    ep       e' una variabile fornita da mne con molti parametri delle epoche
                    │
                    ├── dati EEG
                    ├── epoche
                    ├── canali
                    ├── tempi
                    ├── eventi
                    ├── event_id
                    ├── sampling frequency
                    ├── informazioni sui canali
                    └── ...
      """


def epoch_labels(epochs):           ##l' input epochs sono le epoche salvate in ep nella funzione prima
    """Etichetta evento di ogni epoca (lista di str, lunghezza n_epochs)."""
    inv = {v: k for k, v in epochs.event_id.items()}            ##event_id contiene nomi di epoche (labels) e gli assegna un codice numerico. in event e' incluso il tempo di inizio dell epoca
                                                                ##inv= inverte label->codice in codice->label (dettagli tecnici) serve perceh i dati dopo mne hanno codici mentre io voglio vedere le etichette
    return [inv.get(c, str(c)) for c in epochs.events[:, 2]]    ##restituisce lista di targhette


def get_data_uv(epochs):
    """Dati in microvolt, shape (n_epochs, n_channels, n_times)."""
    return epochs.get_data() * 1e6                      ##prende  i dati EEG e cambia la loro unità da V → µV. perche mne li prende automaticamente in V