"""Sweep parametrico: esegue il modello su una griglia di parametri e salva i dati.

Avvio:  uv run python sweep_spot.py
"""

import time

import numpy as np
import pandas as pd
import mesa

from Model import CoverageModel


# --- COSA SI VARIA ---
# Un parametro alla volta, e per primo beta: e' il cuore della regola di decisione
# (utilita' = deficit - beta * distanza), quindi e' quello su cui una tesi deve
# avere una curva. beta=0 significa "vado sempre dal piu' bisognoso, la distanza
# non conta"; beta alto significa "resto vicino, anche se serve altrove".
GRIGLIA = {
    "beta": [0.0, 0.025, 0.05, 0.10, 0.20, 0.30],
}

# --- SEMI ESPLICITI, NON 'iterations' ---
# iterations=8 equivale a rng=[None]*8, cioe' otto mondi NON riproducibili: rilanciando
# lo sweep otterresti numeri diversi e non sapresti se e' cambiato il codice o il caso.
# Con una lista esplicita ogni valore di beta viene provato sugli STESSI mondi: le
# posizioni dei punti e le quote sono identiche, quindi la differenza misurata e'
# attribuibile a beta e non alla fortuna. E' un confronto appaiato.
# batch_run ispeziona la firma del modello: trovando 'seed=' passera' quello.
SEMI = [1, 2, 3, 4, 5, 6, 7, 8]

PASSI = 600        # durata di ogni run
TRANSITORIO = 300  # passi iniziali da scartare in analisi (vedi blocco 2)


def esegui():
    """Lancia lo sweep e restituisce il dataframe grezzo, una riga per passo."""
    inizio = time.perf_counter()

    risultati = mesa.batch_run(
        CoverageModel, # la classe, non un'istanza: la costruisce lui
        parameters=GRIGLIA, # cosa variare
        rng=SEMI,  # i semi, espliciti
        max_steps=PASSI, # quanto durano
        # 1 e non -1: con -1 si raccoglie SOLO l'ultimo passo, cioe' una fotografia singola. 
        data_collection_period=1,  # raccogli a OGNI passo
        # 1 e non None: con piu' processi il guadagno c'e', ma su Windows la partenza
        # e' "spawn" e ogni processo REIMPORTA questo file. Se batch_run venisse
        # chiamato a livello di modulo invece che dentro if __name__ == "__main__",
        # ogni figlio rilancerebbe lo sweep e ne genererebbe altri: fork bomb.
        # La guardia in fondo al file e' quello che lo impedisce.
        number_processes=1, # un processo solo
        display_progress=True, # barra di avanzamento
    )

    df = pd.DataFrame(risultati)
    print(f"\n{len(df)} righe in {time.perf_counter() - inizio:.1f} s")
    return df


# ======================================================================
# BLOCCO 2 - ANALISI
# ======================================================================

METRICHE = [
    "deficit_residuo",
    "punti_soddisfatti",
    "sovra_servizio",
    "droni_oziosi",
    "droni_in_esplorazione",
]


def aggrega(df, parametro, transitorio=TRANSITORIO):
    """Media in DUE STADI: prima nel tempo dentro ogni run, poi fra le run.

    Perche' non in un colpo solo. Dentro una run, il valore al passo t e' quasi
    identico a quello al passo t-1: l'autocorrelazione a ritardo 1 misurata sui
    nostri dati e' 0.896. Le 2400 righe di un singolo beta NON sono 2400
    osservazioni indipendenti: sono 8 run, una per seme.
    Mediando tutto insieme, l'errore standard verrebbe 0.051 invece di 0.307 -
    sottostimato SEI VOLTE - e differenze dovute al caso sembrerebbero significative.
    E' l'errore statistico piu' comune nell'analisi di simulazioni, e non da' nessun
    segnale: produce semplicemente barre d'errore troppo strette.

    Ritorna due tabelle: una riga per run, e la sintesi per valore del parametro.
    """
    regime = df[df["Step"] >= transitorio]

    # STADIO 1: una riga per (parametro, seme). Qui la media nel tempo e' legittima:
    # sta riassumendo una singola traiettoria, non sommando osservazioni indipendenti.
    per_run = regime.groupby([parametro, "seed"])[METRICHE].mean().reset_index()

    # STADIO 2: fra le run. n = numero di semi, ed e' questo che entra nell'errore.
    sintesi = per_run.groupby(parametro)[METRICHE].agg(["mean", "std", "count"])
    for m in METRICHE:
        sintesi[(m, "err")] = sintesi[(m, "std")] / np.sqrt(sintesi[(m, "count")])
    return per_run, sintesi.sort_index(axis=1)


def confronto_appaiato(per_run, parametro, riferimento, metrica="deficit_residuo"):
    """Differenza rispetto a un valore di riferimento, SEME PER SEME.

    E' il motivo per cui nel blocco 1 i semi sono espliciti. Ogni valore del
    parametro e' stato provato sugli stessi identici mondi, quindi si puo' chiedere
    "su QUESTO mondo, quanto e' cambiato?" invece di confrontare due medie calcolate
    su mondi diversi. La variabilita' fra mondi - che qui e' la fonte di rumore
    dominante - si cancella nella sottrazione.
    """
    tabella = per_run.pivot(index="seed", columns=parametro, values=metrica)
    differenze = tabella.sub(tabella[riferimento], axis=0)
    return pd.DataFrame({
        "differenza_media": differenze.mean(),
        "err": differenze.std() / np.sqrt(len(differenze)),
        "semi_migliorati": (differenze < 0).sum(),
    })


def grafico(sintesi, appaiato, parametro, riferimento, percorso="sweep_beta.png"):
    """Due pannelli: valori assoluti e differenze appaiate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (sx, dx) = plt.subplots(1, 2, figsize=(10, 4))

    x = sintesi.index.values
    m = sintesi[("deficit_residuo", "mean")].values
    e = sintesi[("deficit_residuo", "err")].values
    sx.errorbar(x, m, yerr=e, marker="o", capsize=3, color="tab:red")
    sx.set_xlabel(parametro)
    sx.set_ylabel("deficit residuo a regime")
    sx.set_title("valori assoluti")

    # Le barre d'errore qui sono piu' strette: la varianza fra mondi e' sparita
    # nella sottrazione, quindi lo stesso numero di run distingue differenze piu'
    # piccole. E' il guadagno del disegno appaiato, e si vede a occhio.
    dx.errorbar(appaiato.index.values, appaiato["differenza_media"].values,
                yerr=appaiato["err"].values, marker="o", capsize=3, color="tab:blue")
    dx.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    dx.set_xlabel(parametro)
    dx.set_ylabel(f"differenza rispetto a {parametro}={riferimento}")
    dx.set_title("confronto appaiato, stesso mondo")

    fig.tight_layout()
    fig.savefig(percorso, dpi=120)
    return percorso


FILE_CSV = "sweep_beta.csv"
RIFERIMENTO = 0.05   # il valore di default, contro cui confrontare gli altri


if __name__ == "__main__":
    # LA GUARDIA E' OBBLIGATORIA su Windows, anche con number_processes=1: mesa
    # imposta comunque il metodo di avvio "spawn" all'inizio di batch_run.
    import os

    # Rieseguire lo sweep a ogni ritocco dell'analisi costa 90 secondi per niente:
    # se il csv c'e' gia' lo riuso. Cancellalo per rifare le simulazioni.
    if os.path.exists(FILE_CSV):
        print(f"riuso {FILE_CSV} (cancellalo per rifare le simulazioni)")
        dati = pd.read_csv(FILE_CSV)
    else:
        dati = esegui()
        dati.to_csv(FILE_CSV, index=False)

    per_run, sintesi = aggrega(dati, "beta")
    appaiato = confronto_appaiato(per_run, "beta", RIFERIMENTO)

    print()
    print("=== deficit residuo a regime, per beta ===")
    print(sintesi["deficit_residuo"][["mean", "std", "err"]].round(3).to_string())
    print()
    print(f"=== differenza rispetto a beta={RIFERIMENTO}, stesso mondo ===")
    print(appaiato.round(3).to_string())
    print()
    print("grafico salvato in", grafico(sintesi, appaiato, "beta", RIFERIMENTO))