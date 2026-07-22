"""Sweep sul numero di spot. In Mesa 4 mesa.batch_run NON esiste piu':
le repliche si fanno con scenario.spawn_replications(n) (semi derivati dal seme
dello scenario -> riproducibili ma indipendenti).

Avvio:  uv run python sweep_spot.py
"""
import numpy as np
import pandas as pd

from Model import BoidFlockers, BoidsScenario

STEPS, BURN_IN, REPLICHE = 300, 200, 2
MISURE = ["Copertura", "DistanzaMediaSpot", "Polarizzazione", "Bilanciamento"]

righe = []
for i, n in enumerate([1, 2, 3, 4, 6, 8]):
    base = BoidsScenario(rng=100 + i, scenario_id=i, n_targets=n, target_layout="cerchio")
    for scenario in base.spawn_replications(REPLICHE):
        model = BoidFlockers(scenario=scenario)
        model.run_for(STEPS)
        df = model.datacollector.get_model_vars_dataframe()
        righe.append({"n_spot": n, "replica": scenario.replication_id,
                      **df.iloc[BURN_IN:][MISURE].mean().to_dict()})

out = pd.DataFrame(righe)
out.to_csv("sweep_spot.csv", index=False)
print(out.groupby("n_spot")[MISURE].mean().round(2).to_string())