"""Modello: k droni con capacita' limitata coprono n punti di interesse (Mesa 4).

SETUP: risorse SCARSE. I droni sono pochi, i punti tanti: non si puo' coprire tutto,
e l'obiettivo e' massimizzare i punti coperti.

REGOLA DI DECISIONE (baseline): ogni drone insegue il punto piu' vicino. E' la regola
piu' ingenua possibile ed e' volutamente cosi': serve come termine di paragone.
"""

import numpy as np
from mesa import Model  # la classe base del mondo
from mesa.datacollection import DataCollector # raccolta dati
from mesa.experimental.continuous_space import ContinuousSpace # lo spazio
from mesa.experimental.scenarios import Scenario   # i parametri

from Agents import Drone, TargetAgent


class BoidsScenario(Scenario):
    """Parametri del modello.

    Risorse:
        population_size: numero di DRONI (pochi)
        n_targets: numero di PUNTI da coprire (tanti)
        coverage_radius: CAPACITA' del drone. Copre i punti entro questo raggio.
            Vincolo importante: deve essere > speed/cohere (il raggio di virata),
            altrimenti il drone orbita FUORI dal punto e la copertura oscilla.

    Territorio:
        target_layout: "cluster" (realistico: folle, incendi), "casuale", "griglia"
        n_clusters, cluster_spread: forma dei cluster
    """

    # --- risorse ---
    population_size: int = 12   # droni
    n_targets: int = 60         # punti da coprire
    coverage_radius: float = 8.0

    # --- spazio ---
    width: int = 100
    height: int = 100

    # --- boid ---
    speed: float = 1.0
    vision: float = 10.0
    separation: float = 2.0
    cohere: float = 0.25  # serve cohere > speed/coverage_radius, vedi sopra
    separate: float = 0.015
    match: float = 0.05

    # --- territorio ---
    target_layout: str = "cluster"
    n_clusters: int = 6
    cluster_spread: float = 7.0
    target_positions: object = None  # posizioni esplicite: hanno la precedenza

    def __init__(self, *, rng=None, **kwargs):
        """Come Scenario, ma tollera un seme passato come stringa (InputText)."""
        if isinstance(rng, str):
            if rng.strip():
                rng = int(rng)
            else:
                rng = None
        super().__init__(rng=rng, **kwargs)


class BoidFlockers(Model):
    """Sciame di droni che cerca di coprire piu' punti possibile."""

    def __init__(self, scenario: BoidsScenario | type[BoidsScenario] = BoidsScenario):
        super().__init__(scenario=scenario)
        s: BoidsScenario = self.scenario

        positions = self._target_positions(s)
        self.coverage_radius = float(s.coverage_radius)

        self.space = ContinuousSpace(
            [[0, s.width], [0, s.height]],
            torus=True,
            random=self.random,
            n_agents=s.population_size + len(positions),
        )

        # --- punti di interesse (tutti di peso 1.0, per ora) ---
        TargetAgent.create_agents(self, len(positions), self.space, position=positions)
        self.target_agents = self.agents_by_type[TargetAgent].to_list()
        for i, t in enumerate(self.target_agents):
            t.idx = i

        # --- droni ---
        drone_positions = self.rng.random(size=(s.population_size, 2)) * self.space.size
        directions = self.rng.uniform(-1, 1, size=(s.population_size, 2))
        Drone.create_agents(
            self, 
            s.population_size,
            self.space,
            position=drone_positions,
            direction=directions,
            cohere=s.cohere,
            separate=s.separate,
            match=s.match,
            speed=s.speed,
            vision=s.vision,
            separation=s.separation,
            coverage_radius=s.coverage_radius,
        )
        self.drones = self.agents_by_type[Drone]

        self.datacollector = DataCollector(
            model_reporters={
                "PuntiCoperti": lambda m: m._coperti,
                "Ridondanza": lambda m: m._ridondanza,
                "DroniOziosi": lambda m: m._oziosi,
                "DistanzaMediaPunto": lambda m: m._distanza_media,
            }
        )
        self._update_coverage()
        self.calculate_angles()
        self.datacollector.collect(self)  # riga a t = 0

    # --------------------------------------------------------------- territorio
    def _target_positions(self, s: BoidsScenario) -> np.ndarray:
        """Posizioni dei punti: esplicite, oppure generate secondo il layout."""
        if s.target_positions is not None:
            return np.asarray(s.target_positions, dtype=float).reshape(-1, 2)

        n, w, h = int(s.n_targets), float(s.width), float(s.height)

        if s.target_layout == "cluster":
            # i fenomeni reali (folle, incendi) sono a grappoli, non uniformi:
            # e' il layout che rende il problema interessante, perche' conviene
            # piazzare un drone dove i punti sono fitti
            centri = self.rng.random(size=(int(s.n_clusters), 2)) * [w, h]
            appartenenza = self.rng.integers(0, len(centri), size=n)
            punti = centri[appartenenza] + self.rng.normal(0, s.cluster_spread, size=(n, 2))
            return np.mod(punti, [w, h])  # lo spazio e' un toro

        if s.target_layout == "griglia":
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            xs = np.linspace(w / (cols + 1), w * cols / (cols + 1), cols)
            ys = np.linspace(h / (rows + 1), h * rows / (rows + 1), rows)
            return np.array([[x, y] for y in ys for x in xs])[:n]

        return self.rng.random(size=(n, 2)) * np.array([w, h])  # "casuale"

    # -------------------------------------------------------------------- step
    def step(self):
        self.drones.shuffle_do("step")
        self._update_coverage()
        self.calculate_angles()
        self.datacollector.collect(self)

    def _update_coverage(self):
        """Chi copre cosa. Una sola volta per step: le metriche leggono da qui.

        D[j, i] = distanza fra il drone j e il punto i (toro-aware).
        Un punto e' coperto se almeno un drone lo ha entro il proprio raggio.
        """
        drones = self.drones.to_list()
        pois = self.target_agents
        D = np.array([self.space.calculate_distances(d.position, agents=pois)[0]
                      for d in drones])          # (n_droni, n_punti)

        coperto = D < self.coverage_radius       # matrice booleana
        per_punto = coperto.sum(axis=0)          # da quanti droni e' coperto ogni punto
        per_drone = coperto.sum(axis=1)          # quanti punti copre ogni drone

        for poi, n in zip(pois, per_punto):
            poi.covered_by = int(n)
        for drone, n in zip(drones, per_drone):
            drone.n_covered = int(n)

        n_coperti = int((per_punto > 0).sum())
        self._coperti = n_coperti / len(pois)                     # <-- LA metrica
        self._oziosi = float((per_drone == 0).mean())             # droni che non servono a nulla
        # quanti droni in media stanno addosso allo stesso punto coperto:
        # 1.0 = nessuno spreco, 3.0 = tre droni per punto (due sprecati)
        self._ridondanza = float(per_punto.sum() / n_coperti) if n_coperti else 0.0
        self._distanza_media = float(D.min(axis=0).mean())        # misura continua

    def calculate_angles(self):
        """Rotta di ogni drone in gradi, per orientare il marker."""
        d = np.array([drone.direction for drone in self.drones])
        angles = np.degrees(np.arctan2(d[:, 1], d[:, 0])) - 90.0
        for drone, angle in zip(self.drones, angles):
            drone.angle = angle