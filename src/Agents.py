"""Agenti: droni (boid) attratti dai bersagli invece che dai vicini."""

import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent

EPS = 1e-9


class TargetAgent(ContinuousSpaceAgent):
    """Punto di interesse (POI) da coprire.

    NB: il raggio di copertura NON sta qui: e' una CAPACITA' DEL DRONE. Un punto e'
    coperto se esiste un drone entro il raggio di copertura di quel drone.
    """

    def __init__(self, model, space, position, weight=1.0):
        # ContinuousSpaceAgent vuole (space, model); noi esponiamo (model, space)
        # perche' e' l'ordine in cui li passa create_agents.
        super().__init__(space, model)
        self.position = np.array(position, dtype=float)
        self.weight = float(weight)  # criticita' della zona (per ora tutti a 1.0)
        self.covered_by = 0  # quanti droni lo stanno coprendo ORA (lo aggiorna il modello)
        self.idx = 0  # indice progressivo, per colori e metriche

    def step(self):
        """Il punto non si muove (per ora).

        QUI andra' la dinamica del territorio: comparsa, scomparsa, spostamento o
        cambio di peso dei punti. E' il fenomeno che la tesi deve studiare.
        """


class Drone(ContinuousSpaceAgent):
    """
    REGOLA DI DECISIONE ATTUALE (Baseline ingenua): "dirigiti al bersaglio più vicino".
    Attualmente l'agente ignora se il punto è già presidiato, il "peso" dell'obiettivo e 
    le capacità operative (proprie e dello sciame). È una scelta voluta: serve come 
    benchmark di base per valutare l'efficacia di algoritmi decisionali più complessi.
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        speed=1.0,
        direction=(1, 1),
        vision=10.0,
        separation=2.0,
        cohere=0.15,   # intensita' di sterzata verso il punto
        separate=0.015,
        match=0.05,
        coverage_radius=8.0,   # CAPACITA': copre i punti entro questo raggio
    ):
        super().__init__(space, model)
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > EPS:  # la direzione deve essere un VERSORE fin dall'inizio,
            self.direction /= norm  # altrimenti i pesi non sono confrontabili
        self.speed = speed
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.coverage_radius = float(coverage_radius)
        self.neighbors = []
        self.target = None  # punto inseguito (serve anche alla grafica)
        self.n_covered = 0  # quanti punti sta coprendo ORA (lo aggiorna il modello)
        self.angle = 0.0

    def step(self):
        """Percezione -> aggiornamento della direzione -> movimento."""
        # --- vicini: solo altri droni, i bersagli vanno filtrati ---
        neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
        mask = np.array([isinstance(n, Drone) for n in neighbors], dtype=bool)
        self.neighbors = [n for n, keep in zip(neighbors, mask) if keep]
        boid_distances = distances[mask] if len(neighbors) else distances

        # --- 1. ATTRAZIONE verso il bersaglio piu' vicino ---------------------
        targets = self.model.target_agents
        deltas = self.space.calculate_difference_vector(self.position, agents=targets)
        target_distances = np.linalg.norm(deltas, axis=1)
        i = int(np.argmin(target_distances))
        self.target = targets[i]

        d = target_distances[i]
        # VERSORE, non il vettore grezzo: la sterzata non deve dipendere da quanto
        # e' lontano il bersaglio (altrimenti e' una molla: forte da lontano, nullada vicino).
        #  Cosi' `cohere` = intensita' di sterzata per step, ed e'
        # confrontabile con `separate` e `match`.
        cohere_vector = (deltas[i] / d) * self.cohere_factor if d > EPS else np.zeros(2)

        # --- 2. SEPARAZIONE e 3. ALLINEAMENTO (regole classiche, invariate) ----
        steer = cohere_vector
        n_boids = len(self.neighbors)
        if n_boids > 0:
            delta = self.space.calculate_difference_vector(
                self.position, agents=self.neighbors
            )
            close = boid_distances < self.separation
            separation_vector = (
                -1 * delta[close].sum(axis=0) * self.separate_factor
                if close.any()
                else np.zeros(2)
            )
            match_vector = (
                np.asarray([n.direction for n in self.neighbors]).sum(axis=0)
                * self.match_factor
            )
            # la divisione per n_boids serve a MEDIARE le somme sui vicini:
            # l'attrazione al target NON e' una somma sui vicini, quindi resta fuori.
            steer = steer + (separation_vector + match_vector) / n_boids

        self.direction = self.direction + steer
        norm = np.linalg.norm(self.direction)
        if norm > EPS:
            self.direction /= norm

        self.position += self.direction * self.speed