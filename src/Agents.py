import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent

EPS = 1e-9

class TargetAgent(ContinuousSpaceAgent):
    """Punto di interesse passivo.

    ``occupancy`` e' la ground truth calcolata dal modello. I droni non la leggono
    per decidere: serve a metriche e visualizzazione.
    """

    def __init__(self, model, space, position, priority=1.0):
        super().__init__(space=space, model=model)
        self.position = np.array(position, dtype=float)
        self.priority = float(priority)
        self.occupancy = 0

    def step(self):
        pass


class BaseDrone(ContinuousSpaceAgent):
    """Comportamento comune a tutte le piattaforme UAV del modello.

    Qui vivono solo le parti che ala fissa e quadricottero condividono:
      * percezione di punti e droni;
      * comunicazione locale entro ``drone_sensing_radius``;
      * stima locale dell'occupancy tramite comunicazione;
      * scelta del target;
      * separazione, allineamento, confine ed esplorazione.

    La cinematica e la politica di stazionamento appartengono alle sottoclassi.
    """

    tipo_drone = "base"

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        direction=(1, 1),
        speed=1.0,
        drone_sensing_radius=10.0, # il raggio entro cui il drone vede e comunica con altri droni
        point_sensing_radius=25.0, # il raggio di percezione dei punti
        separation=2.0, # Sotto questa distanza si attiva la separazione
        coverage_radius=8.0, # raggio da cui il drone inizia a coprire un punto
        cohere=0.25, # attrazione verso la destinazione
        separate=0.015, # allontanamento dai droni troppo vicini
        match=0.05, # allineamento alle direzioni dei vicini
        boundary=0.3, # repulsione dai bordi
        margin=20.0, # Distanza dal confine alla quale comincia ad agire _boundary_force()
        beta=0.05, # penalizza la distanza nella scelta della destinazioen del drone, utilità = deficit - beta * distanza
        explore=0.2, # Controlla la variabilità casuale della direzione durante l’esplorazione
        release_delay_max_steps=5, # È l’ampiezza della componente casuale dell’attesa prima dell’uscita per sovraffollamento
    ):
        super().__init__(space=space, model=model)

        # --- stato geometrico iniziale ---
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction = self._normalize(self.direction, fallback=[1.0, 0.0])

        # --- parametri comuni ---
        self.speed = float(speed)
        self.drone_sensing_radius = float(drone_sensing_radius)  # = communication_radius per ipotesi di modello
        self.point_sensing_radius = float(point_sensing_radius)
        self.separation = float(separation)
        self.coverage_radius = float(coverage_radius)
        self.cohere_factor = float(cohere)
        self.separate_factor = float(separate)
        self.match_factor = float(match)
        self.boundary_factor = float(boundary)
        self.margin = float(margin)
        self.beta = float(beta)
        self.explore_factor = float(explore)
        self.release_delay_max_steps = int(release_delay_max_steps)

        # --- stato corrente osservabile ---
        self.target = None
        self.n_covered = 0  # ground truth, scritto dal modello
        self.exploring = False
        self.angle = float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))
        self.moving = True # serve per la rappresentazione grafica, dice se disegnare la freccia che indica la direzione del drone

        # --- fotografia prodotta dalla fase PERCEIVE ---
        self.neighbors = []
        self.neighbor_distances = []
        self.perceived_points = []
        self.perceived_point_distances = []

        # Stima locale dell'occupancy, allineata a ``perceived_points``.
        # Nessun identificatore del punto entra nella conoscenza del drone:
        # perceived_point_occupancies[k] riguarda semplicemente perceived_points[k]
        # nella fotografia locale dello step corrente.
        self.perceived_point_occupancies = []

        # --- decisione current/next state ---
        self.planned_target = None
        self.planned_exploring = False

        # --- attesa casuale prima di lasciare un punto sovraffollato ---
        # None = non e' attiva nessuna attesa. Quando parte, il valore viene estratto
        # una sola volta e decrementato a ogni step finche' il sovraffollamento persiste.
        self.release_wait_remaining = None

    # ------------------------------------------------------------------
    # GEOMETRIA DI BASE
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vettore, fallback=None):
        vettore = np.array(vettore, dtype=float)
        norma = np.linalg.norm(vettore)
        if norma > EPS:
            return vettore / norma
        if fallback is not None:
            return np.array(fallback, dtype=float)
        return np.zeros(2)

    def _clip_position(self, posizione):
        """impedisce alla posizione del drone di uscire dall’area della simulazione."""
        eps = 1e-6
        limite_basso = np.array([eps, eps])
        limite_alto = np.array([self.model.width - eps, self.model.height - eps])
        return np.clip(posizione, limite_basso, limite_alto)

    def _update_angle(self):
        self.angle = float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))

    def _boundary_force(self):
        """Forza che mantiene il drone entro i bordi."""
        forza = np.zeros(2)
        for indice, dim_massima in enumerate((self.model.width, self.model.height)):
            pos_attuale = self.position[indice]
            if pos_attuale < self.margin:
                forza[indice] = (self.margin - pos_attuale) / self.margin
            elif pos_attuale > (dim_massima - self.margin):
                forza[indice] = (dim_massima - self.margin - pos_attuale) / self.margin
        return forza * self.boundary_factor

    # ------------------------------------------------------------------
    # FASE 1 - PERCEZIONE
    # ------------------------------------------------------------------

    def perceive(self):
        """Costruisce una fotografia locale senza modificare altri agenti."""
        # I due raggi hanno semantiche indipendenti. La query deve quindi coprire
        # il maggiore dei due e i risultati vengono filtrati per tipo sotto.
        
        raggio_query = max(self.point_sensing_radius, self.drone_sensing_radius)
        agenti, distanze = self.get_neighbors_in_radius(radius=raggio_query)

        self.neighbors = []
        self.neighbor_distances = []
        self.perceived_points = []
        self.perceived_point_distances = []

        for agente, distanza in zip(agenti, distanze):
            if isinstance(agente, BaseDrone) and distanza <= self.drone_sensing_radius:
                self.neighbors.append(agente)
                self.neighbor_distances.append(float(distanza))
            elif (isinstance(agente, TargetAgent) and distanza <= self.point_sensing_radius):
                self.perceived_points.append(agente)
                self.perceived_point_distances.append(float(distanza))

        # L'occupancy percepita viene costruita nella fase COMMUNICATE, usando
        # soltanto i droni con cui posso comunicare in questo step.
        self.perceived_point_occupancies = []

    # ------------------------------------------------------------------
    # FASE 2 - COMUNICAZIONE
    # ------------------------------------------------------------------

    def communicate(self):
        """Stima localmente quanti droni stanno coprendo ciascun punto percepito.

        La comunicazione resta volutamente ad alto livello: nessun pacchetto,
        protocollo o identificatore globale del punto. Per ogni punto che IO
        percepisco, conto me stesso (se lo copro) e i soli droni entro ``drone_sensing_radius``
        la cui posizione comunicata cade entro ``coverage_radius`` da quel punto.

        In questo modo due punti vicini non vengono associati tramite un ID condiviso:
        la stima nasce ogni step dalla geometria della mia fotografia locale.
        """
        self.perceived_point_occupancies = []

        for punto, mia_distanza in zip(self.perceived_points, self.perceived_point_distances):
            conteggio = 1 if mia_distanza <= self.coverage_radius else 0

            for vicino in self.neighbors:
                distanza_vicino_punto = np.linalg.norm(vicino.position - punto.position) #per fare questo si assume che i droni comunichino la loro posizione in maniera accurata
                if distanza_vicino_punto <= self.coverage_radius:
                    conteggio += 1

            self.perceived_point_occupancies.append(conteggio)

    # ------------------------------------------------------------------
    # FASE 3 - SCELTA DEL TARGET
    # ------------------------------------------------------------------

    def _estimated_occupancy(self, punto):
        """Restituisce la stima associata alla detection locale di ``punto``.

        Il confronto ``is`` non e' un identificatore comunicato: serve soltanto
        internamente, nello stesso step, per ritrovare nella fotografia locale
        l'oggetto TargetAgent che il drone sta gia' considerando.
        """
        for candidato, occupancy in zip(self.perceived_points, self.perceived_point_occupancies):
            if candidato is punto:
                return occupancy
        return 0

    def _perceived_deficit(self, punto, distanza):
        deficit = punto.priority - self._estimated_occupancy(punto)

        # se sto gia' coprendo il punto, valuto quanti droni mancherebbero DOPO la mia eventuale partenza.
        if distanza <= self.coverage_radius:
            deficit += 1
        return deficit

    def _distance_to_perceived_point(self, punto):
        for candidato, distanza in zip(self.perceived_points, self.perceived_point_distances):
            if candidato is punto:
                return distanza
        return None

    def _current_target_is_overcrowded(self):
        """True solo se sto coprendo il mio target e la stima dice occupancy > priority."""
        if self.target is None:
            return False

        distanza = self._distance_to_perceived_point(self.target)
        if distanza is None or distanza > self.coverage_radius:
            return False

        return self._estimated_occupancy(self.target) > self.target.priority

    def _reset_release_wait(self):
        self.release_wait_remaining = None

    def _apply_random_release_wait(self, candidate_target):
        """Ritarda l'uscita da un target sovraffollato.

        Questa funzione viene chiamata solo quando abbiamo gia' stabilito che:
        - esiste un target corrente;
        - il nuovo candidato e' diverso dal target corrente;
        - il target corrente e' percepito come sovraffollato.
        """

        # Estraggo il ritardo una sola volta.
        if self.release_wait_remaining is None:
            if self.release_delay_max_steps > 0:
                self.release_wait_remaining = int( self.model.rng.integers(1, self.release_delay_max_steps + 1) )
            else:
                self.release_wait_remaining = 0

        # Finche' il timer non e' terminato rimango sul target corrente.
        if self.release_wait_remaining > 0:
            self.release_wait_remaining -= 1
            return self.target

        # Timer terminato: posso applicare la nuova decisione.
        return candidate_target

    def decide_target(self):
        """Sceglie il target usando solo informazione percepita/comunicata."""
        migliore_utilita = -np.inf
        migliore_target = None

        for punto, distanza in zip(self.perceived_points, self.perceived_point_distances):
            deficit = self._perceived_deficit(punto, distanza)
            if deficit <= 0:
                continue

            utilita = deficit - self.beta * distanza
            if utilita > migliore_utilita:
                migliore_utilita = utilita
                migliore_target = punto

        sta_lasciando_target = ( self.target is not None and migliore_target is not self.target )

        if sta_lasciando_target and self._current_target_is_overcrowded():
            migliore_target = self._apply_random_release_wait(migliore_target)
        else:
            self._reset_release_wait()

        self.planned_target = migliore_target
        self.planned_exploring = migliore_target is None

    # ------------------------------------------------------------------
    # FASE 4 - POLITICA DI STAZIONAMENTO
    # ------------------------------------------------------------------

    def decide_station(self):
        """Hook comune: l'ala fissa non ha una politica di stazionamento discreta."""
        pass

    # ------------------------------------------------------------------
    # FASE 5 - COMMIT DELLA DECISIONE
    # ------------------------------------------------------------------

    def commit_decision(self):
        """Trasforma lo stato pianificato nello stato corrente."""
        target_precedente = self.target
        self.target = self.planned_target
        self.exploring = self.planned_exploring

        if self.target is not target_precedente:
            self._reset_release_wait()

    # ------------------------------------------------------------------
    # FORZE COMUNI
    # ------------------------------------------------------------------

    def _separation_force(self):
        """Componente Boids che allontana dai vicini sotto ``separation``."""
        numero_vicini = len(self.neighbors)
        if numero_vicini == 0:
            return np.zeros(2)

        delta_vicini = self.space.calculate_difference_vector(
            self.position, agents=self.neighbors
        )

        separation_vector = np.zeros(2)
        numero_vicini_separation = 0


        for i in range(numero_vicini):
            if self.neighbor_distances[i] < self.separation:
                separation_vector -= delta_vicini[i]
                numero_vicini_separation += 1

        # Nessun vicino abbastanza vicino da attivare la separazione.
        if numero_vicini_separation == 0:
            return np.zeros(2)

        # Manteniamo la stessa normalizzazione del contributo presente nel codice
        # originale: la somma viene mediata sul numero totale di vicini in drone_sensing_radius.
        return (separation_vector * self.separate_factor) / numero_vicini_separation

    def _alignment_force(self):
        """Componente Boids che tende ad allineare la rotta a quella dei vicini."""
        numero_vicini = len(self.neighbors)
        if numero_vicini == 0:
            return np.zeros(2)

        somma_direzioni = np.zeros(2)
        for vicino in self.neighbors:
            somma_direzioni += vicino.direction

        return (somma_direzioni * self.match_factor) / numero_vicini

    def _neighbor_force(self):
        """Comodita' per il volo normale: separazione + allineamento."""
        return self._separation_force() + self._alignment_force()

    def _target_attraction(self):
        if self.target is None:
            return np.zeros(2)

        delta = self.target.position - self.position
        distanza = np.linalg.norm(delta)
        if distanza <= EPS:
            return np.zeros(2)
        return (delta / distanza) * self.cohere_factor

    def _rotated_exploration_direction(self):
        angolo = self.model.rng.normal(0, self.explore_factor)
        cos = np.cos(angolo)
        sin = np.sin(angolo)
        dx, dy = self.direction
        return np.array([(cos * dx) - (sin * dy), (sin * dx) + (cos * dy)])

    # ------------------------------------------------------------------
    # FASE 6 - MOVIMENTO: contratto per le sottoclassi
    # ------------------------------------------------------------------

    def move(self):
        raise NotImplementedError("La cinematica deve essere implementata dalla sottoclasse.")


class FixedWingDrone(BaseDrone):
    """Drone ad ala fissa: velocita' costante e sterzata progressiva."""

    tipo_drone = "ala_fissa"

    def move(self):
        
        steer = self._target_attraction()
        steer += self._separation_force()
        steer += self._alignment_force()
        steer += self._boundary_force()

        if self.exploring:
            direzione_ruotata = self._rotated_exploration_direction()
            steer += direzione_ruotata - self.direction

        # La nuova rotta nasce dalla rotta precedente + steer: e' qui che compare
        # la sterzata progressiva e, di conseguenza, il vincolo di raggio di virata.
        self.direction = self._normalize((self.direction + steer), fallback=self.direction)
        self._update_angle() # serve per la visualizzazione su solara, non impatta il movimento
        self.position = self._clip_position(self.position + self.direction * self.speed)
        self.moving = True


class QuadcopterDrone(BaseDrone):
    """Quadricottero con ruoli FREE / OWNER / SUPPORT / DEPARTING.
        Sovrascrive i metodi communicate(), decide_target(), decide_station(), commit_decision() e move() di BaseDrone
    """

    tipo_drone = "quadricottero"

    def __init__(
        self,
        model,
        space,
        avoid_angle_degrees=10.0,
        support_inset=2.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            space=space,
            **kwargs,
        )

        # i vari planned_* rappresentano le decisioni future, 
        # si usano per evitare che l'ordine di esecuzione dei droni vada a incidere sul comportamento.
        # come regola si assume che: Un drone non deve leggere un campo che gli altri potrebbero ancora modificare nella stessa fase.

        # Piccola deviazione usata quando un presidio vicino comunica di essere gia' soddisfatto.
        self.avoid_angle = np.deg2rad(float(avoid_angle_degrees))

        # Distanza di sicurezza dal bordo della coverage alla quale si fermano i support.
        # Il raggio operativo e' coverage_radius - support_inset.
        self.support_inset = float(support_inset)

        # Ruolo di stazionamento corrente e pianificato.
        # station_role = None -> drone libero/in viaggio
        # station_role = "owner" -> owner fermo al centro
        # station_role = "support"  -> support fermo nella posizione interna
        self.station_role = None  
        self.planned_station_role = None

        # Valore pubblicato esclusivamente dall'owner durante communicate().
        # Support ed explorer non costruiscono una propria stima del deficit.
        self.advertised_deficit = None

        # Stato transitorio: il drone non e' ancora SUPPORT mentre raggiunge la sua posizione radiale interna.
        # La decisione di iniziare la rilocazione resta bufferizzata tramite planned_support_relocation.
        self.support_destination = None # support_destination -> sta raggiungendo la posizione da support
        self.planned_support_relocation = False

        # Direzione radiale del lato dal quale il support e' entrato nella zona di copertura.
        self.entry_direction = None

        # Target dal quale il drone sta uscendo.
        # None significa che non siamo in DEPARTING.
        self.departing_from = None # departing_from -> sta uscendo radialmente da un punto
        self.planned_departing = False

        # Guida ricevuta da un drone stazionario che segnala un deficit positivo.
        # target = vedo direttamente il punto e posso applicare la politica di presidio
        # guidance_position = conosco soltanto la direzione verso un presidio comunicato e posso solo avvicinarmi
        # planned_target e planned_guidance_position -> mai entrambi presenti
        # target e guidance_position -> mai entrambi presenti dopo il commit
        # target e planned_guidance_position -> possono essere entrambi presenti temporaneamente, perché descrivono due step differenti
        self.guidance_position = None # guidance_position -> segue il richiamo di un presidio non visto direttamente
        self.planned_guidance_position = None

        # Posizione di un presidio soddisfatto dal quale deviare leggermente durante l'esplorazione.
        self.avoid_position = None # avoid_position -> devia leggermente da un presidio soddisfatto
        self.planned_avoid_position = None

    @property
    def owner(self):
        return self.station_role == "owner"

    # ------------------------------------------------------------------
    # Comunicazione del quadricottero
    # ------------------------------------------------------------------

    def communicate(self):
        """Fa calcolare e pubblicare il deficit soltanto all'owner.

        FREE e SUPPORT non stimano l'occupancy del presidio. L'owner, fermo al
        centro, conta se stesso e i soli OWNER/SUPPORT visibili che ricadono nella
        coverage del proprio target. Il risultato viene pubblicato in
        ``advertised_deficit`` e sara' letto o inoltrato dagli altri droni nelle
        fasi decisionali successive.
        """
        # Il campo ereditato resta necessario a BaseDrone/FixedWingDrone, ma non
        # viene usato dalla politica del quadricottero.
        self.perceived_point_occupancies = []
        self.advertised_deficit = None

        if self.station_role != "owner": # solo l'owner calcola il deficit
            return

        if self.departing_from is not None or self.target is None:
            return

        if not self._target_is_still_perceived(self.target):
            return

        mia_distanza = np.linalg.norm(self.position - self.target.position)
        if mia_distanza > self.coverage_radius:
            return

        # L'owner conta se stesso.
        occupancy = 1

        for vicino in self.neighbors:
            if not isinstance(vicino, QuadcopterDrone):
                continue
            if vicino.station_role not in ("owner", "support"):
                continue
            if vicino.departing_from is not None:
                continue

            distanza_vicino_punto = np.linalg.norm(vicino.position - self.target.position)
            if distanza_vicino_punto <= self.coverage_radius:
                occupancy += 1

        self.advertised_deficit = self.target.priority - occupancy

    # ------------------------------------------------------------------
    # Associazione geometrica dei punti
    # ------------------------------------------------------------------

    @staticmethod
    def _same_point(punto_a, punto_b):
        """Due punti sono considerati uguali tramite la loro posizione."""
        if punto_a is None or punto_b is None:
            return False

        return (np.linalg.norm(punto_a.position - punto_b.position) <= EPS)

    def _target_is_still_perceived(self, target):
        """Controlla geometricamente se il target e' ancora percepito."""
        if target is None:
            return False

        for punto in self.perceived_points:
            if self._same_point(punto, target):
                return True

        return False

    # ------------------------------------------------------------------
    # Punti attualmente coperti
    # ------------------------------------------------------------------

    def _covered_perceived_points(self):
        """Punti percepiti entro coverage_radius."""
        coperti = []

        for punto, distanza in zip(
            self.perceived_points,
            self.perceived_point_distances,
        ):
            if distanza <= self.coverage_radius:
                coperti.append((punto, distanza))

        return coperti

    # ------------------------------------------------------------------
    # Owner e deficit del presidio
    # ------------------------------------------------------------------

    def _owners_for_point(self, punto):
        """Owner visibili associati geometricamente a ``punto``.
        L'oggetto o l'ID del punto non vengono comunicati o confrontati: ogni
        associazione nasce dalla coincidenza geometrica delle posizioni.
        """
        owners = []

        if (self.station_role == "owner" and self.departing_from is None and self._same_point(self.target, punto)):
            owners.append(self)

        for vicino in self.neighbors:
            if not isinstance(vicino, QuadcopterDrone):
                continue
            if vicino.station_role != "owner":
                continue
            if vicino.departing_from is not None:
                continue
            if self._same_point(vicino.target, punto):
                owners.append(vicino)

        return owners

    def _find_owner_for_point(self, punto):
        """Restituisce l'owner autorevole del presidio, se visibile.

        Se esistono piu' owner, tutti i droni che li vedono calcolano lo stesso
        vincitore: prima il piu' vicino al centro e, a parita', il drone con
        ``unique_id`` minore.
        """
        owners = self._owners_for_point(punto)

        if not owners:
            return None

        return min(
            owners,
            key = lambda drone: (np.linalg.norm(drone.position - punto.position), drone.unique_id)
        )

    def _direct_point_information(self):
        """Valuta prima di tutto i punti percepiti.

        Per un punto presidiato il deficit dell'owner e' l'unica autorita'. Un
        punto senza owner resta invece candidabile e l'elezione avverra' solo
        dopo l'ingresso nella sua coverage.
        """
        informazioni = []

        for punto, distanza in zip(self.perceived_points, self.perceived_point_distances):
            owner = self._find_owner_for_point(punto)
            deficit_owner = None

            if owner is not None:
                deficit_owner = owner._station_deficit()

            informazioni.append(
                {
                    "point": punto,
                    "distance": distanza,
                    "owner": owner,
                    "owner_deficit": deficit_owner,
                }
            )

        return informazioni

    @staticmethod
    def _direct_point_rank(info):
        """Necessità, priorità e distanza per scegliere un punto utile."""
        punto = info["point"]
        deficit = info["owner_deficit"]

        # Per un punto ancora senza owner non esiste un deficit autorevole: 
        # la priorita' rappresenta la domanda iniziale del nuovo presidio.
        necessita = punto.priority if deficit is None else deficit

        return (necessita, punto.priority, -info["distance"])

    def _station_deficit(self):
        """Restituisce il deficit pubblicato dall'owner in communicate()."""
        if self.station_role != "owner":
            return None

        return self.advertised_deficit
    
    def _deficit_to_share(self):
        """Informazione che un drone stazionario comunica a un explorer."""
        if self.station_role not in ("owner", "support"):
            return None

        if self.target is None:
            return None

        # Anche un owner che sta per perdere un conflitto inoltra il deficit
        # dell'owner autorevole, non la propria stima concorrente.
        owner = self._find_owner_for_point(self.target)

        if owner is None:
            return None

        return owner._station_deficit()

    # ------------------------------------------------------------------
    # Informazione ricevuta dai droni stazionari
    # ------------------------------------------------------------------

    def _stationary_information(self):
        # _stationary_information() serve a costruire questa informazione: 
        #   - Quali presidi mi stanno comunicando qualcosa, e qual è il messaggio autorevole di ciascun punto?
        # Il risultato contiene al massimo un messaggio per ciascun punto,
        # anche se il drone vede owner e più support dello stesso presidio.
        """Raccoglie e filtra i messaggi sovrapposti dei presidi vicini.
            MASSIMO un messaggio per ogni punto
        """

        # Qui verranno inseriti i messaggi validi. Ogni elemento descrive un presidio, non semplicemente un drone.
        informazioni = []

        for vicino, distanza in zip(self.neighbors, self.neighbor_distances):
            if not isinstance(vicino, QuadcopterDrone):
                continue

            if vicino.station_role not in ("owner", "support"):
                continue

            # Esclude un drone che sta abbandonando il presidio.
            if vicino.departing_from is not None:
                continue

            # Esclude un drone che non ha un punto di riferimento associato
            if vicino.target is None:
                continue

            deficit = vicino._deficit_to_share()
            #NB: riguardo a vicino._deficit_to_share() vale che: 
            # Se vicino è l’owner: restituisce il deficit calcolato e pubblicato dall’owner stesso
            # Se vicino è un support: individua geometricamente l’owner del proprio punto e restituisce il deficit pubblicato dall’owner
            # Quindi il support non comunica una propria stima. Fa solamente da ripetitore

            if deficit is None:
                continue

            informazione = {
                "drone": vicino,
                "position": vicino.target.position.copy(),
                "priority": vicino.target.priority,
                "deficit": deficit,
                "distance": distanza,
                "source_is_owner": vicino.station_role == "owner",
            }

            indice_esistente = None
            for indice, esistente in enumerate(informazioni):
                if (np.linalg.norm(esistente["position"] - informazione["position"])<= EPS):
                    indice_esistente = indice
                    break

            # se il punto non è ancora presente, il messaggio viene aggiunto perchè è il primo ricevuto per quel presidio
            if indice_esistente is None:
                informazioni.append(informazione)
                continue

            # se esiste già un messaggio per quel punto: 
            # bisogna decidere se mantenere il messaggio già memorizzato oppure sostituirlo con quello nuovo.
            esistente = informazioni[indice_esistente]
            preferisci_nuova = (
                informazione["source_is_owner"] and not esistente["source_is_owner"] # Viene scelto l’owner perché rappresenta la fonte diretta e autorevole.
            ) or ( # se in caso le origini delle fonti provengono da entrambi owner o entrambi support si preferisce quella del drone più vicino a noi
                (informazione["source_is_owner"] == esistente["source_is_owner"]) and (informazione["distance"] < esistente["distance"])
            )

            if preferisci_nuova:
                informazioni[indice_esistente] = informazione

        return informazioni

    # ------------------------------------------------------------------
    # Timer per il sovraffollamento
    # ------------------------------------------------------------------

    def _draw_overcrowding_wait(self):
        """Estrae il tempo di attesa di un support sovraffollato."""
        if self.target is None:
            return 0

        distanza = np.linalg.norm(self.position - self.target.position)

        distanza_da_uscire = max(0.0, self.coverage_radius - distanza)

        tempo_minimo = max(1,int(np.ceil(distanza_da_uscire/ max(self.speed, EPS))))

        # 1 vicino al centro, 0 vicino al bordo.
        profondita = np.clip(1.0 - (distanza / self.coverage_radius), 0.0, 1.0)

        extra_massimo = int(np.ceil(self.release_delay_max_steps* profondita))

        # Anche vicino al bordo lasciamo un piccolo
        # intervallo pseudocasuale, se il parametro lo permette.
        if self.release_delay_max_steps > 0:
            extra_massimo = max(1, extra_massimo)

        if extra_massimo == 0:
            return tempo_minimo

        return int(self.model.rng.integers(tempo_minimo,tempo_minimo + extra_massimo + 1))

    def _support_should_depart(self, owner):
        """Gestisce l'attesa del support in caso di sovraffollamento."""
        deficit = owner._station_deficit()

        # Nessuna informazione affidabile oppure punto non sovraffollato.
        if deficit is None or deficit >= 0:
            self._reset_release_wait()
            return False

        # Primo step nel quale rileviamo il sovraffollamento: 
        # estraiamo il timer una volta sola e iniziamo l'attesa dallo step successivo.
        if self.release_wait_remaining is None:
            self.release_wait_remaining = (self._draw_overcrowding_wait())
            return False

        if self.release_wait_remaining > 0:
            self.release_wait_remaining -= 1
            if self.release_wait_remaining > 0:
                return False

        # Il timer e' terminato: rileggiamo esplicitamente il deficit corrente
        # dell'owner e partiamo soltanto se il sovraffollamento persiste.
        deficit_finale = owner._station_deficit()

        if deficit_finale is not None and deficit_finale < 0:
            return True

        self._reset_release_wait()
        return False

    # ------------------------------------------------------------------
    # Decisione del target
    # ------------------------------------------------------------------

    def decide_target(self):
        """Pianifica la destinazione del quadricottero usando informazioni locali.

        Mantiene eventuali transizioni già in corso; altrimenti valuta prima i punti
        percepiti, usando il deficit dell'owner come informazione autorevole, e poi i
        messaggi dei droni stazionari. Se non trova richieste utili, pianifica
        l'esplorazione con un'eventuale deviazione dai presidi soddisfatti.
        La decisione viene salvata nei campi planned_* e applicata nel commit.
        """
        self.planned_guidance_position = None
        self.planned_avoid_position = None

        # È già stato stabilito che questo drone deve diventare support.
        # Gli è stata assegnata una posizione interna alla copertura, ma non l’ha ancora raggiunta.
        if self.support_destination is not None:
            self.planned_target = self.target
            self.planned_exploring = False
            return

        # Durante l'uscita non scegliamo nuove destinazioni.
        if self.departing_from is not None:
            self.planned_target = None
            self.planned_exploring = False
            return

        # OWNER e SUPPORT mantengono il proprio presidio finche' il punto continua ad esistere.
        if self.station_role in ("owner", "support"):
            if self._target_is_still_perceived(self.target):
                self.planned_target = self.target
                self.planned_exploring = False
                return

            # Il punto non e' piu' percepito.
            self.planned_target = None
            self.planned_exploring = True
            self._reset_release_wait()
            return

        # 1) PRIMA I PUNTI. 
        # Per ogni punto gia' presidiato il deficit è comunicato dall'owner del punto.
        informazioni_punti = self._direct_point_information()

        punti_utili = []
        punti_da_evitare = []

        for info in informazioni_punti:
            owner = info["owner"]
            deficit_owner = info["owner_deficit"]

            if owner is None:
                # Un punto senza owner puo' essere scelto. 
                # L'owner verra' eletto solo quando il drone sara' realmente entro coverage_radius.
                punti_utili.append(info)
            elif deficit_owner is not None and deficit_owner > 0:
                punti_utili.append(info)
            else:
                # deficit <= 0 (o temporaneamente non disponibile): non si
                # ricalcola localmente; il presidio viene soltanto evitato.
                punti_da_evitare.append(info)

        if punti_utili:
            scelta = max(punti_utili, key=lambda info: self._direct_point_rank(info))

            self.planned_target = scelta["point"]
            self.planned_exploring = False
            self._reset_release_wait()
            return

        # 2) POI I DRONI. 
        # Solo in assenza di un punto direttamente utile si considerano i messaggi dei presidi stazionari vicini.
        informazioni_stazionarie = self._stationary_information()
        # si filtrano tutte le informazione di punti con deficit > 0
        richieste = [info for info in informazioni_stazionarie if info["deficit"] > 0]

        if richieste:
            scelta = max(
                richieste,
                key=lambda info: (info["deficit"], info["priority"], -info["distance"])
            )

            self.planned_target = None
            self.planned_exploring = False
            self.planned_guidance_position = scelta["position"].copy()
            self._reset_release_wait()
            return

        
        # Questo blocco viene raggiunto solamente quando il drone ha già verificato che:
        #   1) non esistono punti percepiti utili
        #   2) non esistono richiami da droni stazionari con deficit > 0
        # il drone deve quindi esplorare, ma potrebbe conoscere dei punti già soddisfatti da evitare

        self.planned_target = None
        self.planned_exploring = True
        self._reset_release_wait()

        # mettiamo i dati di punti_da_evitare nello stesso formato di candidati_avoid
        candidati_avoid = [
            {
                "position": info["point"].position,
                "priority": info["point"].priority,
                "distance": info["distance"],
            }
            for info in punti_da_evitare
        ]

        
        candidati_avoid.extend(
            {
                "position": info["position"],
                "priority": info["priority"],
                "distance": np.linalg.norm(self.position - info["position"])
            }
            for info in informazioni_stazionarie if info["deficit"] <= 0
        )


        if candidati_avoid:
            # come posizione da evitare si sceglie quella del punto più vicino, a pairtà di vicinanza si evita il punto con priorità maggiore
            scelta_avoid = max(candidati_avoid, key=lambda info: (-info["distance"], info["priority"]))  
            self.planned_avoid_position = scelta_avoid["position"].copy()

    # ------------------------------------------------------------------
    # Elezione OWNER / SUPPORT
    # ------------------------------------------------------------------

    def decide_station(self):
        """
        Decide quale ruolo di presidio il drone dovrà avere dopo il commit_decision():
            -FREE
            -OWNER
            -SUPPORT
            -DEPARTING
            -rilocazione verso SUPPORT
        """
        # questi sono i valori che decide_station andrà a settare
        self.planned_station_role = None # Ruolo che il drone avrà dopo il commit
        self.planned_departing = False  # Deve iniziare l’abbandono del presidio?
        self.planned_support_relocation = False # Deve raggiungere la posizione interna da support?

        # Sono già in uscita? -> non decido ruoli
        if self.departing_from is not None:
            return

        # Sto raggiungendo la posizione support?
        # -> se sono arrivato, divento support
        # -> altrimenti continuo la rilocazione
        if self.support_destination is not None:
            distanza_destinazione = np.linalg.norm(self.position - self.support_destination)
            if distanza_destinazione <= EPS:
                self.planned_station_role = "support"
            return

        # Sono owner?
        # -> risolvo eventuali conflitti
        # -> se vinco rimango owner
        # -> se perdo mi riloco come support
        if self.station_role == "owner":
            if self._target_is_still_perceived(self.target):
                owner_eletto = self._find_owner_for_point(self.target)

                if owner_eletto is self: # VITTORIA
                    # L'owner non abbandona mai per sovraffollamento.
                    self.planned_station_role = "owner"
                elif owner_eletto is not None: # SCONFITTA
                    # Un owner perdente lascia il centro e raggiunge la posizione radiale interna prima di diventare SUPPORT.
                    self.planned_support_relocation = True

                self._reset_release_wait() # Un owner non deve conservare alcun timer di abbandono.
            return

        # Sono support?
        # -> se esiste l’owner, controllo il suo deficit:
        #   -> deficit non negativo: rimango
        #   -> deficit negativo: attesa e possibile partenza
        # -> se l’owner non esiste, ricado nell’elezione
        if self.station_role == "support":
            owner = self._find_owner_for_point(self.target)

            if owner is not None:
                self.planned_station_role = "support"

                if self._support_should_depart(owner):
                    self.planned_station_role = None # annulla il ruolo support pianificato
                    self.planned_departing = True # pianifica l’inizio dell’uscita

                return
            # Owner non trovato: non facciamo return.
            # Il vecchio support prosegue nella normale elezione.

        punto = self.planned_target
        # Da questo momento la funzione gestisce: droni liberi diretti verso un punto, vecchi support rimasti senza owner.
        # Usa planned_target, non target, perché deve lavorare con la decisione appena presa da decide_target().

        # Ho un planned_target?
        # -> no: nessuna elezione
        # -> sì: controllo se sono dentro la coverage
        if punto is None:
            return

        mia_distanza = np.linalg.norm(self.position - punto.position)

        # Il drone può partecipare alla politica di presidio solamente quando è fisicamente dentro la coverage.
        if mia_distanza > self.coverage_radius:
            return 
        
        # Dentro la coverage esiste già un owner?
        # -> sì: mi riloco come support
        # -> no: confronto tutti i candidati
        owner_esistente = self._find_owner_for_point(punto)

        if owner_esistente is not None:
            self.planned_support_relocation = True
            return

        # Nessun owner: i candidati continuano verso il centro. 
        # L'elezione viene resa effettiva soltanto quando il candidato migliore ha raggiunto esattamente il punto;
        # cosi' l'owner non viene congelato sul bordo.
        candidati = [self]

        for vicino in self.neighbors:
            if not isinstance(vicino, QuadcopterDrone):
                continue

            if vicino.departing_from is not None: 
                continue

            if vicino.planned_target is None:
                continue

            if not self._same_point(vicino.planned_target, punto):
                continue

            distanza_vicino = np.linalg.norm(vicino.position - punto.position)

            if distanza_vicino <= self.coverage_radius:
                candidati.append(vicino)

        vincitore = min(candidati, key=lambda drone: (np.linalg.norm(drone.position - punto.position), drone.unique_id))

        distanza_vincitore = np.linalg.norm(vincitore.position - punto.position)

        # Il miglior candidato è al centro?
        # -> no: continuiamo ad avvicinarci
        # -> sì:
        #   -> vincitore: owner
        #   -> altri: rilocazione support
        if distanza_vincitore > EPS:
            return

        if vincitore is self:
            self.planned_station_role = "owner"
        else:
            self.planned_support_relocation = True

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit_decision(self):

        """
        commit_decision() trasforma le decisioni planned_* nello stato corrente del quadricottero.
        Non decide e non muove il drone, applica ciò che è stato stabilito da:
        decide_target()
        decide_station()
        La funzione gestisce tre casi principali, in quest'ordine:
            1. inizio dell'uscita
            2. inizio della rilocazione da support
            3. commit normale
        """
        # Un support ha terminato l'attesa (_support_should_depart() ha concluso l’attesa) e deve uscire.
        if (self.planned_departing and self.departing_from is None):
            self.departing_from = self.target # memorizzazione del punto da abbandonare

            # cancellazione dello stato di presidio
            self.support_destination = None
            self.station_role = None 
            self.planned_station_role = None

            self.target = None
            self.planned_target = None
            self.guidance_position = None
            self.avoid_position = None

            # il drone sta uscendo dalla zona, non è ancora considerato in stato di esplorazione
            self.exploring = False

            self._reset_release_wait()
            return

        # il drone era già in uscita
        if self.departing_from is not None:
            return

        # Questo può accadere quando:
        #   -un drone entra nella coverage e trova già un owner;
        #   -un candidato perde l’elezione;
        #   -un owner perde un conflitto tra più owner.
        # Il drone non diventa subito support. Prima deve raggiungere la posizione radiale interna.
        if self.planned_support_relocation:

            if self.planned_target is None:
                raise RuntimeError("Rilocazione SUPPORT pianificata senza planned_target.")
            
            super().commit_decision()

            self.station_role = None
            self.planned_station_role = None
            self.guidance_position = None
            self.avoid_position = None
            self.exploring = False

            # memorizzazione della direzione di ingresso
            if self.entry_direction is None:
                self._remember_entry_direction(self.target)
                raggio_support = max(0.0, self.coverage_radius - self.support_inset)
                destinazione = (self.target.position + (self.entry_direction * raggio_support))
                self.support_destination = self._clip_position(destinazione)

            self._reset_release_wait()
            return


        # Se il drone:
        #   -non sta iniziando un’uscita;
        #   -non è già in uscita;
        #   -non sta iniziando una rilocazione;
        #   -si arriva al commit normale.
        ruolo_precedente = self.station_role
        target_precedente = self.target

        super().commit_decision()

        self.guidance_position = (None if self.planned_guidance_position is None else self.planned_guidance_position.copy())

        self.avoid_position = (None if self.planned_avoid_position is None else self.planned_avoid_position.copy())

        self.station_role = self.planned_station_role

        # nuovo_presidio è vero quando il drone, dopo il commit, è owner/support e:
        # prima non era stazionario oppure prima presidiava un altro punto.
        nuovo_presidio = (
            self.station_role in ("owner", "support")
            and (ruolo_precedente not in ("owner", "support") or not self._same_point(target_precedente,self.target))
        )

        if nuovo_presidio:
            # La direzione radiale viene memorizzata per qualunque nuovo ruolo:
            # anche un owner puo' diventare support dopo un conflitto tra owner.
            self._remember_entry_direction(self.target)

        # Se il drone prima era owner/support e ora non lo è più, la vecchia direzione d’ingresso viene cancellata.
        if self.station_role is None:
            if ruolo_precedente in ("owner", "support"):
                self.entry_direction = None

        # pulizi dello stato di movimento per i droni stazionari
        if self.station_role is not None:
            self.support_destination = None
            self.guidance_position = None
            self.avoid_position = None

    # ------------------------------------------------------------------
    # Memoria del lato di ingresso
    # ------------------------------------------------------------------

    def _remember_entry_direction(self, punto):
        delta = self.position - punto.position
        norma = np.linalg.norm(delta)

        if norma > EPS:
            self.entry_direction = delta / norma
        else:
            self.entry_direction = self._normalize(-self.direction, fallback=np.array([1.0, 0.0]))

    # ------------------------------------------------------------------
    # Stazionamento
    # ------------------------------------------------------------------

    def _hold_station(self):
        """OWNER e SUPPORT restano esattamente nella posizione raggiunta."""
        self.moving = False

    def _move_exactly_towards_position(self, destination):
        """
            Muove verso una posizione geometrica.
            In caso la distanza dal punto sia minore del passo ci si muove solo della distanza per evitare di sforare
        """
        # calcola vettore e distanza dalla destinazione
        delta = destination - self.position
        distanza = np.linalg.norm(delta)

        # se si è già arrivati ci si ferma
        if distanza <= EPS:
            self.moving = False
            return True

        # Altrimenti orienta il drone esattamente verso la destinazione
        self.direction = delta / distanza
        # Il passo viene limitato alla distanza rimanente
        passo = min(self.speed, distanza)

        self.position = self._clip_position(self.position + self.direction * passo)
        self._update_angle()
        self.moving = passo > EPS
        return distanza <= self.speed + EPS

    def _finish_center_approach(self):
        """
        Aggancia esattamente il centro quando il candidato e' a un passo.
        - se restituisce False, non ha effettuato alcun movimento;
        - se effettua il movimento, restituisce sempre True.
        """
        if self.target is None:
            return False

        delta = self.target.position - self.position
        distanza = np.linalg.norm(delta)

        if distanza > self.speed + EPS:
            return False

        return self._move_exactly_towards_position(self.target.position) 

    # ------------------------------------------------------------------
    # Uscita dalla coverage
    # ------------------------------------------------------------------

    def _move_departure(self):
        # recupera il punto che si sta abbandonando
        punto = self.departing_from

        if punto is None:
            return

        # in caso entry_direction mancasse, si ricalcola
        if self.entry_direction is None:
            self._remember_entry_direction(punto)

        # Il drone assume esattamente la direzione radiale d’ingresso
        self.direction = self._normalize(self.entry_direction,fallback=self.direction)

        # Si sposta di un passo lungo quella direzione.
        self.position = self._clip_position(self.position + self.direction * self.speed)

        # per la rappresentazione grafica nell'app
        self._update_angle()
        self.moving = True

        distanza = np.linalg.norm(self.position - punto.position)

        # si controlla se si è usciti dalla zona di presidio
        if distanza > self.coverage_radius:
            self.departing_from = None
            self.entry_direction = None

            self.target = None
            self.guidance_position = None
            self.avoid_position = None

            self.exploring = True
            self._reset_release_wait()

    # ------------------------------------------------------------------
    # Guida e deviazione
    # ------------------------------------------------------------------

    def _attraction_to_position(self, posizione):
        delta = posizione - self.position
        distanza = np.linalg.norm(delta)

        if distanza <= EPS:
            return np.zeros(2)

        return (delta / distanza) * self.cohere_factor

    def _avoidance_direction(self, posizione):
        """Ruota leggermente la rotta lontano da un presidio pieno."""
        delta = posizione - self.position
        distanza = np.linalg.norm(delta)

        if distanza <= EPS:
            return self.direction.copy()

        verso_presidio = delta / distanza

        # Se stiamo gia' andando dalla parte opposta, non serve deviare ulteriormente.
        if np.dot(self.direction, verso_presidio) <= 0:
            return self.direction.copy()

        cos = np.cos(self.avoid_angle)
        sin = np.sin(self.avoid_angle)

        dx, dy = self.direction

        sinistra = np.array([(cos * dx) - (sin * dy), (sin * dx) + (cos * dy)])
        destra = np.array([(cos * dx) + (sin * dy), (-sin * dx) + (cos * dy)])

        # Scegliamo la rotazione meno diretta verso il presidio soddisfatto.
        if (np.dot(sinistra, verso_presidio) < np.dot(destra, verso_presidio)):
            return sinistra

        return destra

    # ------------------------------------------------------------------
    # Movimento
    # ------------------------------------------------------------------

    def move(self):

        """
        move() esegue lo stato corrente prodotto da commit_decision(),
        per questo utilizza target, guidance_position, station_role, departing_from e support_destination, 
        non i rispettivi campi planned_*.
        La priorità dei rami è:
            1. Uscita dal presidio
            2. Rilocazione come support
            3. Owner/support già stazionario
            4. Aggancio finale al centro
            5. Volo normale
        """
        # Stato transitorio di uscita.
        if self.departing_from is not None:
            self._move_departure()
            return

        # Questo stato si verifica quando il drone:
        # è entrato nella coverage;
        # ha trovato un owner esistente, oppure ha perso l’elezione;
        # deve raggiungere la propria posizione interna da support.
        # Finché non arriva, non è ancora support: è un drone in rilocazione.
        if self.support_destination is not None:
            self._move_exactly_towards_position(self.support_destination)
            return

        # Un drone che presidia e' immobile: 
        # niente separazione, rientro verso il centro o altro movimento interno alla coverage.
        if self.station_role in ("owner", "support"):
            self._hold_station()
            return

        # Un punto ancora senza owner richiede che il candidato raggiunga il centro prima dell'elezione. 
        # Lo snap finale evita overshoot/oscillazioni dovute al passo di lunghezza costante.
        # False: non è stato eseguito l’aggancio; continua con il volo normale;
        # True: il drone è stato portato al centro; termina move().
        if self.target is not None and self._finish_center_approach():
            return

        # Volo normale.
        neighbor_force = (self._separation_force() + self._alignment_force())

        boundary_force = self._boundary_force()

        if self.target is not None:
            desiderata = (self._target_attraction() + neighbor_force + boundary_force)

        elif self.guidance_position is not None:
            desiderata = (self._attraction_to_position(self.guidance_position) + neighbor_force + boundary_force)

        else:
            if self.avoid_position is not None:
                direzione_base = (self._avoidance_direction(self.avoid_position))
            else:
                direzione_base = (self._rotated_exploration_direction())

            desiderata = (direzione_base + neighbor_force + boundary_force)

        self.direction = self._normalize(desiderata, fallback=self.direction)

        self._update_angle()

        self.position = self._clip_position(self.position + (self.direction * self.speed))

        self.moving = True
