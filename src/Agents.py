import numpy as np
from mesa.experimental.continuous_space import ContinuousSpaceAgent


EPS = 1e-9   # soglia anti-divisione-per-zero

class TargetAgent(ContinuousSpaceAgent):
    """Punto di interesse: una zona che richiede droni.

    priority = QUOTA ASSOLUTA: quanti droni vorrebbe su di se'. Un punto di
               priorita' 3 "chiede" 3 droni, indipendentemente dagli altri punti.
    occupancy = quanti droni lo stanno presidiando ORA. Non lo calcola il punto:
                lo aggiorna il modello a ogni passo e il punto lo espone, cosi'
                i droni vicini possono leggerlo per decidere.
    """

    def __init__(self, model, space, position, priority=1.0):
        # --- ORDINE DEGLI ARGOMENTI: (space, model), invertito rispetto alla firma ---
        # Due API impongono ordini opposti:
        #   - la classe base ContinuousSpaceAgent.__init__ vuole  (space, model)
        #   - la fabbrica Agent.create_agents(model, n, *args) passa SEMPRE model per primo
        # Siccome creo i droni con create_agents, la firma del metodo deve iniziare con'model'.
        # Quindi espongo (model, space) e qui inverto.
        # Se i due valori vengono scambiati -> AttributeError: 'ContinuousSpace'
        # object has no attribute 'register_agent'  (Mesa usa lo spazio come modello).
        super().__init__(space=space, model=model)
        self.position = np.array(position, dtype=float)
        self.priority = float(priority)   # quota di droni desiderata
        self.occupancy = 0 # droni presenti ora (lo scrive il modello)
        self.idx = 0  # indice progressivo (colori/metriche)

    def step(self):
        # Per ora il punto sta fermo. Qui andra' la dinamica del territorio:
        # comparsa, scomparsa, spostamento, cambio di priorita'.
        pass

class Drone(ContinuousSpaceAgent):
    """Drone che presidia i punti bisognosi.

    Le tre regole di Reynolds, ma la COESIONE (verso il centro dei vicini) e'
    sostituita dall'ATTRAZIONE verso il punto piu' bisognoso che percepisce. In
    piu': una forza di confine (niente toro) e un'esplorazione quando non vede
    punti da servire.

    Regola di scelta: massimizza  deficit - beta * distanza,  dove
        deficit = priority - occupancy  (quanto e' sguarnito il punto),
    con una correzione: se il drone sta GIA' presidiando quel punto si scomputa
    dall'occupancy, perche' la domanda giusta e' "quanto sarebbe sguarnito se me ne
    andassi?". Senza, sarebbe lui stesso a rendere il punto pieno e se ne andrebbe.
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        direction=(1, 1),
        speed=1.0,             # sempre costante, il drone non accellera ne si ferma
        vision=10.0,           # raggio con cui vede gli ALTRI DRONI (regole boid)
        sensing_radius=25.0,   # raggio con cui percepisce i PUNTI
        separation=2.0,        # distanza minima da mantenere dagli altri droni
        coverage_radius=8.0,   # entro questo raggio "presidia" un punto
        cohere=0.25,           # peso dell'attrazione al punto
        separate=0.015,        # peso della separazione
        match=0.05,            # peso dell'allineamento
        boundary=0.3,          # peso della forza di confine
        margin=20.0,           # entro quanto dal bordo la forza di confine si attiva
        beta=0.05,             # costo di viaggio: quanti "droni di deficit" per unita' di distanza
        explore=0.2,           # intensita' della sterzata casuale in esplorazione (angolo massimo della sterzata)
    ):
        # ATTENZIONE ordine: la base vuole (space, model)
        super().__init__(space=space, model=model)

        self.position = np.array(position, dtype=float)

        # 'direction' deve essere un VERSORE (lunghezza 1): e' cio' che rende
        # 'speed' davvero lo spostamento per passo. La normalizzo alla nascita.
        self.direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > EPS:
            self.direction = self.direction / norm
        # se norm fosse 0 (direzione [0,0]) dividere darebbe NaN: la lascio com'e'


        # parametri fisici e pesi (fissi per tutta la vita del drone)
        self.speed = speed
        self.vision = vision
        self.sensing_radius = sensing_radius
        self.separation = separation
        self.coverage_radius = coverage_radius
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.boundary_factor = boundary
        self.margin = margin
        self.beta = beta
        self.explore_factor = explore

        # stato che cambia ad ogni passo
        self.neighbors = []       # droni vicini (lo riempie step)
        self.target = None        # punto scelto ora (serve a grafica/metriche)
        self.n_covered = 0        # quanti punti sto presidiando (lo scrive il modello)
        self.exploring = False    # sto esplorando? (nessun punto da servire)
        self.angle = 0.0          # rotta in gradi, solo per il disegno


    def boundary_force(self):
        """Forza che tiene il drone dentro l'area (non c'e' il toro).

        Zero lontano dai bordi; entro 'margin' dal bordo cresce linearmente,
        puntando verso l'interno. E' un vettore come le altre forze: si somma a steer.
        """
        forza = np.zeros(2)
        dimensioni_mappa = (self.model.width, self.model.height)
        #enumerate(dimensioni_mappa) genera automaticamente l'indice per ogni elemento. 
        # Restituisce una tupla (indice, valore) ad ogni giro:
        #  - Giro 1: indice=0 (Asse X), dim_massima = self.model.width
        #  - Giro 2: indice=1 (Asse Y), dim_massima = self.model.height
        for indice, dim_massima in enumerate(dimensioni_mappa):

            pos_attuale = self.position[indice]
    
            # 1. Controlla il bordo vicino allo ZERO (basso/sinistro)
            if pos_attuale < self.margin:
                forza[indice] = (self.margin - pos_attuale) / self.margin
                
            # 2. Controlla il bordo LONTANO (alto/destro)
            elif pos_attuale > (dim_massima - self.margin):
                forza[indice] = (dim_massima - self.margin - pos_attuale) / self.margin

        return forza * self.boundary_factor


    def step(self):
        """Un passo del drone: percezione -> forze -> movimento."""

        # --- PERCEZIONE ---
        # get_neighbors_in_radius(radius=self.vision) ritorna una coppia:
        # la lista degli agenti e l'array delle loro distanze, allineati (il vicino k dista distanze[k]).
        # Mesa esclude già self.
        agenti, distanze = self.get_neighbors_in_radius(radius=self.sensing_radius)

        self.neighbors = []          # droni entro vision
        distanze_droni = []
        punti_percepiti = []         # punti entro sensing_radius
        distanze_punti = []

        # zip() prende due o più liste separate e le "allaccia" insieme,
        # accoppiando il primo elemento della prima lista con il primo della seconda, il secondo con il secondo, e così via.
        for agente, distanza in zip(agenti, distanze):
            if isinstance(agente, Drone) and distanza <= self.vision: #NOTA BENE: per funzionare deve valere sempre la diseq. vision <= sensing_radius
                self.neighbors.append(agente)
                distanze_droni.append(distanza)
            elif isinstance(agente, TargetAgent):
                punti_percepiti.append(agente)
                distanze_punti.append(distanza) # sono tutte distanze già calcolate, cioè è stata fatta la norma. Più avanti si userà quella del punto scelto per normalizzare


        # --- CALCOLO DELLA DESTINAZIONE ---
        self.target = None # il punto scelto (None = nessuno)
        distanza_target = None  # la sua distanza (serve dopo, per muoversi)
        migliore_utilita = -np.inf # parto da -infinito: qualunque punto valido batte questo

        for punto, distanza in zip(punti_percepiti, distanze_punti):
            deficit = punto.priority - punto.occupancy # quanto e' sguarnito

            if distanza <= self.coverage_radius:
                # Sto gia' presidiando questo punto: mi scomputo dall'occupancy.
                # La domanda giusta e' "quanto sarebbe sguarnito SE ME NE ANDASSI?"
                deficit += 1

            if deficit <= 0:
                continue  # gia' pieno: lo salto
            utilita = deficit - self.beta * distanza # deficit meno costo di viaggio
            if utilita > migliore_utilita:
                migliore_utilita = utilita
                self.target = punto
                distanza_target = distanza

        # --- MOVIMENTO ---
        cohere_vector = np.zeros(2) # se esploro, l'attrazione è nulla
        self.exploring = False

        if self.target is not None and distanza_target > EPS:
            # CASO 1: ho un punto scelto -> sono attratto da lui SEMPRE, anche da
            # dentro il raggio coverage_radius. Dentro la zona l'attrazione non mi fa arrivare (ci
            # sono gia'): mi fa GIRARE INTORNO al punto, con raggio speed/cohere, che
            # e' il modo in cui resto in stazionamento. 
            # Spegnendola,tiravo dritto a velocita' piena e uscivo dalla zona in pochi passi.

            # la funzione calculate_difference_vector ritorna sempre una matrice (n, 2)
            # [0] prende la prima (e unica) riga: il vettore verso il target.
            delta = self.space.calculate_difference_vector(
                self.position, agents=[self.target] # come argomento si aspetta una array di agenti, quindi mettiamo il target in una array
            )[0] 

            # PATTERN "versore x peso" (lo stesso per il movimento):
            # delta = vettore verso il target, con lunghezza = distanza reale
            # delta/distanza = VERSORE: stessa direzione, lunghezza 1 (butto via la distanza)
            # * cohere_factor = sterzata di modulo ESATTAMENTE cohere, a qualunque distanza
            # La normalizzazione e' cio' che rende 'cohere' un peso confrontabile con gli altri
            # (separate, match): senza, l'attrazione crescerebbe con la distanza -> il bug "molla".
            cohere_vector = (delta / distanza_target) * self.cohere_factor
        
        elif self.target is None:
            # CASO 3: nessun punto bisognoso in vista -> esploro (blocco 3)
            self.exploring = True

        # Il ramo con distanza_target <= EPS (drone esattamente sul punto) non fa
        # nulla di proposito: servirebbe solo a evitare la divisione per zero.

        # la sterzata totale parte dall'attrazione; le altre forze si sommano nel blocco 3
        steer = cohere_vector


        # --- SEPARAZIONE E ALLINEAMENTO ---
        numero_vicini = len(self.neighbors) # tutti quelli dentro vision
        if numero_vicini > 0: # se non ho vicini non ho necessità di mantenere una distanza da loro
            # vettori dal drone verso ogni vicino (allineati con distanze_droni) perché distanze_droni e self.neighbors sono stato costruiti insieme nelle stesse iterazioni
            delta_vicini = self.space.calculate_difference_vector(
                self.position, agents=self.neighbors
            )

            #SEPARAZIONE: ci si allontana dai vicini "troppo vicini"
            separation_vector = np.zeros(2)
            for i in range(numero_vicini):
                if distanze_droni[i] < self.separation:
                    # -delta punta VIA dal vicino (delta punta verso di lui)
                    # Più droni stretti = spinta più forte, perché sommo un vettore per ciascuno.
                    separation_vector = separation_vector - delta_vicini[i]

            separation_vector = separation_vector * self.separate_factor

            # ALLINEAMENTO: adotta la direzione media dei vicini
            somma_direzioni = np.zeros(2)
            for vicino in self.neighbors:
                somma_direzioni = somma_direzioni + vicino.direction
            match_vector = somma_direzioni * self.match_factor

            # separazione e allineamento sono SOMME sui vicini: 
            # le medio dividendo per il numero di vicini. L'attrazione no (e' un punto solo) -> resta fuori.
            steer = steer + ((separation_vector + match_vector) / numero_vicini)

        # --- CONFINE (forza verso l'interno vicino ai bordi) --- 
        steer = steer + self.boundary_force()

        # --- ESPLORAZIONE (solo se non ci sono punti da servire) ---
        if self.exploring:
            # ruoto la direzione attuale di un angolo casuale piccolo -> cammino
            # persistente (il drone mantiene la rotta e la aggiusta, non tremola)
            angolo = self.model.rng.normal(0, self.explore_factor)
            cos = np.cos(angolo)
            sin = np.sin(angolo)
            dx = self.direction[0]
            dy = self.direction[1]
            # direzione ruotata di 'angolo' (formula della rotazione 2d)
            dir_ruotata_x = (cos * dx) - (sin * dy)
            dir_ruotata_y = (sin * dx) + (cos * dy)
            # steer riceve la DIFFERENZA (ruotata - attuale), come le altre forze
            steer = steer + np.array([dir_ruotata_x - dx, dir_ruotata_y - dy])

        # --- APPLICAZIONE DEL  MOVIMENTO --- 
        self.direction = self.direction + steer
        norm = np.linalg.norm(self.direction)
        if norm > EPS:
            self.direction = self.direction / norm

        # calcolo la nuova posizione, poi "taglio" ai bordi come rete di sicurezza:
        # il confinamento morbido frena ma non blocca del tutto, e su Mesa 3.5.1 una
        # posizione fuori dai limiti solleva ValueError. 
        # np.clip la riporta dentro.
        # rotta in gradi, senza, resterebbe 0.0 per sempre e in visualizzazione tutti i droni punterebbero nella stessa direzione.
        self.angle = float(np.degrees(np.arctan2(self.direction[1], self.direction[0])))

        nuova_posizione = self.position + self.direction * self.speed
        eps = 1e-6  # margine per stare STRETTAMENTE dentro (il bordo esatto puo' dare errore)
        limite_basso = np.array([eps, eps])
        limite_alto = np.array([self.model.width - eps, self.model.height - eps])
        self.position = np.clip(nuova_posizione, limite_basso, limite_alto)