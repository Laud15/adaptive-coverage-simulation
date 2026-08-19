import numpy as np
import pandas as pd
import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from Agents import FixedWingDrone, QuadcopterDrone, TargetAgent


DRONE_CLASS_BY_TYPE = {
    "ala_fissa": FixedWingDrone,
    "quadricottero": QuadcopterDrone,
}

# Condizioni iniziali disponibili per i punti di interesse.
# Sono scenari geometrici, non politiche adattive: vengono scelti una volta in __init__.
DISPOSIZIONI_PUNTI = (
    "casuali",
    "gruppi",
    "sparsi",
    "cerchio",
    "bordi",
    "centrali",
)

# --- CLASSE THREAD-SAFE PER LA RACCOLTA DATI ---
class ThreadSafeDataCollector(mesa.DataCollector):
    """Versione personalizzata di DataCollector che previene le Race Conditions.

    Quando SolaraViz legge i dati per aggiornare i grafici mentre il motore
    in background sta scrivendo il passo successivo, le colonne del dataframe
    potrebbero avere momentaneamente lunghezze diverse (es. 531 vs 530).
    Questa classe pareggia al volo le colonne tagliando la più lunga alla
    lunghezza minima comune, evitando il crash di Pandas/Matplotlib.
    """

    def get_model_vars_dataframe(self):
        if not self.model_vars:
            return pd.DataFrame()

        # Trova la lunghezza minima comune tra tutti i dati raccolti
        min_len = min(len(valori) for valori in self.model_vars.values())

        # Taglia temporaneamente ogni colonna a quella dimensione
        dati_sicuri = {chiave: valori[:min_len] for chiave, valori in self.model_vars.items()}
        return pd.DataFrame(dati_sicuri)


class CoverageModel(mesa.Model):
    """Il mondo: un rettangolo chiuso con punti di interesse e droni.

    Responsabilita' del modello (in ordine di scrittura):
      1. creare lo spazio, i punti e i droni         
      2. aggiornare l'occupancy dei punti a ogni step 
      3. raccogliere le metriche di deficit       
    """

    def __init__ (
        self,
        # --- AMBIENTE: com'e' fatto il territorio ---
        width=100.0, # larghezza del territorio (asse x), in unita' di simulazione
        height=100.0, # altezza del territorio (asse y)
        n_droni=40, # quanti droni esistono, fisso per tutta la simulazione
        n_punti=12, # quanti punti di interesse creare all'inizio
        priorita_massima=3, # quota massima sorteggiabile: ogni punto chiedera' fra 1 e 3 droni
        margine_punti=0.0, # stabilisce quanto devono essere tenuti lontani dai confini i centri dei punti di interesse quando vengono generati.
        disposizione_punti="casuali",  # casuali | gruppi | sparsi | cerchio | bordi | centrali
 
        # --- SCHIERAMENTO: da dove partono i droni ---
        partenza="sparsi", # sparsi | base | alto | basso | sinistra | destra
        rumore_partenza=1.0, # dispersione attorno a base/lato di partenza

        # --- TIPO DI DRONE ---
        tipo_drone="quadricottero", # "quadricottero" | "ala_fissa"

        # --- SCALA FISICA (non cambia la dinamica, la rende interpretabile) ---
        metri_per_unita=1.0,
        secondi_per_step=1.0,
 
        # --- GEOMETRIA DEL DRONE: distanze, tutte nelle stesse unita' del mondo ---
        speed=1.0, # quanto avanza a ogni passo
        drone_sensing_radius=10.0, # entro questo raggio vede/comunica con gli altri droni
        point_sensing_radius=10.0, # entro questo raggio percepisce i punti
        separation=2.0, # sotto questa distanza un altro drone e' "troppo vicino" e lo scansa
        coverage_radius=8.0, # entro questo raggio da un punto, il drone lo sta PRESIDIANDO
 
        # --- PESI DELLE FORZE: quanto conta ciascuna spinta rispetto alle altre ---
        cohere=0.25, # quanto tira l'attrazione verso il punto scelto
        separate=0.015, # quanto spinge la separazione dai droni troppo vicini
        match=0.05,# quanto tira l'allineamento alla rotta media dei vicini
        boundary=0.3, # quanto spinge il bordo verso l'interno
        margin=12.0, # a che distanza dal bordo la spinta del bordo si accende
        quadcopter_margin=2.0,# margine ridotto: il quadricottero puo' virare sul posto

        # --- DECISIONE ED ESPLORAZIONE ---
        beta=0.05, # costo del viaggio: quanto penalizza la distanza nella scelta del punto
        explore=0.2, # quanto sterza a caso quando non vede nessun punto da servire

        # --- RILASCIO DA SOVRAFFOLLAMENTO ---
        # BaseDrone/ala fissa: massimo ritardo casuale della logica di rilascio comune.
        # Quadricottero: ampiezza massima della parte pseudocasuale del timer dei support.
        release_delay_max_steps=5,

        # --- COORDINAMENTO QUADRICOTTERO ---
        # Piccola deviazione applicata da un explorer quando incontra un presidio
        # gia' soddisfatto. E' in gradi solo per renderne immediata l'interpretazione.
        avoid_angle_degrees=10.0,

        # Quanto all'interno del bordo della coverage si fermano i support.
        support_inset=2.0,

        raccogli_agenti=False, # Quando è True, il datacollector registra anche dati per agente
        seed=None, # seme casuale: stesso seed = simulazione identica
    ):
        # --- SEME ---
        # In Mesa 3.5.1 si passa rng=, NON seed=: 'seed=' funziona ancora ma emette FutureWarnin.
        # Dopo questa riga esistono:
        #   self.rng -> numpy Generator (lo usa l'esplorazione dei droni)
        #   self.random -> random.Random della stdlib (lo vuole ContinuousSpace)
        # entrambi derivati dallo stesso seme: stesso seed = stessa simulazione.
        super().__init__(rng=seed)

        # --- VINCOLI DI VALIDITA' ---

        if tipo_drone not in DRONE_CLASS_BY_TYPE:
            raise ValueError(f"tipo_drone='{tipo_drone}' sconosciuto: usa {tuple(DRONE_CLASS_BY_TYPE)}.")

        if disposizione_punti not in DISPOSIZIONI_PUNTI:
            raise ValueError(
                f"disposizione_punti='{disposizione_punti}' sconosciuta: "
                f"usa {DISPOSIZIONI_PUNTI}."
            )

        if margine_punti < 0 or 2 * margine_punti >= min(width, height):
            raise ValueError(
                f"margine_punti ({margine_punti}) non valido per un mondo {width}x{height}."
            )

        # Chi copre un punto deve anche percepirlo.
        if coverage_radius > point_sensing_radius:
            raise ValueError(
                f"coverage_radius ({coverage_radius}) > "
                f"point_sensing_radius ({point_sensing_radius}): "
                "un drone potrebbe coprire un punto senza percepirlo."
            )

        # Owner e support devono potersi vedere reciprocamente. 
        # I support si fermano al raggio coverage_radius - support_inset; 
        # non e' invece necessario imporre un ordine tra i raggi di percezione di droni e punti.
        support_operating_radius = coverage_radius - support_inset
        if (tipo_drone == "quadricottero" and drone_sensing_radius + 1e-9 < support_operating_radius):
            raise ValueError(
                f"drone_sensing_radius ({drone_sensing_radius}) < "
                f"coverage_radius - support_inset ({support_operating_radius}): "
                "owner e support potrebbero non riuscire a comunicare."
            )

        # Il vincolo di raggio di virata appartiene SOLO all'ala fissa.
        if tipo_drone == "ala_fissa":
            if cohere <= 0:
                raise ValueError("cohere deve essere > 0 per il drone ad ala fissa.")
            self.raggio_virata = speed / cohere
            if self.raggio_virata >= coverage_radius:
                raise ValueError(
                    f"raggio di virata speed/cohere = {self.raggio_virata:.2f} >= "
                    f"coverage_radius = {coverage_radius}: l'ala fissa orbiterebbe fuori dalla zona."
                )
        else:
            self.raggio_virata = None

        for nome_margine, valore_margine in (("margin", margin),("quadcopter_margin", quadcopter_margin)):
            if valore_margine <= 0 or 2 * valore_margine >= min(width, height):
                raise ValueError(
                    f"{nome_margine} ({valore_margine}) non valido per un mondo "
                    f"{width}x{height}: deve essere positivo e la forza di "
                    "confine non deve agire ovunque."
                )

        if 2 * margin >= min(width, height):
            raise ValueError(
                f"margin ({margin}) troppo grande per un mondo {width}x{height}: "
                "la forza di confine agirebbe ovunque."
            )

        if metri_per_unita <= 0 or secondi_per_step <= 0:
            raise ValueError("metri_per_unita e secondi_per_step devono essere > 0.")

        if release_delay_max_steps < 0:
            raise ValueError("release_delay_max_steps deve essere >= 0.")

        if avoid_angle_degrees < 0:
            raise ValueError("avoid_angle_degrees deve essere >= 0.")

        if tipo_drone == "quadricottero" and not (0.0 < support_inset < coverage_radius):
            raise ValueError("support_inset deve essere > 0 e < coverage_radius.")

        # --- GEOMETRIA ---
        # width/height devono stare SUL MODELLO perche' Drone._boundary_force() legge self.model.width / self.model.height,
        # e il np.clip finale di Agents.py legge gli stessi due nomi.
        # Se li chiami in un altro modo (self.larghezza...) il drone muore con AttributeError al primo step,
        # non alla costruzione: l'errore arriva tardi e sembra scollegato dalla causa.
        self.width = float(width)
        self.height = float(height)
        self.n_droni = int(n_droni)
        self.n_punti = int(n_punti)
        self.tipo_drone = tipo_drone
        self.drone_class = DRONE_CLASS_BY_TYPE[tipo_drone]
        self.disposizione_punti = disposizione_punti
        self.margine_punti = float(margine_punti)

        # Percezione e comunicazione tra droni coincidono per ipotesi di modello.
        self.communication_radius = float(drone_sensing_radius)

        # Scala fisica: per ora e' metadato esplicito, non altera le equazioni.
        self.metri_per_unita = float(metri_per_unita)
        self.secondi_per_step = float(secondi_per_step)
        self.tempo_simulato_s = 0.0
        self.velocita_reale_m_s = float(speed) * self.metri_per_unita / self.secondi_per_step

        # --- SPAZIO CONTINUO ---
        # dimensions: una riga per asse -> [[x_min, x_max], [y_min, y_max]].
        # L'ORIGINE DEVE ESSERE 0: _boundary_force() confronta la posizione con 'margin' assumendo che il bordo basso sia 0, e il clip usa [eps, width-eps].
        # Con [[50, 150], ...] i droni si comporterebbero come se il bordo fosse a 0 -> forza di confine sbagliata e ValueError dallo spazio.
        # torus=False: territorio chiuso e delimitato (una piazza, non Pac-Man).
        # random=self.random: se lo ometti Mesa emette UserWarning e usa un RNG non seminato -> simulazione non riproducibile.
        # n_agents: solo un suggerimento di pre-allocazione dell'array interno delle posizioni. 
        # Se lo sbagli l'array viene ridimensionato da solo: costa un po' di tempo, non e' un bug.
        self.space = ContinuousSpace(
            [[0.0, self.width], [0.0, self.height]],
            torus=False,
            random=self.random,
            n_agents=self.n_droni + self.n_punti,
        )

        # --- PUNTI DI INTERESSE: CONDIZIONE INIZIALE ---
        # Come per ``partenza`` dei droni, la disposizione dei punti e' una scelta
        # iniziale del mondo. Non cambia durante la simulazione.
        # Tutte le modalita' usano self.rng: stesso seed + stessi parametri = stesso
        # territorio, anche quando la geometria e' pseudo-randomica.
        posizioni_punti = self._genera_posizioni_punti(disposizione=disposizione_punti, margine=margine_punti)

        # Priorita' = QUOTA ASSOLUTA di droni desiderata, quindi a VALORI INTERI
        # (il tipo resta float, e va bene cosi').
        # Con una quota frazionaria (es. 2.5) il punto smette di attrarre al terzo
        # drone, quindi ne CONSUMA 3 pur dichiarandone 2.5: domanda_totale e
        # deficit_incomprimibile qui sotto risulterebbero sottostimati, e il confronto
        # con l'oracolo centralizzato misurerebbe uno scarto che e' solo
        # arrotondamento. Con quote intere lo scarto e' zero.
        # ATTENZIONE a integers(): l'estremo alto e' ESCLUSO. integers(1, 3) da' 1 o 2,
        # mai 3 -> serve priorita_massima + 1.
        priorita_punti = np.zeros(self.n_punti)
        for i in range(self.n_punti):
            priorita_punti[i] = self.rng.integers(1, priorita_massima + 1)

        # create_agents(model, n, *args, **kwargs): passa SEMPRE model per primo, ed
        # e' il motivo per cui TargetAgent.__init__ inizia con 'model' e poi inverte
        # in super().__init__(space, model).
        # Un argomento viene DISTRIBUITO (uno per agente) se e' list/tuple/ndarray di
        # lunghezza esattamente n, altrimenti viene RIPETUTO uguale per tutti.
        # Percio' self.space (oggetto, non sequenza) arriva uguale a tutti: giusto.
        # E percio' le posizioni sono un array (n, 2) e non una tupla condivisa: con
        # n=2 una tupla (x, y) verrebbe scambiata per "una posizione per agente" e i
        # due punti riceverebbero position=x e position=y. Verificato eseguendolo.
        # list(...) perche' create_agents ritorna un AgentSet: lo congelo in una lista
        # ordinata e stabile, che e' quello che servira' alle metriche.
        self.target_agents = list(
            TargetAgent.create_agents(
                self,
                self.n_punti,
                self.space,
                position=posizioni_punti,
                priority=priorita_punti,
            )
        )

        # idx = indice progressivo per colori e metriche. Lo assegno qui e non nel
        # costruttore perche' il punto da solo non sa in che ordine e' stato creato.
        for i in range(self.n_punti):
            self.target_agents[i].idx = i

        # --- DIAGNOSTICA STRUTTURALE ---
        # domanda_totale = somma delle quote = quanti "posti drone" chiede il
        # territorio. Con priorita' come quota assoluta, se domanda_totale > n_droni
        # il sistema e' in DEFICIT STRUTTURALE: nessun algoritmo, nemmeno l'oracolo
        # centralizzato, puo' azzerare il deficit residuo.
        # deficit_incomprimibile e' il PAVIMENTO della metrica principale: senza
        # saperlo si legge un grafico che non scende a zero e si pensa a un bug del
        # coordinamento, quando invece mancano proprio i droni.
        self.domanda_totale = 0.0
        for punto in self.target_agents:
            self.domanda_totale += punto.priority
        self.deficit_incomprimibile = max(0.0, self.domanda_totale - self.n_droni)

        # ATTENZIONE: deficit_incomprimibile e' un pavimento CONDIZIONATO. Assume che
        # ogni drone fornisca al massimo una unita' di occupazione. Ma l'occupazione
        # totale e' la somma di n_covered, e un drone in mezzo a due zone conta per
        # entrambe: con zone sovrapposte 10 droni possono produrre 13 di occupazione,
        # e il deficit residuo scende SOTTO il pavimento. Non e' un errore di calcolo:
        # e' l'ipotesi che salta. Verificato - senza sovrapposizioni, zero violazioni.
        # zone_sovrapposte dice se il pavimento e' valido per QUESTO mondo: 0 = si'.
        self.zone_sovrapposte = 0
        for i in range(self.n_punti):
            for j in range(i + 1, self.n_punti):
                d = np.linalg.norm(self.target_agents[i].position - self.target_agents[j].position)
                if d < 2.0 * coverage_radius:
                    self.zone_sovrapposte += 1

        # --- DRONI: POSIZIONI E DIREZIONI INIZIALI ---
        # Le modalita' laterali distribuiscono i droni lungo un lato e li orientano
        # inizialmente verso l'interno. Il piccolo rumore e' ortogonale al bordo e
        # impedisce di collocarli tutti sulla stessa coordinata esatta.
        posizioni_droni = np.zeros((self.n_droni, 2))
        direzioni_droni = np.zeros((self.n_droni, 2))

        if partenza == "sparsi":
            for i in range(self.n_droni):
                posizioni_droni[i, 0] = self.rng.uniform(0.0, self.width)
                posizioni_droni[i, 1] = self.rng.uniform(0.0, self.height)
                angolo = self.rng.uniform(0.0, 2.0 * np.pi)
                direzioni_droni[i] = [np.cos(angolo), np.sin(angolo)]

        elif partenza == "base":
            base_x = self.width / 2.0
            base_y = self.height / 2.0
            for i in range(self.n_droni):
                posizioni_droni[i, 0] = base_x + self.rng.normal(0.0, rumore_partenza)
                posizioni_droni[i, 1] = base_y + self.rng.normal(0.0, rumore_partenza)
                angolo = self.rng.uniform(0.0, 2.0 * np.pi)
                direzioni_droni[i] = [np.cos(angolo), np.sin(angolo)]

        elif partenza in ("alto", "basso", "sinistra", "destra"):
            for i in range(self.n_droni):
                scarto = abs(self.rng.normal(0.0, rumore_partenza))

                if partenza == "sinistra":
                    posizioni_droni[i] = [min(scarto, self.width * 0.05), self.rng.uniform(0.0, self.height)]
                    direzioni_droni[i] = [1.0, 0.0]
                elif partenza == "destra":
                    posizioni_droni[i] = [self.width - min(scarto, self.width * 0.05), self.rng.uniform(0.0, self.height)]
                    direzioni_droni[i] = [-1.0, 0.0]
                elif partenza == "basso":
                    posizioni_droni[i] = [self.rng.uniform(0.0, self.width), min(scarto, self.height * 0.05)]
                    direzioni_droni[i] = [0.0, 1.0]
                else:  # alto
                    posizioni_droni[i] = [self.rng.uniform(0.0, self.width), self.height - min(scarto, self.height * 0.05)]
                    direzioni_droni[i] = [0.0, -1.0]
        else:
            raise ValueError(
                f"partenza='{partenza}' sconosciuta: usa 'sparsi', 'base', "
                "'alto', 'basso', 'sinistra' o 'destra'."
            )

        # Rete di sicurezza: tutte le posizioni devono essere strettamente interne.
        eps = 1e-6
        posizioni_droni[:, 0] = np.clip(posizioni_droni[:, 0], eps, self.width - eps)
        posizioni_droni[:, 1] = np.clip(posizioni_droni[:, 1], eps, self.height - eps)

        # Posizioni e direzioni sono array (n, 2) ESPLICITI, mai tuple condivise:
        # con n_droni=2 una tupla verrebbe scambiata per "un valore per agente".
        # La stessa _boundary_force() viene usata dalle due piattaforme, ma con
        # margini diversi: l'ala fissa deve anticipare la virata, mentre il
        # quadricottero puo' reagire negli ultimi passi vicino al confine.
        margine_confine = (quadcopter_margin if self.drone_class is QuadcopterDrone else margin)

        # I parametri comuni vengono passati a entrambe le sottoclassi; il parametro
        # di deviazione dai presidi soddisfatti viene aggiunto solo al quadricottero.
        parametri_drone = dict(
            position=posizioni_droni,
            direction=direzioni_droni,
            speed=speed,
            drone_sensing_radius=drone_sensing_radius,
            point_sensing_radius=point_sensing_radius,
            separation=separation,
            coverage_radius=coverage_radius,
            cohere=cohere,
            separate=separate,
            match=match,
            boundary=boundary,
            margin=margine_confine,
            beta=beta,
            explore=explore,
            release_delay_max_steps=release_delay_max_steps,
        )
        if self.drone_class is QuadcopterDrone:
            parametri_drone.update(avoid_angle_degrees=avoid_angle_degrees, support_inset=support_inset)

        self.drone_agents = list(
            self.drone_class.create_agents(
                self,
                self.n_droni,
                self.space,
                **parametri_drone,
            )
        )

        # Il modello tiene una copia dei parametri che gli servono per conto suo:
        # coverage_radius e' il raggio con cui contera' l'occupancy nel blocco 4.
        # Lo leggo dal parametro e non da un drone a caso, cosi' resta valido anche
        # il giorno in cui i droni diventeranno eterogenei.
        self.coverage_radius = float(coverage_radius)
        self.drone_sensing_radius = float(drone_sensing_radius)
        self.point_sensing_radius = float(point_sensing_radius)
        self.boundary_margin = float(margine_confine)

        # Fotografia iniziale. Per l'ala fissa la coverage e' geometrica; per il
        # quadricottero un drone conta soltanto dopo l'elezione OWNER/SUPPORT, quindi
        # e' corretto che un quad nato nella zona ma ancora FREE non venga contato.
        self.aggiorna_occupancy()

        # --- RACCOLTA DATI ---
        # I reporter sono i METODI qui sotto, passati come CoverageModel.nome (la funzione, non il suo risultato: niente parentesi).
        # Mesa li invoca a ogni collect passando il modello. Metodi normali e non lambda per due motivi:
        #  1) puoi chiamarli a mano da uno script di prova
        #  2) in un traceback compare il loro nome invece di un anonimo "<lambda>".
        # "deficit_incomprimibile" e' invece nella forma STRINGA: Mesa legge l'attributo omonimo del modello.
        #  E' una costante, ma averla ripetuta in ogni riga permette di disegnare la linea del pavimento nei grafici senza doverla recuperare a parte.
        reporter_modello = {
            "deficit_residuo": CoverageModel.deficit_residuo,
            "deficit_normalizzato": CoverageModel.deficit_normalizzato,
            "punti_soddisfatti": CoverageModel.punti_soddisfatti,
            "sovra_servizio": CoverageModel.sovra_servizio,
            "droni_oziosi": CoverageModel.droni_oziosi,
            "droni_in_esplorazione": CoverageModel.droni_in_esplorazione,
            "deficit_incomprimibile": "deficit_incomprimibile",
            "tempo_simulato_s": "tempo_simulato_s",
        }

        # I dati per singolo agente sono spenti di default: 
        # sono n_droni + n_punti righe A OGNI PASSO (52 x 600 = 31200 righe per una sola run),
        # e in uno sweep con decine di combinazioni esplodono. Accendili quando devi ispezionare una singola simulazione, non quando ne lanci cento.
        reporter_per_tipo = None
        if raccogli_agenti:
            reporter_drone = {
                "n_covered": "n_covered",
                "exploring": "exploring",
                "tipo_drone": "tipo_drone",
                "moving": "moving",
                "release_wait_remaining": "release_wait_remaining",
            }
            if self.drone_class is QuadcopterDrone:
                reporter_drone["station_role"] = "station_role"

            reporter_per_tipo = {
                TargetAgent: {"idx": "idx", "priority": "priority", "occupancy": "occupancy"},
                self.drone_class: reporter_drone,
            }

        self.datacollector = ThreadSafeDataCollector(model_reporters=reporter_modello, agenttype_reporters=reporter_per_tipo)

        # Prima riga: lo stato a t=0, prima che si muova qualunque cosa.
        # Serve come riferimento per il transitorio: senza, il primo dato che hai e' gia' il risultato di un passo e non sai da dove sei partito.
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # GENERAZIONE DELLE CONDIZIONI INIZIALI DEI PUNTI
    # ------------------------------------------------------------------

    def _genera_posizioni_punti(self, disposizione, margine):
        """Costruisce le posizioni iniziali dei punti di interesse.

        Le modalita' sono volutamente semplici e leggibili: servono a creare mondi
        con geometrie diverse, non a modellare un processo dinamico di formazione
        dei punti. Tutta la casualita' passa da ``self.rng``, quindi e' riproducibile
        tramite ``seed``.
        """
        n = self.n_punti
        posizioni = np.zeros((n, 2), dtype=float)
        if n == 0:
            return posizioni

        # Rettangolo effettivamente disponibile dopo margine_punti.
        x_min = float(margine)
        x_max = self.width - float(margine)
        y_min = float(margine)
        y_max = self.height - float(margine)
        larghezza = x_max - x_min
        altezza = y_max - y_min

        if disposizione == "casuali":
            # Baseline originale: punti indipendenti e uniformi in tutto il territorio.
            for i in range(n):
                posizioni[i, 0] = self.rng.uniform(x_min, x_max)
                posizioni[i, 1] = self.rng.uniform(y_min, y_max)

        elif disposizione == "gruppi":
            # Tre cluster al massimo. I centri sono casuali, ma ogni cluster riceve almeno un punto quando n lo permette.
            # La dispersione e' il 5% della dimensione piu' piccola del rettangolo disponibile.
            n_gruppi = min(3, n)
            centri = np.zeros((n_gruppi, 2), dtype=float)

            # Evito di mettere il centro del cluster proprio sul margine, cosi' il rumore gaussiano non viene tagliato quasi tutto da un solo lato.
            padding_x = 0.15 * larghezza
            padding_y = 0.15 * altezza
            for g in range(n_gruppi):
                centri[g, 0] = self.rng.uniform(x_min + padding_x, x_max - padding_x)
                centri[g, 1] = self.rng.uniform(y_min + padding_y, y_max - padding_y)

            assegnazioni = np.arange(n) % n_gruppi
            self.rng.shuffle(assegnazioni)
            sigma = 0.05 * min(larghezza, altezza)

            for i in range(n):
                centro = centri[assegnazioni[i]]
                posizioni[i] = centro + self.rng.normal(0.0, sigma, size=2)

        elif disposizione == "sparsi":
            # Suddivido il territorio in celle e uso una sola posizione per cella.
            # Un piccolo jitter evita una griglia perfettamente artificiale, mantenendo pero' i punti molto piu' separati rispetto alla baseline casuale.
            rapporto = larghezza / altezza
            n_colonne = max(1, int(np.ceil(np.sqrt(n * rapporto))))
            n_righe = max(1, int(np.ceil(n / n_colonne)))

            passo_x = larghezza / n_colonne
            passo_y = altezza / n_righe
            celle = []
            for riga in range(n_righe):
                for colonna in range(n_colonne):
                    celle.append(
                        [
                            x_min + (colonna + 0.5) * passo_x,
                            y_min + (riga + 0.5) * passo_y,
                        ]
                    )

            celle = np.asarray(celle, dtype=float)
            self.rng.shuffle(celle)
            jitter_x = 0.15 * passo_x
            jitter_y = 0.15 * passo_y

            for i in range(n):
                posizioni[i, 0] = celle[i, 0] + self.rng.uniform(-jitter_x, jitter_x)
                posizioni[i, 1] = celle[i, 1] + self.rng.uniform(-jitter_y, jitter_y)

        elif disposizione == "cerchio":
            # Punti equispaziati su una circonferenza centrata nel territorio.
            # L'angolo iniziale e' casuale: la forma resta un cerchio, ma il seed decide la rotazione globale della configurazione.
            centro = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
            raggio = 0.35 * min(larghezza, altezza)
            fase = self.rng.uniform(0.0, 2.0 * np.pi)

            for i in range(n):
                angolo = fase + (2.0 * np.pi * i / n)
                posizioni[i] = centro + raggio * np.array(
                    [np.cos(angolo), np.sin(angolo)]
                )

        elif disposizione == "bordi":
            # Distribuzione vicino ai quattro bordi. 
            # Le assegnazioni ai lati vengono bilanciate e poi mischiate, cosi' non dipendono dall'indice del punto.
            fascia = 0.08 * min(larghezza, altezza)
            lati = np.arange(n) % 4
            self.rng.shuffle(lati)

            for i, lato in enumerate(lati):
                scarto = self.rng.uniform(0.0, fascia)
                if lato == 0:      # sinistra
                    posizioni[i] = [x_min + scarto, self.rng.uniform(y_min, y_max)]
                elif lato == 1:    # destra
                    posizioni[i] = [x_max - scarto, self.rng.uniform(y_min, y_max)]
                elif lato == 2:    # basso
                    posizioni[i] = [self.rng.uniform(x_min, x_max), y_min + scarto]
                else:              # alto
                    posizioni[i] = [self.rng.uniform(x_min, x_max), y_max - scarto]

        elif disposizione == "centrali":
            # Tutti i punti cadono nel rettangolo centrale, largo/alto il 30% del territorio disponibile.
            # E' una concentrazione centrale, non un cluster puntiforme: i punti mantengono comunque una certa dispersione.
            centro_x = (x_min + x_max) / 2.0
            centro_y = (y_min + y_max) / 2.0
            semi_x = 0.15 * larghezza
            semi_y = 0.15 * altezza

            for i in range(n):
                posizioni[i, 0] = self.rng.uniform(centro_x - semi_x, centro_x + semi_x)
                posizioni[i, 1] = self.rng.uniform(centro_y - semi_y, centro_y + semi_y)

        else:
            # In pratica questo ramo e' protetto dal guardrail in __init__,
            # ma tenerlo rende la funzione autonoma e piu' facile da testare direttamente.
            raise ValueError(f"disposizione punti non riconosciuta: {disposizione}")

        # Sicurezza comune a tutte le modalita':
        # il rumore di cluster/jitter non puo' portare punti fuori dal rettangolo consentito da margine_punti.
        posizioni[:, 0] = np.clip(posizioni[:, 0], x_min, x_max)
        posizioni[:, 1] = np.clip(posizioni[:, 1], y_min, y_max)
        return posizioni

    # ------------------------------------------------------------------
    # METRICHE
    # ------------------------------------------------------------------

    def deficit_residuo(self):
        """Quanti droni mancano in totale perche' ogni punto raggiunga la sua quota.

        E' LA metrica: quella che il coordinamento deve minimizzare.
        Il max(0, ...) non e' cosmetico. Senza, un punto sovra-servito darebbe un
        contributo NEGATIVO che compenserebbe un punto sguarnito altrove: un mondo con
        meta' punti deserti e meta' affollati risulterebbe perfetto.
        """
        totale = 0.0
        for punto in self.target_agents:
            mancante = punto.priority - punto.occupancy
            if mancante > 0:
                totale += mancante
        return totale

    def deficit_normalizzato(self):
        """deficit_residuo come frazione della domanda totale.

        Serve a CONFRONTARE run con territori diversi: un deficit di 6 su una domanda
        di 26 e uno di 6 su una domanda di 60 non sono la stessa prestazione. Negli
        sweep in cui varii n_punti o priorita_massima il deficit grezzo non e'
        confrontabile fra una combinazione e l'altra: questo si'.
        """
        if self.domanda_totale <= 0:
            return 0.0
        return self.deficit_residuo() / self.domanda_totale

    def punti_soddisfatti(self):
        """Quanti punti hanno raggiunto (o superato) la loro quota.

        Guarda la stessa cosa del deficit ma per TESTE invece che per quantita': dice
        se il sistema serve pochi punti bene o molti punti a meta'. Due configurazioni
        con lo stesso deficit residuo possono avere qui numeri molto diversi.
        """
        n = 0
        for punto in self.target_agents:
            if punto.occupancy >= punto.priority:
                n += 1
        return n

    def sovra_servizio(self):
        """Droni in eccesso su punti gia' pieni.

        E' l'altra faccia dello spreco, e NON e' il complemento dei droni oziosi: un
        drone ozioso non presidia niente, un drone in sovra-servizio presidia qualcosa
        che era gia' a posto. Sprechi diversi, rimedi diversi.
        """
        totale = 0.0
        for punto in self.target_agents:
            eccesso = punto.occupancy - punto.priority
            if eccesso > 0:
                totale += eccesso
        return totale

    def droni_oziosi(self):
        """Droni che in questo istante non stanno presidiando nessun punto."""
        n = 0
        for drone in self.drone_agents:
            if drone.n_covered == 0:
                n += 1
        return n

    def droni_in_esplorazione(self):
        """Droni che non hanno proprio nessun punto bisognoso in vista.

        E' un SOTTOINSIEME dei droni oziosi: un ozioso puo' essere in viaggio verso un
        punto che ha gia' scelto, un esploratore no. La distanza fra i due numeri e'
        diagnostica: se sono quasi uguali il problema e' che i droni non TROVANO i
        punti (l'esplorazione casuale e' debole); se sono molto diversi, i punti li
        trovano ma ci mettono troppo ad arrivarci.
        """
        n = 0
        for drone in self.drone_agents:
            if drone.exploring:
                n += 1
        return n

    def aggiorna_occupancy(self):
        """
        Riconta, per ogni punto, quanti droni lo stanno presidiando adesso.

        Non lo calcolano i punti da soli: il punto non sa chi ha intorno. Lo fa il
        modello, che vede tutti, e lo SCRIVE dentro ogni punto. I droni NON lo
        leggono per decidere: usano la propria stima locale costruita tramite
        percezione e comunicazione. ``occupancy`` resta la ground truth per metriche
        e visualizzazione. Per il quadricottero la sola presenza geometrica non
        basta: contano esclusivamente OWNER e SUPPORT, che per politica sono fermi.
        L'ala fissa conserva invece la definizione geometrica originale.
        """
        # Azzero prima tutti i contatori dei droni: n_covered viene ricostruito da zero a ogni passo, non incrementato all'infinito.
        for drone in self.drone_agents:
            drone.n_covered = 0 # n_covered indica quanti punti sta coprendo fisicamente quel drone.

        for punto in self.target_agents:
            # Una chiamata per punto: distanze da QUESTO punto a TUTTI i droni.
            # L'array torna nello stesso ordine di self.drone_agents (verificato eseguendolo), quindi distanze[i] e' la distanza del drone i.
            distanze, _ = self.space.calculate_distances(punto.position, agents=self.drone_agents)

            presidianti = 0
            for i in range(len(self.drone_agents)):
                drone = self.drone_agents[i]

                # Se è fuori dalla coverage, non viene contato.
                if distanze[i] > self.coverage_radius:
                    continue
                # Per i quadricotteri viene applicato anche il controllo del ruolo.
                # Un quadricottero viene contato soltanto se: è dentro coverage_radius E station_role è owner oppure support.
                if (self.drone_class is QuadcopterDrone and getattr(drone, "station_role", None) not in ("owner", "support")):
                    continue

                presidianti += 1
                # NOTA: un drone entro coverage_radius di DUE punti vicini conta per entrambi.
                # Percio' la somma delle occupancy puo' superaren_droni: e' voluto (un drone in mezzo a due zone le presidia davvero entrambe),
                # ma va ricordato leggendo le metriche.
                drone.n_covered += 1 # quanti punti sono coperti da questo drone

            punto.occupancy = presidianti # quanti droni coprono questo punto

    def step(self):
        """Uno step del mondo, esplicitamente separato in fasi.

        Non c'e' parallelismo reale: ogni fase termina per TUTTI i droni prima che
        inizi la successiva. ``shuffle_do`` randomizza soltanto l'ordine interno alla
        fase; i metodi sono progettati per scrivere il proprio stato, non quello altrui.
        """
        droni = self.agents_by_type[self.drone_class]

        # 1. Il territorio cambia.
        self.agents_by_type[TargetAgent].do("step")

        # 2. Tutti costruiscono una fotografia locale dello stesso stato spaziale.
        droni.shuffle_do("perceive")

        # 3. Tutti leggono le fotografie dei vicini e costruiscono la propria stima.
        droni.shuffle_do("communicate")

        # 4. Tutti scelgono il target senza ancora modificare target/ruolo corrente.
        droni.shuffle_do("decide_target")

        # 5. La fase esiste per tutte le piattaforme: BaseDrone la definisce come no-op,
        #    QuadcopterDrone la specializza con owner/support e rilascio da sovraffollamento.
        droni.shuffle_do("decide_station")

        # 6. Le decisioni diventano stato corrente.
        droni.shuffle_do("commit_decision")

        # 7. Movimento fisico.
        droni.shuffle_do("move")

        # 8. Ground truth: il modello riconta i presidi reali DOPO il movimento.
        self.aggiorna_occupancy()

        # 9. Tempo fisico e misura. La riga t=0 e' stata raccolta in __init__.
        self.tempo_simulato_s += self.secondi_per_step
        self.datacollector.collect(self)
