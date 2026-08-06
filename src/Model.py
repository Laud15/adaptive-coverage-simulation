import numpy as np
import pandas as pd
import mesa
from mesa.experimental.continuous_space import ContinuousSpace

from Agents import Drone, TargetAgent


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

    def __init__(
        self,
        # --- AMBIENTE: com'e' fatto il territorio ---
        width=100.0,           # larghezza del territorio (asse x), in unita' di simulazione
        height=100.0,          # altezza del territorio (asse y)
        n_droni=40,            # quanti droni esistono, fisso per tutta la simulazione
        n_punti=12,            # quanti punti di interesse creare all'inizio
        priorita_massima=3,    # quota massima sorteggiabile: ogni punto chiedera' fra 1 e 3 droni
        margine_punti=0.0,     # quanto lontano dal bordo devono nascere i punti (0 = ovunque)
 
        # --- SCHIERAMENTO: da dove partono i droni ---
        partenza="sparsi",     # "sparsi" = gia' distribuiti sul territorio | "base" = tutti dal centro
        rumore_partenza=1.0,   # quanto si sparpagliano attorno alla base (usato solo con "base")
 
        # --- GEOMETRIA DEL DRONE: distanze, tutte nelle stesse unita' del mondo ---
        speed=1.0,             # quanto avanza a ogni passo
        vision=10.0,           # entro questo raggio VEDE GLI ALTRI DRONI (per separarsi e allinearsi)
        sensing_radius=25.0,   # entro questo raggio PERCEPISCE I PUNTI (per sceglierne uno)
        separation=2.0,        # sotto questa distanza un altro drone e' "troppo vicino" e lo scansa
        coverage_radius=8.0,   # entro questo raggio da un punto, il drone lo sta PRESIDIANDO
 
        # --- PESI DELLE FORZE: quanto conta ciascuna spinta rispetto alle altre ---
        cohere=0.25,           # quanto tira l'attrazione verso il punto scelto
        separate=0.015,        # quanto spinge la separazione dai droni troppo vicini
        match=0.05,            # quanto tira l'allineamento alla rotta media dei vicini
        boundary=0.3,          # quanto spinge il bordo verso l'interno
        margin=12.0,           # a che distanza dal bordo la spinta del bordo si accende
 
        # --- DECISIONE ED ESPLORAZIONE ---
        beta=0.05,             # costo del viaggio: quanto penalizza la distanza nella scelta del punto
        explore=0.2,           # quanto sterza a caso quando non vede nessun punto da servire
        raccogli_agenti=False, # Quando è True, il datacollector registra a ogni passo anche i dati dei singoli agenti, non solo le metriche aggregate del modello:
        seed=None,             # seme casuale: stesso seed = simulazione identica
    ):
        # --- SEME ---
        # In Mesa 3.5.1 si passa rng=, NON seed=: 'seed=' funziona ancora ma emette FutureWarnin.
        # Dopo questa riga esistono:
        #   self.rng    -> numpy Generator (lo usa l'esplorazione dei droni)
        #   self.random -> random.Random della stdlib (lo vuole ContinuousSpace)
        # entrambi derivati dallo stesso seme: stesso seed = stessa simulazione.
        super().__init__(rng=seed)

        # --- VINCOLI DI VALIDITA' ---

        # 1. vision <= sensing_radius. Agents.py fa UNA sola chiamata
        #    get_neighbors_in_radius(sensing_radius) e poi filtra i droni a vision.
        #    Se vision > sensing_radius, i droni fra i due raggi non entrano nemmeno
        #    nella lista: separazione e allineamento perdono vicini SENZA errore.
        if vision > sensing_radius:
            raise ValueError(
                f"vision ({vision}) > sensing_radius ({sensing_radius}): la percezione "
                f"dei droni vicini verrebbe troncata in silenzio. Vedi Agents.py, "
                f"blocco PERCEZIONE."
            )

        # 2. raggio di virata R = speed/cohere: la curva piu' stretta che il drone
        #    riesce fisicamente a fare. Se R supera coverage_radius, il drone non
        #    chiude la curva dentro la zona e ORBITA fuori dal punto (milling):
        #    l'occupancy oscilla e il punto non risulta mai davvero presidiato.
        #    Il drone orbita SEMPRE attorno al target. Se il raggio di
        #    virata supera coverage_radius, orbita fuori dalla zona e non presidia piu'.
        #    si mette l'uguale perchè il caso in cui il drone girà esattamente sulla circonferenza è irrealistico
        self.raggio_virata = speed / cohere
        if self.raggio_virata >= coverage_radius:
            raise ValueError(
                f"raggio di virata speed/cohere = {self.raggio_virata:.1f} > "
                f"coverage_radius = {coverage_radius}: i droni orbiterebbero FUORI "
                f"dalla zona invece di presidiarla."
            )

        # 3. coverage_radius <= sensing_radius. Il drone si scomputa dall'occupancy
        #    solo dei punti che PERCEPISCE (la correzione sta dentro il ciclo sui punti
        #    percepiti). Se coverage_radius superasse sensing_radius, esisterebbero
        #    droni contati come presidianti di un punto che non vedono: per loro lo
        #    scomputo non scatta e l'oscillazione ritorna, in silenzio.
        if coverage_radius > sensing_radius:
            raise ValueError(
                f"coverage_radius ({coverage_radius}) > sensing_radius ({sensing_radius}): "
                f"ci sarebbero droni che presidiano un punto senza percepirlo."
            )

        # 4. le due fasce di confine non devono coprire tutto il mondo, altrimenti
        #    non esiste una regione in cui i droni si muovono liberi e stai
        #    misurando il comportamento del bordo, non quello dell'algoritmo.
        if 2 * margin >= min(width, height):
            raise ValueError(
                f"margin ({margin}) troppo grande per un mondo {width}x{height}: "
                f"la forza di confine agirebbe ovunque."
            )

        # --- GEOMETRIA ---
        # width/height devono stare SUL MODELLO perche' Drone.boundary_force() legge
        # self.model.width / self.model.height, e il np.clip finale di Agents.py legge
        # gli stessi due nomi. Se li chiami in un altro modo (self.larghezza...) il
        # drone muore con AttributeError al primo step, non alla costruzione: l'errore
        # arriva tardi e sembra scollegato dalla causa.
        self.width = float(width)
        self.height = float(height)
        self.n_droni = int(n_droni)
        self.n_punti = int(n_punti)

        # --- SPAZIO CONTINUO ---
        # dimensions: una riga per asse -> [[x_min, x_max], [y_min, y_max]].
        # L'ORIGINE DEVE ESSERE 0: boundary_force() confronta la posizione con
        # 'margin' assumendo che il bordo basso sia 0, e il clip usa [eps, width-eps].
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

        # --- PUNTI DI INTERESSE ---
        # Posizioni: uniformi nel rettangolo. Ciclo esplicito e non
        # rng.uniform(low=[...], high=[...]) perche' i due assi hanno estensione
        # DIVERSA (width != height): scritto cosi' l'asimmetria e' sotto gli occhi e
        # non puoi invertire per sbaglio width con height dentro una lista.
        # margine_punti=0 -> i punti possono nascere attaccati al bordo (decisione di
        # modello: e' ammesso, il drone che ci orbita intorno e' un fenomeno noto).
        # Alzarlo serve solo a ISOLARE quel fenomeno negli esperimenti controllati.
        posizioni_punti = np.zeros((self.n_punti, 2))
        for i in range(self.n_punti):
            posizioni_punti[i, 0] = self.rng.uniform(
                margine_punti, self.width - margine_punti
            )
            posizioni_punti[i, 1] = self.rng.uniform(
                margine_punti, self.height - margine_punti
            )

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
                d = np.linalg.norm(
                    self.target_agents[i].position - self.target_agents[j].position
                )
                if d < 2.0 * coverage_radius:
                    self.zone_sovrapposte += 1

        # --- DRONI: POSIZIONI INIZIALI ---
        # Lo schieramento iniziale NON e' un dettaglio: e' la condizione al contorno
        # del transitorio, quindi decide quanto vale la "rapidita' di riassestamento"
        # che i relatori vogliono misurare. Va dichiarato ed e' un parametro
        # sperimentale, non un default nascosto.
        #   "sparsi" -> i droni sono gia' distribuiti sul territorio. Baseline neutra:
        #               nessuna configurazione privilegiata, si arriva presto al
        #               regime. E' quello che vuoi per studiare la REAZIONE a un
        #               cambiamento del territorio (il transitorio iniziale sporca poco).
        #   "base"   -> tutti da una stazione al centro. E' il caso realistico del
        #               dispiegamento da zero e da' un transitorio lungo e leggibile:
        #               e' quello che vuoi per misurare il COPRIMENTO iniziale.
        posizioni_droni = np.zeros((self.n_droni, 2))
        if partenza == "sparsi":
            for i in range(self.n_droni):
                posizioni_droni[i, 0] = self.rng.uniform(0.0, self.width)
                posizioni_droni[i, 1] = self.rng.uniform(0.0, self.height)

        elif partenza == "base":
            base_x = self.width / 2.0
            base_y = self.height / 2.0
            for i in range(self.n_droni):
                # IL RUMORE NON E' COSMETICO. Due droni esattamente sovrapposti hanno
                # vettore di separazione [0,0]: la regola di separazione non li spinge
                # via, e se hanno anche la stessa direzione percorrono la STESSA
                # traiettoria per sempre. Avresti n droni che valgono come uno solo,
                # senza nessun errore che te lo segnali.
                posizioni_droni[i, 0] = base_x + self.rng.normal(0.0, rumore_partenza)
                posizioni_droni[i, 1] = base_y + self.rng.normal(0.0, rumore_partenza)
        else:
            raise ValueError(f"partenza='{partenza}' sconosciuta: usa 'sparsi' o 'base'.")

        # --- DRONI: DIREZIONI INIZIALI ---
        # Sorteggio l'ANGOLO e poi prendo (cos, sin), NON due componenti a caso da
        # normalizzare dopo: componenti uniformi in un quadrato, una volta
        # normalizzate, si addensano sulle diagonali. Avresti uno stormo che parte
        # con un bias direzionale sistematico, difficile da vedere e impossibile da
        # spiegare quando i risultati sono anisotropi.
        direzioni_droni = np.zeros((self.n_droni, 2))
        for i in range(self.n_droni):
            angolo = self.rng.uniform(0.0, 2.0 * np.pi)
            direzioni_droni[i, 0] = np.cos(angolo)
            direzioni_droni[i, 1] = np.sin(angolo)

        # Posizioni e direzioni sono array (n, 2) ESPLICITI, mai tuple condivise:
        # con n_droni=2 una tupla verrebbe scambiata per "un valore per agente".
        self.drone_agents = list(
            Drone.create_agents(
                self,
                self.n_droni,
                self.space,
                position=posizioni_droni,
                direction=direzioni_droni,
                speed=speed,
                vision=vision,
                sensing_radius=sensing_radius,
                separation=separation,
                coverage_radius=coverage_radius,
                cohere=cohere,
                separate=separate,
                match=match,
                boundary=boundary,
                margin=margin,
                beta=beta,
                explore=explore,
            )
        )

        # Il modello tiene una copia dei parametri che gli servono per conto suo:
        # coverage_radius e' il raggio con cui contera' l'occupancy nel blocco 4.
        # Lo leggo dal parametro e non da un drone a caso, cosi' resta valido anche
        # il giorno in cui i droni diventeranno eterogenei.
        self.coverage_radius = coverage_radius

        # Fotografia iniziale dei presidi: alcuni droni potrebbero gia' nascere dentro
        # la zona di un punto. Senza questa chiamata l'occupancy resterebbe 0 fino alla
        # fine del primo step, e al primo giro TUTTI i punti sembrerebbero sguarniti:
        # i droni prenderebbero la prima decisione su un'informazione falsa.
        self.aggiorna_occupancy()

        # --- RACCOLTA DATI ---
        # I reporter sono i METODI qui sotto, passati come CoverageModel.nome (la
        # funzione, non il suo risultato: niente parentesi). Mesa li invoca a ogni
        # collect passando il modello. Metodi normali e non lambda per due motivi:
        # puoi chiamarli a mano da uno script di prova, e in un traceback compare il
        # loro nome invece di un anonimo "<lambda>".
        # "deficit_incomprimibile" e' invece nella forma STRINGA: Mesa legge
        # l'attributo omonimo del modello. E' una costante, ma averla ripetuta in ogni
        # riga permette di disegnare la linea del pavimento nei grafici senza doverla
        # recuperare a parte.
        reporter_modello = {
            "deficit_residuo": CoverageModel.deficit_residuo,
            "deficit_normalizzato": CoverageModel.deficit_normalizzato,
            "punti_soddisfatti": CoverageModel.punti_soddisfatti,
            "sovra_servizio": CoverageModel.sovra_servizio,
            "droni_oziosi": CoverageModel.droni_oziosi,
            "droni_in_esplorazione": CoverageModel.droni_in_esplorazione,
            "deficit_incomprimibile": "deficit_incomprimibile",
        }

        # I dati per singolo agente sono spenti di default: sono n_droni + n_punti
        # righe A OGNI PASSO (52 x 600 = 31200 righe per una sola run), e in uno sweep
        # con decine di combinazioni esplodono. Accendili quando devi ispezionare una
        # singola simulazione, non quando ne lanci cento.
        reporter_per_tipo = None
        if raccogli_agenti:
            reporter_per_tipo = {
                TargetAgent: {"idx": "idx", "priority": "priority", "occupancy": "occupancy"},
                Drone: {"n_covered": "n_covered", "exploring": "exploring"},
            }

        self.datacollector = ThreadSafeDataCollector(
            model_reporters=reporter_modello,
            agenttype_reporters=reporter_per_tipo,
        )

        # Prima riga: lo stato a t=0, prima che si muova qualunque cosa. Serve come
        # riferimento per il transitorio: senza, il primo dato che hai e' gia' il
        # risultato di un passo e non sai da dove sei partito.
        self.datacollector.collect(self)

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
        """Riconta, per ogni punto, quanti droni lo stanno presidiando adesso.

        Non lo calcolano i punti da soli: il punto non sa chi ha intorno. Lo fa il
        modello, che vede tutti, e lo SCRIVE dentro ogni punto. I droni poi lo leggono.
        E' l'unico canale attraverso cui due droni lontani si coordinano senza parlarsi:
        se uno arriva su un punto, l'occupancy sale, il deficit scende, e gli altri
        smettono di essere attratti da quel punto.
        """
        # Azzero prima tutti i contatori dei droni: n_covered viene ricostruito da zero
        # a ogni passo, non incrementato all'infinito.
        for drone in self.drone_agents:
            drone.n_covered = 0

        for punto in self.target_agents:
            # Una chiamata per punto: distanze da QUESTO punto a TUTTI i droni.
            # L'array torna nello stesso ordine di self.drone_agents (verificato
            # eseguendolo), quindi distanze[i] e' la distanza del drone i.
            distanze, _ = self.space.calculate_distances(
                punto.position, agents=self.drone_agents
            )

            presidianti = 0
            for i in range(len(self.drone_agents)):
                if distanze[i] <= self.coverage_radius:
                    presidianti += 1
                    # NOTA: un drone entro coverage_radius di DUE punti vicini conta
                    # per entrambi. Percio' la somma delle occupancy puo' superare
                    # n_droni: e' voluto (un drone in mezzo a due zone le presidia
                    # davvero entrambe), ma va ricordato leggendo le metriche.
                    self.drone_agents[i].n_covered += 1

            punto.occupancy = presidianti

    def step(self):
        """Un passo del mondo: il territorio cambia, i droni reagiscono, si ricontano i presidi."""

        # 1. IL TERRITORIO
        # Oggi TargetAgent.step() non fa niente, ma l'ordine conta gia': il territorio
        # cambia PRIMA che i droni decidano, cosi' nello stesso passo i droni reagiscono
        # alla situazione nuova e non a quella del passo precedente.
        self.agents_by_type[TargetAgent].do("step")

        # 2. I DRONI
        # shuffle_do e non do: esegue step() in ordine CASUALE, rimescolato a ogni passo.
        # Serve perche' i droni non si muovono davvero in simultanea: chi va per primo
        # aggiorna la propria direction, e chi va dopo, nella regola di allineamento,
        # legge quella GIA' AGGIORNATA. Con un ordine fisso il drone 0 avrebbe per sempre
        # il privilegio di decidere per primo e lo stormo erediterebbe un bias
        # sistematico dall'ordine di creazione. Rimescolare non elimina l'asincronia
        # (servirebbe uno step a due fasi, e quello tocca Agents.py): la rende casuale
        # invece che sistematica. E' un'ipotesi di modello da dichiarare in tesi.
        self.agents_by_type[Drone].shuffle_do("step")

        # 3. I PRESIDI
        # DOPO il movimento, cosi' l'occupancy fotografa le posizioni di adesso. I droni
        # la leggeranno al passo successivo, quando saranno ancora in queste posizioni:
        # percezione e conteggio restano coerenti. Calcolandola prima, ogni drone
        # deciderebbe su una fotografia vecchia di un passo.
        self.aggiorna_occupancy()

        # 4. LA MISURA
        # Ultima riga dello step, DOPO aggiorna_occupancy: tutte le metriche leggono
        # l'occupancy, quindi devono vederla gia' aggiornata. Chiamando collect prima,
        # ogni riga del dataframe descriverebbe le posizioni di adesso con i presidi
        # del passo precedente: uno sfasamento di un passo, invisibile a occhio nei
        # grafici e sufficiente a falsare la misura dei tempi di reazione.
        self.datacollector.collect(self)