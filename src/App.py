import solara

import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D

from mesa.visualization import SolaraViz, SpaceRenderer, Slider, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from Model import CoverageModel


# --- RICONOSCIMENTO DEGLI AGENTI ---
# NIENTE isinstance, e non e' pigrizia. Solara ricarica i moduli mentre l'app gira:
# dopo un reload, Agents.TargetAgent e' un oggetto-classe NUOVO, mentre queste
# funzioni hanno catturato quello VECCHIO. isinstance() confronta l'identita' della
# classe, quindi comincia a restituire False su agenti perfettamente validi - senza
# sollevare niente. Il sintomo e' che pezzi del disegno spariscono dopo qualche
# ricarica e non tornano piu'.
# Il controllo sugli attributi invece regge, perche' guarda cosa l'oggetto SA FARE
# e non da quale oggetto-classe discende. Verificato ricaricando il modulo a mano.
def e_un_punto(agente):
    """Vero per un TargetAgent: solo i punti hanno una quota."""
    return hasattr(agente, "priority")


def e_un_drone(agente):
    """Vero per un Drone: solo i droni sanno di stare esplorando."""
    return hasattr(agente, "exploring")


# --- COLORI ---
# In alto e non sparsi nel codice: sono l'unica cosa che si cambia davvero spesso,
# e averli qui evita di andarli a cercare dentro gli if.
# I quattro stati di un punto sono una SCALA ORDINATA, e i colori la seguono:
# rosso -> arancione -> verde -> ciano = nessuno, pochi, giusti, troppi.
# Il ciano e non il viola per l'eccesso: il viola e' gia' "drone in viaggio", e
# condividere una tinta fra due stati concettualmente scollegati confonde, anche se
# le forme (quadrato/cerchio) sarebbero diverse.
COLORE_SCOPERTO = "tab:red"      # punto con ZERO droni: nessuno lo sta guardando
COLORE_PARZIALE = "tab:orange"   # ha qualcuno ma non abbastanza
COLORE_SERVITO = "tab:green"     # punto esattamente alla sua quota
COLORE_ECCESSO = "tab:cyan"      # punto con piu' droni di quanti ne chieda
COLORE_STAZIONE = "tab:blue"     # drone che sta presidiando
COLORE_VIAGGIO = "tab:purple"    # drone con un target ma non ancora arrivato
COLORE_ESPLORA = "tab:gray"      # drone che non vede nessun punto da servire


def agent_portrayal(agent):
    """Dice a Mesa come disegnare UN agente. Viene chiamata per ognuno, a ogni frame.

    Deve restituire un AgentPortrayalStyle. In Mesa 3.5.1 restituire un dizionario
    funziona ancora ma e' DEPRECATO (emette un warning e sparira' in Mesa 4): quasi
    tutti i tutorial in circolazione usano ancora la forma vecchia.
    """

    if e_un_punto(agent):
        # --- PUNTI DI INTERESSE ---
        # DUE CANALI VISIVI INDIPENDENTI, ed e' una scelta, non un dettaglio:
        #   dimensione = priorita'  -> la DOMANDA, che non cambia
        #   colore     = deficit    -> lo STATO, che cambia a ogni passo
        # Mescolandoli in un canale solo non distingueresti un punto di quota 3
        # soddisfatto da uno di quota 1 soddisfatto: due situazioni molto diverse.
        # QUATTRO stati e non tre: separare "zero droni" da "qualcuno ma non
        # abbastanza" e' la distinzione che conta di piu' guardando la mappa. Un punto
        # di quota 3 con due droni sopra sta per essere risolto; uno con zero droni
        # non lo ha ancora trovato nessuno. Prima erano tutti e due rossi, e nel mondo
        # di default erano 6 punti rossi di cui solo 2 davvero deserti.
        # Guardo occupancy e non il segno del deficit, perche' il deficit da solo non
        # distingue 0 da "qualcuno": priority 3 con 0 droni e priority 1 con 0 droni
        # hanno deficit diverso ma sono lo stesso stato.
        if agent.occupancy == 0:
            colore = COLORE_SCOPERTO
        elif agent.occupancy < agent.priority:
            colore = COLORE_PARZIALE
        elif agent.occupancy == agent.priority:
            colore = COLORE_SERVITO
        else:
            colore = COLORE_ECCESSO

        return AgentPortrayalStyle(
            color=colore,
            marker="s",                       # quadrato: i punti sono luoghi, non veicoli
            size=90 + 70 * agent.priority,    # quota 1 -> 160, quota 3 -> 300
            zorder=1,                         # sotto ai droni: i droni non devono sparirci sopra
            edgecolors="black",
            linewidths=0.5,
        )

    # --- DRONI ---
    # Per il quadricottero "in stazione" e' un ruolo operativo, non la semplice
    # coincidenza geometrica n_covered > 0. Un explorer che attraversa una coverage
    # o un support in uscita non deve apparire falsamente stazionario.
    if hasattr(agent, "station_role"):
        if agent.station_role in ("owner", "support"):
            colore = COLORE_STAZIONE
        elif agent.exploring:
            colore = COLORE_ESPLORA
        else:
            colore = COLORE_VIAGGIO
    else:
        # Per l'ala fissa non esistono ruoli discreti di hovering: resta valida
        # la classificazione geometrica originale.
        if agent.n_covered > 0:
            colore = COLORE_STAZIONE
        elif agent.exploring:
            colore = COLORE_ESPLORA
        else:
            colore = COLORE_VIAGGIO

    # L'owner del quadricottero usa una stella: il COLORE continua a descrivere lo
    # stato operativo, la FORMA aggiunge solo il ruolo locale di stazionamento.
    marker_drone = "*" if getattr(agent, "owner", False) else "o"
    dimensione_drone = 60 if getattr(agent, "owner", False) else 28

    return AgentPortrayalStyle(
        color=colore,
        marker=marker_drone,
        size=dimensione_drone,
        zorder=2,
        # edgecolors e linewidths vanno dati a TUTTI gli agenti o a nessuno: il backend
        # accumula solo i valori non nulli, quindi impostandoli soltanto sui punti
        # l'array risulta di 12 elementi contro 52 agenti e la maschera di zorder va
        # fuori sincrono con un IndexError. Verificato eseguendolo.
        edgecolors="black",
        linewidths=0.3,
    )


# --- LEGENDA ---
# Voci costruite a mano con Line2D "vuote": servono solo come campioni di colore,
# non disegnano dati. E' il modo standard di fare una legenda quando i colori
# vengono da un unico scatter multicolore e non da serie separate.
def _voce_legenda(colore, forma, etichetta):
    return Line2D(
        [], [], color=colore, marker=forma, linestyle="none",
        markersize=8, markeredgecolor="black", markeredgewidth=0.3, label=etichetta,
    )


# ETICHETTE CORTE, e la lunghezza qui e' un vincolo misurato, non uno stile.
# La legenda sta sotto la mappa e la figura viene salvata con bbox_inches="tight":
# la larghezza della legenda diventa la larghezza dell'IMMAGINE. Con gli assi a
# aspect="equal" la mappa quadrata deve poi starci dentro, quindi una legenda larga
# la rimpicciolisce. Misurato: con etichette da ~27 caratteri l'immagine esce
# 511x553 (rapporto 0.92, mappa piena); con una da 40 esce 593x310 (rapporto 1.91,
# mappa ridotta a un francobollo). I nomi degli agenti non li ripeto ("punto
# scoperto" -> "scoperto"): il colore del quadrato o del cerchio lo dice gia'.
VOCI_LEGENDA = [
    _voce_legenda(COLORE_SCOPERTO, "s", "scoperto"),
    _voce_legenda(COLORE_PARZIALE, "s", "parziale"),
    _voce_legenda(COLORE_SERVITO, "s", "servito"),
    _voce_legenda(COLORE_ECCESSO, "s", "sovra-servito"),
    _voce_legenda(COLORE_STAZIONE, "o", "in stazione"),
    _voce_legenda(COLORE_VIAGGIO, "o", "in viaggio"),
    _voce_legenda(COLORE_ESPLORA, "o", "esplora"),
    _voce_legenda(COLORE_STAZIONE, "*", "owner quad"),
    # Il cerchio tratteggiato. La dicitura dice DI CHI E' il raggio, che e' la
    # lettura sbagliata da prevenire: il cerchio sta attorno al punto ma il raggio
    # e' del drone, ed e' il luogo delle posizioni da cui un drone lo presidia.
    Line2D([], [], color="black", linestyle="--", linewidth=0.7, alpha=0.6,
           label="presidio (raggio del drone)"),
]


def compatta(ax, larghezza, altezza):
    """Rimpicciolisce la figura che contiene questi assi.

    Serve perche' SolaraViz dispone i componenti in una griglia con celle di
    altezza fissa (6 colonne x 10 righe), mentre matplotlib crea le figure alla
    dimensione di default: eccedono la cella e si sovrappongono al componente
    accanto. Ne' make_plot_component ne' SpaceRenderer accettano una figsize, ma
    entrambi chiamano un post_process con gli Axes - e dagli Axes si risale alla
    figura. E' l'unico punto di aggancio disponibile.
    """
    ax.get_figure().set_size_inches(larghezza, altezza)


def compatta_grafico(ax):
    """post_process dei tre grafici: solo la dimensione."""
    compatta(ax, 4.6, 2.9)


def configura_assi(ax):
    """post_process del renderer: SOLO configurazione degli assi, nessun disegno.

    Perche' solo configurazione: SolaraViz applica post_process UNA VOLTA SOLA
    (tiene un flag _post_process_applied che non riazzera mai) e intanto ripulisce
    patches/collections/lines/artists a OGNI frame. Un cerchio disegnato qui
    comparirebbe al primo frame e sparirebbe al secondo. Verificato eseguendolo.
    Legenda, titolo e proprieta' degli assi invece sopravvivono, perche' non stanno
    in quelle liste: la legenda vive in ax.legend_.
    """
    # aspect equal: senza, con width != height matplotlib stira gli assi, i cerchi
    # di copertura diventano ellissi e le distanze sullo schermo non corrispondono
    # piu' a quelle del modello. In un modello di copertura e' fuorviante.
    compatta(ax, 5.0, 5.4)
    ax.set_aspect("equal")
    # Legenda SOTTO e non a destra: con bbox_inches="tight" una legenda laterale
    # allarga la figura, che sfonda la sua cella nella griglia di SolaraViz e va a
    # sovrapporsi al componente accanto. Sotto la figura cresce in altezza, dove c'e'
    # spazio, e su tre colonne resta compatta. Fuori dagli assi e non dentro per non
    # coprire gli agenti.
    ax.legend(
        handles=VOCI_LEGENDA, loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=3, frameon=False, fontsize=8,
    )


class CustomSpaceRenderer(SpaceRenderer):
    """Renderer personalizzato che garantisce il disegno dei cerchi e delle frecce

    anche dopo il Reset o il cambio dei parametri da slider.
    """
    def draw_agents(self, *args, **kwargs):
        # 1. Fa il disegno standard dei droni e dei punti
        risultato = super().draw_agents(*args, **kwargs)
        ax = self.canvas

        # 2. Recupera gli agenti dallo spazio corrente
        punti = [a for a in self.space.agents if e_un_punto(a)]
        droni = [a for a in self.space.agents if e_un_drone(a)]

        # 3. Zone di copertura: UNA EllipseCollection invece di N add_patch(Circle).
        # Misurato: 12 Circle con add_patch costano 25.0 ms per frame, una
        # EllipseCollection 0.38 ms - sessantacinque volte meno, e con 12 punti e'
        # gia' il 20% del costo totale di un frame. Il disegno e' identico.
        # units="xy" + offset_transform=transData: larghezza e altezza sono in
        # coordinate del MODELLO, non in punti-schermo. Senza, i cerchi non
        # seguirebbero lo zoom e non varrebbero piu' coverage_radius.
        if punti:
            xy = np.array([p.position for p in punti])
            diametro = np.full(len(punti), 2.0 * punti[0].model.coverage_radius)
            ax.add_collection(EllipseCollection(
                widths=diametro, heights=diametro, angles=np.zeros(len(punti)),
                units="xy", offsets=xy, offset_transform=ax.transData,
                facecolors="none", edgecolors="black", linestyles="--",
                linewidths=0.7, alpha=0.45, zorder=0,
            ))

        # 4. Priorita' numerica accanto a ogni punto.
        # Usiamo scatter con marker matematico (es. "$3$") invece di ax.text:
        # scatter crea una PathCollection, cioe' lo stesso tipo di oggetto grafico
        # che il renderer Mesa/Matplotlib gestisce e ripulisce correttamente a ogni frame.
        if punti:
            offset_x = max(1.2, 0.018 * self.space.width)
            for punto in punti:
                x_punto = float(punto.position[0])
                y_punto = float(punto.position[1])

                # Di norma metto il numero a destra del quadrato.
                # Vicino al bordo destro lo sposto a sinistra per non tagliarlo.
                if x_punto + offset_x < self.space.width:
                    x_label = x_punto + offset_x
                else:
                    x_label = x_punto - offset_x

                ax.scatter(
                    [x_label],
                    [y_punto],
                    marker=f"${int(punto.priority)}$",
                    s=85,
                    c="black",
                    zorder=4,
                )

        # 5. Frecce di direzione solo per i droni che si stanno realmente muovendo.
        # Un quadricottero in hovering conserva l'ultima direction come memoria
        # cinematica, ma disegnarla come freccia suggerirebbe falsamente movimento.
        droni_in_movimento = [d for d in droni if getattr(d, "moving", True)]
        if droni_in_movimento:
            x = np.array([d.position[0] for d in droni_in_movimento])
            y = np.array([d.position[1] for d in droni_in_movimento])
            ang = np.radians(np.array([d.angle for d in droni_in_movimento]))
            ax.quiver(x, y, np.cos(ang), np.sin(ang),
                      scale=45, width=0.0035, alpha=0.55, zorder=3)

        return risultato


# --- PARAMETRI REGOLABILI DALL'INTERFACCIA ---
# I MINIMI NON SONO ARBITRARI. Spostare uno slider RICOSTRUISCE il modello da zero,
# quindi una combinazione che viola un guardrail solleva ValueError e pianta
# l'interfaccia. Con i parametri fissi (speed=1, coverage_radius=8):
#     cohere         >= speed/coverage_radius = 0.125  ->  minimo 0.15   (guardrail 2)
#     point_sensing_radius >= coverage_radius = 8      ->  minimo 8.0
#     ala fissa:     2*margin = 24 < 100
#     quadricottero: 2*quadcopter_margin = 4 < 100
# Per questo speed e coverage_radius NON sono esposti: renderli regolabili
# accoppierebbe i vincoli fra loro e nessuna scelta di estremi sarebbe piu' sicura.
# Quei tre si cambiano da run_batch.py, dove un ValueError e' un'informazione utile
# e non un'applicazione che si chiude in faccia a chi la sta usando.
model_params = {
    "seed": Slider("seme casuale", value=42, min=0, max=200, step=1),
    "n_droni": Slider("droni", value=20, min=5, max=90, step=5),
    "n_punti": Slider("punti di interesse", value=12, min=2, max=30, step=1),
    "priorita_massima": Slider("quota massima per punto", value=3, min=1, max=6, step=1),
    "disposizione_punti": {
        "type": "Select",
        "value": "casuali",
        "values": ["casuali", "gruppi", "sparsi", "cerchio", "bordi", "centrali"],
        "label": "disposizione iniziale dei punti",
    },
    "tipo_drone": {
        "type": "Select",
        "value": "quadricottero",
        "values": ["quadricottero", "ala_fissa"],
        "label": "tipo di drone",
    },
    "partenza": {
        "type": "Select",
        "value": "sparsi",
        "values": ["sparsi", "base", "alto", "basso", "sinistra", "destra"],
        "label": "schieramento iniziale",
    },
    "beta": Slider("beta: costo del viaggio", value=0.05, min=0.0, max=0.30, step=0.01),
    "cohere": Slider("attrazione al punto", value=0.25, min=0.15, max=0.60, step=0.05),
    "point_sensing_radius": Slider(
        "raggio di percezione dei punti",
        value=10.0, min=8.0, max=25.0, step=0.5,
    ),
    "drone_sensing_radius": Slider(
        "raggio di comunicazione dei droni",
        value=10.0, min=8.0, max=25.0, step=0.5,
    ),
    "match": Slider("allineamento fra droni", value=0.05, min=0.0, max=0.20, step=0.01),
    "separation": Slider("distanziamento tra droni vicini", value=0.015, min=0.0, max=0.05, step=0.05),
    "explore": Slider("intensita' dell'esplorazione", value=0.2, min=0.0, max=0.60, step=0.05),
    "avoid_angle_degrees": Slider(
        "deviazione da presidio soddisfatto",
        value=10.0, min=0.0, max=30.0, step=1.0,
    ),
    "support_inset": Slider(
        "rientro support dal bordo",
        value=2.0, min=0.5, max=4.0, step=0.5,
    ),
    "release_delay_max_steps": Slider(
        "variabilita' attesa sovraffollamento",
        value=5, min=0, max=20, step=1,
    ),
}


# --- GRAFICI ---
# I nomi sono ESATTAMENTE le chiavi dei model_reporters del DataCollector: se non
# combaciano il grafico resta vuoto senza dire perche'.
# Primo grafico: il deficit insieme al suo pavimento. Vanno letti sempre insieme -
# un deficit che si assesta su 4 e' un fallimento se il pavimento e' 0 e un successo
# perfetto se il pavimento e' 4.
# Solo il deficit residuo. deficit_incomprimibile NON e' qui apposta: e' una
# costante, quindi una retta orizzontale che non aggiunge informazione nel tempo,
# spesso vale 0 e schiaccia la curva vera in alto, e - cosa peggiore - puo' essere
# legittimamente attraversata verso il basso (vedi il commento nel modello), il che
# la fa sembrare un bug. Resta un attributo del modello, da leggere quando serve.
grafico_deficit = make_plot_component(
    {"deficit_residuo": "tab:red"}, post_process=compatta_grafico
)

# Secondo: i due tipi di drone inattivo. La DISTANZA fra le due curve e' la
# diagnostica: sono i droni che un punto l'hanno scelto ma non lo stanno presidiando.
grafico_droni = make_plot_component(
    {"droni_oziosi": "tab:gray", "droni_in_esplorazione": "tab:purple"}, post_process=compatta_grafico
)

# Terzo: i due modi di sbagliare, punti lasciati indietro e droni sprecati.
grafico_punti = make_plot_component(
    {"punti_soddisfatti": "tab:green", "sovra_servizio": "tab:orange"}, post_process=compatta_grafico
)




# --- COMPONENTE INTERFACCIA: SWITCH PER MOSTRARE/NASCONDERE I GRAFICI ---
# Parte disattivato (False) di default per garantire le massime prestazioni
mostra_grafici = solara.reactive(False)

@solara.component
def PannelloGrafici(model):
    with solara.Column():
        solara.Switch(label="Mostra i grafici (rallenta l'app)", value=mostra_grafici)
        if mostra_grafici.value:
            # make_plot_component restituisce sempre (funzione, numero_di_pagina):
            # il secondo elemento e' un intero, quindi non ci sono kwargs da estrarre.
            for componente, _pagina in (grafico_deficit, grafico_droni, grafico_punti):
                componente(model)


# --- LA PAGINA ---
modello = CoverageModel()

renderer = CustomSpaceRenderer(modello, backend="matplotlib")
renderer.setup_agents(agent_portrayal)   # stile visivo droni/punti
renderer.post_process = configura_assi # configurazione assi e legenda


# QUESTE DUE RIGHE SONO OBBLIGATORIE, e la loro assenza non da' nessun errore.
# SolaraViz ridisegna cosi':
#       if renderer.space_mesh: renderer.draw_structure()
#       if renderer.agent_mesh: renderer.draw_agents()
# Sono CONDIZIONALI, e space_mesh/agent_mesh nascono None: si valorizzano soltanto
# alla prima chiamata esplicita. Senza, SolaraViz non disegna mai nulla e il pannello
# della mappa resta bianco con gli assi di default da 0 a 1 - mentre legenda e
# grafici funzionano regolarmente, il che rende il sintomo fuorviante.
renderer.draw_structure()
renderer.draw_agents()

# Il nome della variabile conta: solara cerca 'page' a livello di modulo.
# Avvio:  uv run solara run app.py
page = SolaraViz(
    modello,
    renderer,
    components=[PannelloGrafici],
    model_params=model_params,
    name="Copertura adattiva di punti di interesse",
)
