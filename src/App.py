"""Visualizzazione: k droni con capacita' limitata contro n punti da coprire.

Avvio:  uv run solara run App.py

Verde  = punto COPERTO da almeno un drone
Grigio = punto SCOPERTO
Drone grigio = non sta coprendo nulla (risorsa sprecata)
"""

from matplotlib.markers import MarkerStyle
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from Agents import TargetAgent
from Model import BoidFlockers, BoidsScenario

# Il backend matplotlib fa una scatter() separata per ogni marker distinto: con un
# marker per grado il disegno costa ~4x. 15 gradi e' impercettibile a occhio.
PASSO_MARKER = 15
MARKER_CACHE = {deg: MarkerStyle("^") for deg in range(0, 360, PASSO_MARKER)}
for deg, marker in MARKER_CACHE.items():
    marker._transform = marker.get_transform().rotate_deg(deg)


def agent_draw(agent):
    if isinstance(agent, TargetAgent):
        coperto = agent.covered_by > 0
        return AgentPortrayalStyle(
            color="tab:green" if coperto else "0.75",
            size=90 if coperto else 45,
            marker="*",
            zorder=1,
            edgecolors="black" if coperto else "0.5",
            linewidths=0.6,
        )

    # drone grigio = non copre niente: si vede subito la risorsa sprecata
    attivo = agent.n_covered > 0
    return AgentPortrayalStyle(
        color="tab:red" if attivo else "0.55",
        size=30,
        marker=MARKER_CACHE[int(round(agent.angle / PASSO_MARKER) * PASSO_MARKER) % 360],
        zorder=2,
        edgecolors="none",  # se un agente dichiara edgecolors devono farlo TUTTI
        linewidths=0.0,
    )


model_params = {
    "rng": {"type": "InputText", "value": 42, "label": "Seme casuale"},
    # --- risorse: qui si crea la scarsita' ---
    "population_size": Slider("Droni (k)", 12, 2, 40, 1),
    "n_targets": Slider("Punti da coprire (n)", 60, 5, 150, 5),
    "coverage_radius": Slider("Raggio di copertura", 8, 2, 25, 1),
    # --- territorio ---
    "target_layout": {
        "type": "Select",
        "value": "cluster",
        "values": ["cluster", "casuale", "griglia"],
        "label": "Disposizione dei punti",
    },
    "n_clusters": Slider("Numero di cluster", 6, 1, 15, 1),
    "cluster_spread": Slider("Ampiezza dei cluster", 7, 2, 25, 1),
    # --- boid ---
    # Vincolo: il raggio di virata (speed/cohere) deve stare DENTRO il raggio di
    # copertura, altrimenti il drone orbita fuori dal punto e la copertura oscilla.
    "speed": Slider("Velocita'", 1.0, 0.5, 5.0, 0.5),
    "cohere": Slider("Attrazione al punto", 0.25, 0.02, 0.6, 0.01),
    "vision": Slider("Raggio di percezione", 10, 1, 50, 1),
    "separation": Slider("Distanza minima", 2, 1, 20, 1),
    "separate": Slider("Separazione", 0.015, 0.0, 0.2, 0.005),
    "match": Slider("Allineamento", 0.05, 0.0, 0.3, 0.01),
}

model = BoidFlockers(scenario=BoidsScenario())

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_draw).render()

copertura = make_plot_component({"PuntiCoperti": "tab:green", "DroniOziosi": "tab:gray"})
spreco = make_plot_component({"Ridondanza": "tab:red"})

page = SolaraViz(
    model,
    renderer,
    components=[copertura, spreco],
    model_params=model_params,
    name="Copertura di punti con risorse scarse",
    play_interval=10,   # il default e' 100 ms: da solo mette un tetto di 10 fps
    render_interval=1,
)
page  # noqa: B018