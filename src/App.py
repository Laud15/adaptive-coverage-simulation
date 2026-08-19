import solara

import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D

from mesa.visualization import SolaraViz, SpaceRenderer, Slider, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

from Model import CoverageModel

# --- AGENT RECOGNITION ---
# NO isinstance. Solara reloads modules while the app is running:
# after a reload, Agents.TargetAgent is a NEW class object, 
# while these functions captured the OLD one. isinstance() compares class identity,
# so it starts returning False for perfectly valid agents - without raising anything. 
# The symptom is that parts of the drawing disappear after a few reloads and never return.
# Attribute checks remain reliable instead, because they examine what the object CAN DO rather than which class object it inherits from.
def is_point(agent):
    """True for a TargetAgent: only points have a quota."""
    return hasattr(agent, "priority")

def is_drone(agent):
    """True for a Drone: only drones know whether they are exploring."""
    return hasattr(agent, "exploring")

# --- COLORS ---
# Kept at the top instead of scattered through the code: these are the only values changed frequently, and placing them here avoids searching inside conditionals.
# A point's four states form an ORDERED SCALE, and the colors follow it:
# red -> orange -> green -> cyan = none, too few, enough, too many.
UNCOVERED_COLOR = "tab:red" # point with ZERO drones: nobody is watching it
PARTIAL_COLOR = "tab:orange" # it has some drones, but not enough
SERVED_COLOR = "tab:green" # point exactly at its quota
OVERSERVED_COLOR = "tab:cyan" # point with more drones than requested
STATION_COLOR = "tab:blue" # drone currently stationing
TRAVEL_COLOR = "tab:purple" # drone with a target but not there yet
EXPLORING_COLOR = "tab:gray" # drone that sees no point to serve

def agent_portrayal(agent):
    """Tells Mesa how to draw an agent. Called for every agent, on every frame.

    It must return an AgentPortrayalStyle. In Mesa 3.5.1, returning a dictionary
    still works but is DEPRECATED (it emits a warning and will disappear in Mesa 4):
    almost all tutorials currently available still use the old form.
    """
    if is_point(agent):
        # --- POINTS OF INTEREST ---
        # TWO INDEPENDENT VISUAL CHANNELS, which is a deliberate choice, not a detail:
        #   size  = priority -> DEMAND, which does not change
        #   color = deficit  -> STATE, which changes at every step
        # Combining them into a single channel would make it impossible to distinguish a satisfied quota-3 point from a satisfied quota-1 point.
        if agent.occupancy == 0:
            color = UNCOVERED_COLOR
        elif agent.occupancy < agent.priority:
            color = PARTIAL_COLOR
        elif agent.occupancy == agent.priority:
            color = SERVED_COLOR
        else:
            color = OVERSERVED_COLOR

        return AgentPortrayalStyle(
            color=color,
            marker="s", # square: points are locations, not vehicles
            size=90 + 70 * agent.priority, # quota 1 -> 160, quota 3 -> 300
            zorder=1, # below drones: drones must not disappear behind them
            edgecolors="black",
            linewidths=0.5,
        )

    # --- DRONES ---
    # For a quadcopter, "in station" is an operational role, not merely the geometric coincidence n_covered > 0. 
    # An explorer crossing a coverage zone or a departing support must not falsely appear stationary.
    if hasattr(agent, "station_role"):
        if agent.station_role in ("owner", "support"):
            color = STATION_COLOR
        elif agent.exploring:
            color = EXPLORING_COLOR
        else:
            color = TRAVEL_COLOR
    else:
        # Fixed-wing drones have no discrete hovering roles: the original geometric classification remains valid.
        if agent.n_covered > 0:
            color = STATION_COLOR
        elif agent.exploring:
            color = EXPLORING_COLOR
        else:
            color = TRAVEL_COLOR

    # A quadcopter owner uses a star: COLOR continues to describe the operational state
    drone_marker = "*" if getattr(agent, "owner", False) else "o"
    drone_size = 60 if getattr(agent, "owner", False) else 28

    return AgentPortrayalStyle(
        color=color,
        marker=drone_marker,
        size=drone_size,
        zorder=2,
        edgecolors="black",
        linewidths=0.3,
    )

# --- LEGEND ---
# Entries built manually with "empty" Line2D objects: they serve only as color samples and do not draw data. 
def _legend_entry(color, shape, label):
    return Line2D(
        [], [], color=color, marker=shape, linestyle="none",
        markersize=8, markeredgecolor="black", markeredgewidth=0.3, label=label,
    )

LEGEND_ENTRIES = [
    _legend_entry(UNCOVERED_COLOR, "s", "uncovered"),
    _legend_entry(PARTIAL_COLOR, "s", "partially served"),
    _legend_entry(SERVED_COLOR, "s", "served"),
    _legend_entry(OVERSERVED_COLOR, "s", "over-served"),
    _legend_entry(STATION_COLOR, "o", "stationary"),
    _legend_entry(TRAVEL_COLOR, "o", "traveling"),
    _legend_entry(EXPLORING_COLOR, "o", "exploring"),
    _legend_entry(STATION_COLOR, "*", "owner quad"),
    # The dashed circle. 
    # The label states WHOSE radius it is, preventing the likely misreading: the circle surrounds the point, but the radius
    # belongs to the drone and defines the set of positions from which a drone stations at that point.
    Line2D([], [], color="black", linestyle="--", linewidth=0.7, alpha=0.6, label="stationing zone (drone radius)")]

def resize_figure(ax, width, height):
    """Shrinks the figure containing these axes.

    SolaraViz arranges components in a grid with fixed-height cells (6 columns x
    10 rows), while matplotlib creates figures at the default size: they exceed
    the cell and overlap the adjacent component. Neither make_plot_component nor
    SpaceRenderer accepts a figsize, but both call a post_process with the Axes -
    and the figure can be retrieved from the Axes. This is the only available
    hook.
    """
    ax.get_figure().set_size_inches(width, height)

def resize_plot(ax):
    """post_process for the three plots: size only."""
    resize_figure(ax, 4.6, 2.9)

def configure_axes(ax):
    """ Renderer post_process: axes configuration only, no drawing.

    Why configuration only: SolaraViz applies post_process just once (it keeps a
    _post_process_applied flag that it never resets), while clearing patches/
    collections/lines/artists on EVERY frame. A circle drawn here would appear on
    the first frame and disappear on the second. The legend, title, and axes
    properties instead survive because they are not in those lists: the legend
    lives in ax.legend_.
    """
    # equal aspect: without it, when width != height matplotlib stretches the axes,
    # coverage circles become ellipses and on-screen distances no longer match model distances.
    resize_figure(ax, 5.0, 5.4)
    ax.set_aspect("equal")
    # Legend BELOW rather than on the right: with bbox_inches="tight", a side legend widens the figure,
    # causing it to overflow its SolaraViz grid cell and overlap the adjacent component.
    # Below the figure it grows vertically, where there is space, and stays compact over three columns. 
    # Outside the axes rather than inside, so it does not cover the agents.
    ax.legend(handles=LEGEND_ENTRIES, loc="upper center", bbox_to_anchor=(0.5, -0.05),ncol=3, frameon=False, fontsize=8)

class CustomSpaceRenderer(SpaceRenderer):
    """Custom renderer that ensures circles and arrows are drawn even after a Reset
    or a parameter change through a slider.
    """
    def draw_agents(self, *args, **kwargs):
        # 1) Draws drones and points using the standard renderer
        axes = super().draw_agents(*args, **kwargs)
       

        # 2) Retrieves agents from the current space
        points = [a for a in self.space.agents if is_point(a)]
        drones = [a for a in self.space.agents if is_drone(a)]

        # 3) Coverage zones: ONE EllipseCollection instead of N add_patch(Circle) calls.
        # Improves rendering performance while producing an identical drawing.
        if points:
            xy = np.array([p.position for p in points])
            diameter = np.full(len(points), 2.0 * points[0].model.coverage_radius)
            axes.add_collection(EllipseCollection(
                widths=diameter, heights=diameter, angles=np.zeros(len(points)),
                units="xy", offsets=xy, offset_transform=axes.transData,
                facecolors="none", edgecolors="black", linestyles="--",
                linewidths=0.7, alpha=0.45, zorder=0,
            ))

        # 4) Numerical priority next to each point.
        # Use scatter with a mathematical marker (e.g., "$3$") instead of ax.text:
        # scatter creates a PathCollection,
        # the same type of graphical object that the Mesa/Matplotlib renderer manages and clears correctly on every frame.
        if points:
            offset_x = max(1.2, 0.018 * self.space.width)
            for point in points:
                point_x = float(point.position[0])
                point_y = float(point.position[1])

                # Normally place the number to the right of the square.
                # Near the right boundary, move it to the left to avoid clipping it.
                if point_x + offset_x < self.space.width:
                    x_label = point_x + offset_x
                else:
                    x_label = point_x - offset_x

                axes.scatter(
                    [x_label],
                    [point_y],
                    marker=f"${int(point.priority)}$",
                    s=85,
                    c="black",
                    zorder=4,
                )

        # 5) Direction arrows only for drones that are actually moving.
        # A hovering quadcopter retains its last direction as kinematic memory,
        # but drawing it as an arrow would falsely suggest movement.
        moving_drones = [d for d in drones if getattr(d, "moving", True)]
        if moving_drones:
            x = np.array([d.position[0] for d in moving_drones])
            y = np.array([d.position[1] for d in moving_drones])
            ang = np.radians(np.array([d.angle for d in moving_drones]))
            axes.quiver(x, y, np.cos(ang), np.sin(ang),
                      scale=45, width=0.0035, alpha=0.55, zorder=3)

        return axes

# --- PARAMETERS ADJUSTABLE THROUGH THE INTERFACE ---
# THE MINIMUM VALUES ARE NOT ARBITRARY. Moving a slider REBUILDS the model from scratch,
# so a combination that violates a guardrail raises ValueError and crashes the interface.
# With the fixed parameters (speed, coverage_radius):
#     cohere >= speed/coverage_radiu
#     point_sensing_radius >= coverage_radius 
#     fixed wing: 2*margin 
#     quadcopter: 2*quadcopter_margin 
# For this reason, speed and coverage_radius are NOT exposed: making them adjustable would couple the constraints and no choice of limits would remain safe.
model_params = {
    "seed": Slider("random seed", value=42, min=0, max=200, step=1),
    "n_drones": Slider("drones", value=20, min=5, max=90, step=5),
    "n_points": Slider("points of interest", value=12, min=2, max=30, step=1),
    "max_priority": Slider("maximum quota per point", value=3, min=1, max=6, step=1),
    "point_layout": {
        "type": "Select",
        "value": "random",
        "values": ["random", "clusters", "dispersed", "circle", "edges", "central"],
        "label": "initial point layout",
    },
    "drone_type": {
        "type": "Select",
        "value": "quadcopter",
        "values": ["quadcopter", "fixed_wing"],
        "label": "drone type",
    },
    "deployment": {
        "type": "Select",
        "value": "dispersed",
        "values": ["dispersed", "base", "top", "bottom", "left", "right"],
        "label": "initial deployment",
    },
    "beta": Slider("beta: travel cost", value=0.05, min=0.0, max=0.30, step=0.01),
    "cohere": Slider("attraction to point", value=0.25, min=0.15, max=0.60, step=0.05),
    "point_sensing_radius": Slider(
        "point sensing radius",
        value=10.0, min=8.0, max=25.0, step=0.5,
    ),
    "drone_sensing_radius": Slider(
        "drone communication radius",
        value=10.0, min=8.0, max=25.0, step=0.5,
    ),
    "match": Slider("alignment between drones", value=0.05, min=0.0, max=0.20, step=0.01),
    "separation": Slider("separation between nearby drones", value=2.0, min=0.5, max=5.0, step=0.5),
    "separate": Slider("separation force strength",value=0.015, min=0.0, max=0.05, step=0.005),
    "explore": Slider("exploration intensity", value=0.2, min=0.0, max=0.60, step=0.05),
    "avoid_angle_degrees": Slider(
        "deviation from satisfied station",
        value=10.0, min=0.0, max=30.0, step=1.0,
    ),
    "support_inset": Slider(
        "support inset from boundary",
        value=2.0, min=0.5, max=4.0, step=0.5,
    ),
    "release_delay_max_steps": Slider(
        "overcrowding wait variability",
        value=5, min=0, max=20, step=1,
    ),
}

# --- PLOTS ---
# Names are EXACTLY the DataCollector model_reporters keys: if they do not
# match, the plot remains empty without explaining why.
# First plot: in the best case, the deficit will tend toward a lower asymptote corresponding to the unavoidable deficit. 
deficit_plot = make_plot_component({"residual_deficit": "tab:red"}, post_process=resize_plot)

# Second: the two types of inactive drone. 
# The GAP between the two curves is diagnostic: it represents drones that selected a point but are not stationing there.
drone_plot = make_plot_component({"idle_drones": "tab:gray", "exploring_drones": "tab:purple"}, post_process=resize_plot)

# Third: the two failure modes, points left behind and wasted drones.
point_plot = make_plot_component({"satisfied_points": "tab:green", "overservice": "tab:orange"}, post_process=resize_plot)

# --- INTERFACE COMPONENT: SWITCH TO SHOW/HIDE PLOTS ---
# Starts disabled (False) by default to ensure maximum performance
show_plots = solara.reactive(False)

@solara.component
def PlotPanel(model):
    with solara.Column():
        solara.Switch(label="Show plots (slows down the app)", value=show_plots)
        if show_plots.value:
            # make_plot_component always returns (function, page_number):
            # the second element is an integer, so there are no kwargs to extract.
            for component, _page in (deficit_plot, drone_plot, point_plot):
                component(model)

# --- PAGE ---
model = CoverageModel()

renderer = CustomSpaceRenderer(model, backend="matplotlib")
renderer.setup_agents(agent_portrayal) # drone/point visual style
renderer.post_process = configure_axes # axes and legend configuration

# THESE TWO LINES ARE MANDATORY, and their absence produces no error.
# SolaraViz redraws as follows:
#       if renderer.space_mesh: renderer.draw_structure()
#       if renderer.agent_mesh: renderer.draw_agents()
# They are CONDITIONAL, and space_mesh/agent_mesh start as None: they are initialized only upon the first explicit call.
# Without them, SolaraViz never draws anything and the map panel remains blank with default axes from 0 to 1.
renderer.draw_structure()
renderer.draw_agents()

# The variable name matters: solara looks for 'page' at module level.
# Run: uv run solara run app.py
page = SolaraViz(
    model, # the current simulation
    renderer, # the component that draws the map
    components=[PlotPanel], # additional components, in this case the plots
    model_params=model_params, # interface controls
    name="Adaptive coverage of points of interest", # title
)
