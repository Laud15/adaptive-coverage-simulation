import numpy as np

# Geometric configurations available both for the initial condition and
# for point reconfiguration events.
POINT_LAYOUTS = (
    "random",
    "clusters",
    "dispersed",
    "circle",
    "edges",
    "central",
)

def generate_point_positions(
    *,
    layout,
    n_points,
    study_x_min,
    study_x_max,
    study_y_min,
    study_y_max,
    margin,
    rng
):
    """Generate fixed point positions inside the allowed study area.

    The function does not create TargetAgent instances and does not modify the
    model. It only returns an array of positions.

    Randomness is supplied by the caller so initial conditions and event
    scenarios can use independent random-number generators.
    """
    n = int(n_points)
    positions = np.zeros((n, 2), dtype=float)

    if n == 0:
        return positions

    # Rectangle available to point centers after applying point_margin.
    x_min = float(study_x_min) + float(margin)
    x_max = float(study_x_max) - float(margin)
    y_min = float(study_y_min) + float(margin)
    y_max = float(study_y_max) - float(margin)

    width = x_max - x_min
    height = y_max - y_min

    if layout == "random":
        # Independent uniform positions across the available territory.
        for i in range(n):
            positions[i, 0] = rng.uniform(x_min, x_max)
            positions[i, 1] = rng.uniform(y_min, y_max)

    elif layout == "clusters":
        # Use at most three clusters.
        n_clusters = min(3, n)
        centers = np.zeros((n_clusters, 2), dtype=float)

        # Keep cluster centers away from the boundaries to leave enough margin for the point cloud to spread naturally without getting clipped.
        padding_x = 0.15 * width
        padding_y = 0.15 * height

        for cluster_idx in range(n_clusters):
            centers[cluster_idx, 0] = rng.uniform(x_min + padding_x, x_max - padding_x)
            centers[cluster_idx, 1] = rng.uniform(y_min + padding_y, y_max - padding_y)

        assignments = np.arange(n) % n_clusters
        rng.shuffle(assignments)

        sigma = 0.05 * min(width, height)

        for i in range(n):
            center = centers[assignments[i]]
            positions[i] = center + rng.normal(0.0, sigma, size=2)

    elif layout == "dispersed":
        # Divide the territory into cells and place at most one point per cell.
        aspect_ratio = width/height
        n_columns = max(1, int(np.ceil(np.sqrt(n *  aspect_ratio))))
        n_rows = max(1, int(np.ceil(n / n_columns)))

        x_step = width / n_columns
        y_step = height / n_rows

        cells = []

        for row in range(n_rows):
            for column in range(n_columns):
                cells.append([x_min + (column + 0.5) * x_step, y_min + (row + 0.5) * y_step])

        cells = np.asarray(cells, dtype=float)
        rng.shuffle(cells)

        jitter_x = 0.15 * x_step
        jitter_y = 0.15 * y_step

        for i in range(n):
            positions[i, 0] = (cells[i, 0] + rng.uniform(-jitter_x, jitter_x))
            positions[i, 1] = (cells[i, 1] + rng.uniform(-jitter_y, jitter_y))

    elif layout == "circle":
        # Equally spaced points on a circle centered in the territory.
        center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        radius = 0.35 * min(width, height)
        phase = rng.uniform(0.0, 2.0 * np.pi)

        for i in range(n):
            angle = phase + (2.0 * np.pi * i / n)
            positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])

    elif layout == "edges":
        # Assign points approximately equally among the four boundaries.
        band = 0.08 * min(width, height)
        sides = np.arange(n) % 4
        rng.shuffle(sides)

        for i, side in enumerate(sides):
            offset = rng.uniform(0.0, band)
            if side == 0:
                # Left boundary.
                positions[i] = [x_min + offset, rng.uniform(y_min, y_max)]

            elif side == 1:
                # Right boundary.
                positions[i] = [x_max - offset, rng.uniform(y_min, y_max)]

            elif side == 2:
                # Bottom boundary.
                positions[i] = [rng.uniform(x_min, x_max), y_min + offset]

            else:
                # Top boundary.
                positions[i] = [rng.uniform(x_min, x_max), y_max - offset]

    elif layout == "central":
        # Place points inside the central 30% of the available territory.
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        half_width = 0.15 * width
        half_height = 0.15 * height

        for i in range(n):
            positions[i, 0] = rng.uniform(center_x - half_width, center_x + half_width)
            positions[i, 1] = rng.uniform(center_y - half_height, center_y + half_height)

    else:
        raise ValueError(f"Unrecognized point layout: {layout!r}. " f"Use one of {POINT_LAYOUTS}.")

    # Common safety condition: noise and jitter cannot move point centers
    # outside the area allowed by point_margin.
    positions[:, 0] = np.clip(positions[:, 0], x_min, x_max)
    positions[:, 1] = np.clip(positions[:, 1], y_min, y_max)

    return positions