"""Connect to the viewer and log some data."""

import numpy as np
import rerun as rr

# Initialize the SDK and give our recording a unique name
rr.init("rerun_example_quick_start_connect")

# Connect to a local viewer using the default port
rr.connect_grpc()


# Create some data
SIZE = 10

pos_grid = np.meshgrid(*[np.linspace(-10, 10, SIZE)] * 3)
positions = np.vstack([d.reshape(-1) for d in pos_grid]).T

col_grid = np.meshgrid(*[np.linspace(0, 255, SIZE)] * 3)
colors = np.vstack([c.reshape(-1) for c in col_grid]).astype(np.uint8).T

# Log the data
rr.log(
    # name under which this entity is logged (known as "entity path")
    "my_points",
    # log data as a 3D point cloud archetype
    rr.Points3D(positions, colors=colors, radii=0.5),
)
