import logging

import networkx
from matplotlib import pyplot as plt

import config

import network_model as nm

default_position = (49, 1)  # Define a default position
fail_count = 0  # Count the number of nodes without latlong
defaulted = 0  # Count the number of nodes with default position

for node in nm.G.nodes():
    try:
        nm.G.nodes[node]["latlong"]
        pass
    except KeyError:
        fail_count += 1
        # Get the neighbors of the node
        neighbours = networkx.all_neighbors(nm.G, node)

        # Get the latlong of the neighbors
        neighbour_positions = [nm.G.nodes[neighbour]["latlong"] for neighbour in neighbours if
                               "latlong" in nm.G.nodes[neighbour]]
        # If there are neighbors with latlong, compute the average
        if neighbour_positions:
            avg_lat = sum(float(pos[0]) for pos in neighbour_positions) / len(neighbour_positions)
            avg_long = sum(float(pos[1]) for pos in neighbour_positions) / len(neighbour_positions)
            nm.G.nodes[node]["latlong"] = (avg_lat, avg_long)
        else:
            # If there are no neighbors with latlong, assign a default position
            nm.G.nodes[node]["latlong"] = default_position
            default_position = (default_position[0] + 0.1, default_position[1] + 0.1)
            defaulted += 1

logging.info("Complete: " + str(fail_count) + " nodes without latlong, " + str(defaulted) + " nodes defaulted")


def convert(latlong):
    # reflect latlong for plottable coordinates
    return latlong[1], latlong[0]


positions = networkx.get_node_attributes(nm.G, "latlong")
converted_positions = {node: convert(latlong) for node, latlong in positions.items()}

ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')

edgelist = [e for e in nm.G.edges if e not in networkx.selfloop_edges(nm.G)]

networkx.draw_networkx_nodes(nm.G, pos=converted_positions, node_size=1)

# networkx.draw_networkx_edges(nm.G, pos=converted_positions, edgelist=edgelist)

# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()

plt.savefig("map.png")
logging.info("saved map.png")
from keras.models import load_model

# trained_model = load_model('trained_model.h5')
