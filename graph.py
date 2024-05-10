import logging

import networkx
from matplotlib import pyplot as plt

import config

import network_model as nm

default_position = (53, 2)  # Define a default position

for node in nm.G.nodes():
    try:
        nm.G.nodes[node]["latlong"]
        pass
    except KeyError:
        # Get the neighbors of the node
        neighbours = networkx.all_neighbors(nm.G, node)

        # Get the latlong of the neighbors
        neighbour_positions = [nm.G.nodes[neighbour]["latlong"] for neighbour in neighbours if "latlong" in nm.G.nodes[neighbour]]
        # If there are neighbors with latlong, compute the average
        if neighbour_positions:
            avg_lat = sum(float(pos[0]) for pos in neighbour_positions) / len(neighbour_positions)
            avg_long = sum(float(pos[1]) for pos in neighbour_positions) / len(neighbour_positions)
            nm.G.nodes[node]["latlong"] = (avg_lat, avg_long)
        else:
            # If there are no neighbors with latlong, assign a default position
            nm.G.nodes[node]["latlong"] = default_position

positions = networkx.get_node_attributes(nm.G, "latlong")

print(nm.G.nodes(data=True)["DRBYSMS"]["latlong"])

networkx.draw(nm.G, pos=positions, with_labels=False, node_size=2)
plt.savefig("filename.png")

networkx.draw(nm.G, pos=networkx.get_node_attributes(nm.G, "latlong"), with_labels=True, node_size=10)

from keras.models import load_model

trained_model = load_model('trained_model.h5')
