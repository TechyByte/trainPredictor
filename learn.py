import pickle

import networkx

with open("raw_model.pkl", "rb") as file:
    raw_model: networkx.Graph = pickle.load(file)

