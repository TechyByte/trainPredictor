import logging
import pickle
import time

import networkx as nx
import json
import pandas as pd
import geopandas
import config


from utilities import BiDict

G = nx.DiGraph()

tiploc_stanox = BiDict()


def load_toc_file(f):
    """Load and process a TOC JSON file to construct the graph and mapping."""
    for line in f:
        data = json.loads(line)
        try:
            tiploc_entry = data["TiplocV1"]
            tiploc_stanox[tiploc_entry["tiploc_code"]] = tiploc_entry["stanox"]
        except:
            try:
                schedule_entry = data["JsonScheduleV1"]
                previous_stop = None
                for stop in schedule_entry["schedule_segment"]["schedule_location"]:
                    if previous_stop is None:
                        previous_stop = stop["tiploc_code"]
                    else:
                        G.add_edge(previous_stop, stop["tiploc_code"])
                        previous_stop = stop["tiploc_code"]

            except:
                continue
    for k, v in tiploc_stanox.items():
        try:
            G.nodes[k]["stanox"] = v
            G.nodes[k]["incidents"] = []
        except KeyError:
            continue


# Initialize the cache
routes_cache = {}


def get_routes(origin, destination):
    # Create a key for the cache
    cache_key = (origin, destination)

    # Check if the result is in the cache
    if cache_key in routes_cache:
        return routes_cache[cache_key]

    # If the result is not in the cache, compute the routes
    paths = []
    try:
        for path in nx.all_shortest_paths(G, origin, destination):
            paths.append(path)
            logging.debug(path)

        # Store the result in the cache
        routes_cache[cache_key] = paths

        return paths
    except KeyError:
        logging.error("Invalid station code.")


logging.info("Preparing network model...")
tic = time.perf_counter()
load_toc_file(open("input_files/toc_json/toc-full.json"))
toc = time.perf_counter()
logging.info(f"Network graph inferred (took {toc - tic:0.4f} seconds)")

logging.info("Populating geospatial data...")
df = pd.read_csv("input_files/tiploc_spatial_data/tiploc.csv")
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['EASTING'], df['NORTHING'], crs='epsg:27700')).to_crs("4326")
tic = time.perf_counter()

for row in gdf.itertuples():
    try:
        G.nodes[row.TIPLOC]["latlong"] = (row.geometry.y, row.geometry.x)
        G.nodes[row.TIPLOC]["name"] = row.NAME
    except KeyError:
        continue



toc = time.perf_counter()
logging.info(f"Geospatial data processed (took {toc - tic:0.4f} seconds)")

with open("bare_network_model.pkl", "wb") as file:
    pickle.dump(G, file)

if __name__ == "__main__":
    #print((G.nodes["DIGBY"]).adjacents())
    #logging.info(G.nodes["EXETRSD"]) # should equal {'stanox': '83421', 'latlong': (50.72978236583622, -3.543543710734685), 'name': 'EXETER ST DAVIDS', 'weather_city_id': None}
    #logging.info(G.nodes["EXETRSD"]["stanox"]) # should equal 83421
    #logging.info(gdf[gdf["TIPLOC"] == "EXETRSD"]["geometry"].values[0].coords[:][0][::-1]) #EXETRSD lat/long
    print(get_routes("EXETRSD", "EXMOUTH"))
