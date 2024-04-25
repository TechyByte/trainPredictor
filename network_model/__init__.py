import time

import networkx as nx
import json

import pandas as pd
import geopandas


from utilities import BiDict

G = nx.DiGraph()

tiploc_stanox = BiDict()


def load_toc_file(f):
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
        except KeyError:
            continue


def get_routes(origin, destination):
    paths = []
    try:
        for path in nx.all_shortest_paths(G, origin, destination):
            paths.append(path)
            print(path)
        return paths
    except KeyError:
        print("Error")


print("Preparing network model...")
tic = time.perf_counter()
load_toc_file(open("input_files/toc_json/toc-full.json"))
toc = time.perf_counter()
print(f"Network graph inferred ({toc - tic:0.4f} seconds so far)")

print("Populating geospatial data...")
df = pd.read_csv("input_files/tiploc_spatial_data/tiploc.csv")
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['EASTING'], df['NORTHING'], crs='epsg:27700')).to_crs("4326")

for row in gdf.itertuples():
    try:
        G.nodes[row.TIPLOC]["latlong"] = (row.geometry.y, row.geometry.x)
        G.nodes[row.TIPLOC]["name"] = row.NAME
    except KeyError:
        continue

toc = time.perf_counter()
print(f"Geospatial data processed (took {toc - tic:0.4f} seconds)")

if __name__ == "__main__":
    get_routes("EXETRSD", "TIVIPW")
    get_routes("EXETRSD", "BHAMNWS")
    get_routes("EXETRSD", "EXMOUTH")
    #print((G.nodes["DIGBY"]).adjacents())
    print(G.nodes["EXETRSD"])
    print(G.nodes["EXETRSD"]["stanox"])
    print(gdf[gdf["TIPLOC"] == "EXETRSD"]["geometry"].values[0].coords[:][0][::-1]) #EXETRSD lat/long
