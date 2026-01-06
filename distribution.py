import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math
import streamlit as st
import io

# This is the Version 6 of Distribution, with respect to the original Edo's Distribution the poles in the chart are spaciated maximum of sample_distance.
# This change gives more realism to the gride and avoid huge voltage drops in the Power Flow analyisis 

# Streamlit app title
st.title("Grid Routing")

#============================= INPUTS ======================================================================================
project_name =      st.text_input("Project Name"," ")
sampling_distance = st.slider("Inter-pole distance (meters)", min_value=10, max_value=100, value=40)
user_distance =     st.slider("Maximum user-pole distance (meters)", min_value=10, max_value=100, value=35)
max_associations =  st.slider("Maximum number of users per pole", min_value=10, max_value=50, value=16)

# Post-processing option
densify_long_edges = st.checkbox(
    "After pruning, add intermediate poles so no branch exceeds the inter-pole distance",
    value=True,
)

# File uploader for roads and users data
roads_file = st.file_uploader("Upload Roads File (GeoPackage format)", type=['gpkg'])
users_file = st.file_uploader("Upload Users File (GeoPackage or Excel)", type=['gpkg', 'xlsx'])

#==========================================================================================================================
# Function Definitions
def load_and_transform_data(file, target_crs=32633):
    if file is None:
        return None
    if file.name.endswith('.gpkg'):
        gdf = gpd.read_file(file)
        if gdf.crs.to_epsg() != target_crs:
            gdf = gdf.to_crs(epsg=target_crs)
        return gdf[gdf.is_valid]
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.to_crs(epsg=target_crs)
            return gdf[gdf.is_valid]
    return None

def sample_points_along_line(line, sampling_distance):
    points = []
    current_distance = 0.0
    while current_distance < line.length:
        point = line.interpolate(current_distance)
        points.append(point)
        current_distance += sampling_distance
    return points

def collect_sampled_points(gdf_roads, sampling_distance):
    sampled_points = []
    for _, row in gdf_roads.iterrows():
        geometry = row.geometry
        if not geometry.is_empty:
            sampled_points.append(Point(geometry.coords[0]))
            sampled_points.append(Point(geometry.coords[-1]))
            if geometry.length > sampling_distance:
                sampled_points.extend(sample_points_along_line(geometry, sampling_distance))
    return sampled_points

def associate_buildings_to_poles(gdf_buildings, gdf_poles, user_distance, max_associations):
    building_association = {index: [] for index in gdf_poles.index}
    associated_buildings = set()  # Global set for already associated buildings

    # Iteration along the poles
    for pole in gdf_poles.itertuples():
        buffer = pole.geometry.buffer(user_distance)

        nearby_buildings = gdf_buildings[
            ~gdf_buildings.index.isin(associated_buildings) &
            gdf_buildings.geometry.within(buffer)
        ].copy()

        nearby_buildings['distance_to_pole'] = nearby_buildings.geometry.distance(pole.geometry)
        nearby_buildings = nearby_buildings.sort_values(by='distance_to_pole')

        num_associated = 0
        for building_id in nearby_buildings.index:
            if num_associated >= max_associations:
                break
            if building_id not in associated_buildings:
                building_association[pole.Index].append(building_id)
                associated_buildings.add(building_id)
                num_associated += 1

        if len(building_association[pole.Index]) > max_associations:
            print(f"Warning: Pole {pole.Index} exceeded max associations.")
            building_association[pole.Index] = building_association[pole.Index][:max_associations]

    return pd.DataFrame(
        [(pole_id, building_id) for pole_id, buildings in building_association.items() for building_id in buildings],
        columns=['pole_id', 'building_id']
    )

def place_poles_for_unassociated_buildings(gdf_unassociated_buildings, user_distance, max_associations):
    new_poles = []
    all_associations = []

    while not gdf_unassociated_buildings.empty:
        largest_cluster = None
        max_cluster_size = 0

        # Find the biggest cluster of buildings
        for building in gdf_unassociated_buildings.itertuples():
            buffer = building.geometry.buffer(user_distance)
            intersecting_buildings = gdf_unassociated_buildings[gdf_unassociated_buildings.geometry.intersects(buffer)]
            if len(intersecting_buildings) > max_cluster_size:
                max_cluster_size = len(intersecting_buildings)
                largest_cluster = intersecting_buildings

        if largest_cluster is not None:
            buffers = [b.geometry.buffer(user_distance) for b in largest_cluster.itertuples()]
            merged_buffer = buffers[0]
            for buf in buffers[1:]:
                merged_buffer = merged_buffer.union(buf)

            # Pole placement (centroid)
            pole_location = merged_buffer.centroid
            new_poles.append(pole_location)

            # Sort buildings by distance to the new pole
            buildings_in_cluster = [(b.Index, b.geometry) for b in largest_cluster.itertuples()]
            buildings_in_cluster.sort(key=lambda x: pole_location.distance(x[1]))

            # Take only the closest up to max_associations
            closest_buildings = buildings_in_cluster[:max_associations]

            # Store associations as (pole_geometry, building_id)
            for building_id, _building_geom in closest_buildings:
                all_associations.append((pole_location, building_id))

            # Remove associated buildings
            associated_indices = [b[0] for b in closest_buildings]
            gdf_unassociated_buildings = gdf_unassociated_buildings[
                ~gdf_unassociated_buildings.index.isin(associated_indices)
            ]
        else:
            break

    return gpd.GeoDataFrame({'geometry': new_poles}, crs=gdf_unassociated_buildings.crs), all_associations

def create_graph_and_mst(gdf_associated_poles):
    G = nx.Graph()
    pole_coords = [(pole.geometry.x, pole.geometry.y) for pole in gdf_associated_poles.itertuples()]
    for i, (x1, y1) in enumerate(pole_coords):
        for j, (x2, y2) in enumerate(pole_coords):
            if i != j:
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                G.add_edge(i, j, weight=distance)
    return nx.minimum_spanning_tree(G)


def densify_mst_edges(gdf_poles: gpd.GeoDataFrame, mst: nx.Graph, sampling_distance: float) -> tuple[gpd.GeoDataFrame, nx.Graph]:
    """
    Post-process the MST by inserting intermediate poles on any edge longer than sampling_distance.

    - Works in *planar metric* CRS (meters).
    - Does NOT follow roads: it splits the straight segment between MST endpoints.
    - Guarantees all resulting edges have length <= sampling_distance (up to floating eps).

    Node convention in this app:
    - MST nodes are 0..N-1 and refer to row positions (iloc) in gdf_poles.
    """

    if sampling_distance <= 0:
        return gdf_poles, mst

    poles = gdf_poles.copy().reset_index(drop=True)
    n0 = len(poles)

    # Build a new graph from scratch to avoid in-place mutation issues
    G2 = nx.Graph()
    G2.add_nodes_from(range(n0))

    new_geoms = []  # geometries of inserted poles, appended at the end

    def geom_at(node_id: int):
        if node_id < n0:
            return poles.geometry.iloc[node_id]
        else:
            return new_geoms[node_id - n0]

    next_id = n0

    for u, v in list(mst.edges()):
        pu = geom_at(u)
        pv = geom_at(v)
        L = pu.distance(pv)

        if L <= sampling_distance + 1e-9:
            G2.add_edge(u, v, weight=float(L))
            continue

        # Split into n_segments so each segment <= sampling_distance
        n_segments = int(math.ceil(L / sampling_distance))
        chain = [u]

        for k in range(1, n_segments):
            frac = k / n_segments
            x = pu.x + frac * (pv.x - pu.x)
            y = pu.y + frac * (pv.y - pu.y)
            new_geoms.append(Point(float(x), float(y)))
            chain.append(next_id)
            next_id += 1

        chain.append(v)

        for a, b in zip(chain[:-1], chain[1:]):
            pa = geom_at(a)
            pb = geom_at(b)
            d = pa.distance(pb)
            G2.add_edge(a, b, weight=float(d))

    # Append new poles (no buildings associated to these)
    if new_geoms:
        gdf_new = gpd.GeoDataFrame({'geometry': new_geoms}, crs=poles.crs)
        poles = pd.concat([poles, gdf_new], ignore_index=True)
        poles = gpd.GeoDataFrame(poles, geometry='geometry', crs=gdf_poles.crs)

    return poles, G2

def plot_map_with_mst_and_connections(gdf_roads, gdf_final_poles, gdf_buildings, mst):
    minx, miny, maxx, maxy = gdf_final_poles.total_bounds
    distance_x = maxx - minx
    distance_y = maxy - miny

    buffer_x = distance_x * 0.20
    buffer_y = distance_y * 0.20

    x_min, x_max = minx - buffer_x, maxx + buffer_x
    y_min, y_max = miny - buffer_y, maxy + buffer_y

    fig, ax = plt.subplots(figsize=(12, 6))

    gdf_roads.plot(color='gray', linewidth=0.5, ax=ax, label='Roads')
    gdf_buildings.plot(color='red', markersize=15, ax=ax, label='Users')

    for edge in mst.edges():
        line = LineString([gdf_final_poles.geometry.iloc[edge[0]], gdf_final_poles.geometry.iloc[edge[1]]])
        ax.plot(*line.xy, color='k', linewidth=1)
    ax.plot([], [], color='k', linewidth=1, label='Network')

    gdf_final_poles.plot(color='k', markersize=5, ax=ax, label='Poles')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.title(f'Distribution Network - {project_name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.legend(loc='upper left')

    st.pyplot(fig)

def save_mst_to_geojson(gdf_final_poles, mst):
    # MST nodes
    nodes = []
    for idx, pole in enumerate(gdf_final_poles.itertuples()):
        nodes.append({'id': idx, 'geometry': pole.geometry})
    gdf_nodes = gpd.GeoDataFrame(nodes, crs=gdf_final_poles.crs)

    # MST edges
    edges = []
    for u, v, data in mst.edges(data=True):
        start = gdf_final_poles.geometry.iloc[u]
        end = gdf_final_poles.geometry.iloc[v]
        edges.append({
            'source': u,
            'target': v,
            'weight': data['weight'],
            'geometry': LineString([start, end])
        })
    gdf_edges = gpd.GeoDataFrame(edges, crs=gdf_final_poles.crs)

    nodes_geojson = io.BytesIO()
    gdf_nodes.to_file(nodes_geojson, driver="GeoJSON")
    nodes_geojson.seek(0)

    edges_geojson = io.BytesIO()
    gdf_edges.to_file(edges_geojson, driver="GeoJSON")
    edges_geojson.seek(0)

    return nodes_geojson, edges_geojson

# Run main process
if roads_file and users_file:
    gdf_roads = load_and_transform_data(roads_file)
    gdf_buildings = load_and_transform_data(users_file)

    if gdf_roads is not None and gdf_buildings is not None:
        sampled_points = collect_sampled_points(gdf_roads, sampling_distance)
        gdf_associated_poles = gpd.GeoDataFrame({'geometry': sampled_points}, crs=gdf_buildings.crs)

        associations_df = associate_buildings_to_poles(
            gdf_buildings, gdf_associated_poles, user_distance, max_associations
        )

        # Keep only poles with at least one building
        gdf_associated_poles = gdf_associated_poles[gdf_associated_poles.index.isin(associations_df['pole_id'])].copy()

        # IMPORTANT: after pruning, reset pole ids to 0..N-1 (stable) and remap associations accordingly
        gdf_associated_poles["old_pole_id"] = gdf_associated_poles.index
        gdf_associated_poles = gdf_associated_poles.reset_index(drop=True)
        old_to_new = dict(zip(gdf_associated_poles["old_pole_id"].tolist(), gdf_associated_poles.index.tolist()))

        associations_df = associations_df.copy()
        associations_df["pole_id"] = associations_df["pole_id"].map(old_to_new)
        if associations_df["pole_id"].isna().any():
            n_bad = int(associations_df["pole_id"].isna().sum())
            st.warning(f"{n_bad} association rows lost during pole-id remap after pruning and were dropped.")
            associations_df = associations_df.dropna(subset=["pole_id"]).copy()
        associations_df["pole_id"] = associations_df["pole_id"].astype(int)

        gdf_unassociated_buildings = gdf_buildings[~gdf_buildings.index.isin(associations_df['building_id'])]

        gdf_new_poles, new_associations = place_poles_for_unassociated_buildings(
            gdf_unassociated_buildings, user_distance, max_associations
        )

        # Merge poles
        gdf_new_poles = gdf_new_poles.reset_index(drop=True)

        # Drop helper column before merge
        gdf_associated_poles_export = gdf_associated_poles.drop(columns=["old_pole_id"], errors="ignore")
        gdf_final_poles = pd.concat([gdf_associated_poles_export, gdf_new_poles], ignore_index=True)
        gdf_final_poles = gpd.GeoDataFrame(gdf_final_poles, geometry="geometry", crs=gdf_buildings.crs)

        # Update associations with new poles
        # new_associations: (pole_geometry, building_id)
        new_associations_df = pd.DataFrame(new_associations, columns=['pole', 'building_id'])

        # Map new pole geometries to their ids in the merged gdf_final_poles
        offset_new = len(gdf_associated_poles_export)
        geom_to_local = {geom.wkb: i for i, geom in enumerate(gdf_new_poles.geometry)}
        def _map_new_pole_id(geom):
            key = geom.wkb
            if key not in geom_to_local:
                return None
            return int(offset_new + geom_to_local[key])

        new_associations_df['pole_id'] = new_associations_df['pole'].apply(_map_new_pole_id)
        if new_associations_df['pole_id'].isna().any():
            n_bad = int(new_associations_df['pole_id'].isna().sum())
            st.warning(f"{n_bad} new-association rows could not be mapped to new pole ids and were dropped.")
            new_associations_df = new_associations_df.dropna(subset=['pole_id']).copy()
        new_associations_df['pole_id'] = new_associations_df['pole_id'].astype(int)

        # Append new associations
        associations_df = pd.concat(
            [associations_df, new_associations_df[['pole_id', 'building_id']]],
            ignore_index=True
        )

        # Minimum Spanning Tree (MST)
        mst = create_graph_and_mst(gdf_final_poles)

        # Post-process: insert intermediate poles on any long MST branch
        if densify_long_edges:
            gdf_final_poles, mst = densify_mst_edges(gdf_final_poles, mst, sampling_distance)

        # Post-process: add intermediate poles along long branches so max edge <= sampling_distance
        if densify_long_edges:
            gdf_final_poles, mst = densify_mst_edges(gdf_final_poles, mst, sampling_distance)

        # Calculate project information
        number_of_buildings = len(gdf_buildings)
        distribution_network_length = sum(nx.get_edge_attributes(mst, 'weight').values()) / 1000
        number_of_poles_code = len(gdf_final_poles)
        number_of_poles_length = math.ceil((distribution_network_length * 1000) / sampling_distance)

        # Display project metrics
        st.subheader("Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("N° connections", number_of_buildings)
        col2.metric("Network Length (km)", f"{distribution_network_length:.2f}")
        col3.metric("N° of Poles (Code)", number_of_poles_code)
        col4.metric("N° of Poles (Length)", number_of_poles_length)

        # Display charts
        plot_map_with_mst_and_connections(gdf_roads, gdf_final_poles, gdf_buildings, mst)

        # Save MST as GeoJSON
        nodes_geojson, edges_geojson = save_mst_to_geojson(gdf_final_poles, mst)

        # =================================================================================
        # NEW FIX: Remap associations pole_id to match mst_nodes.geojson 'id' (0..N-1)
        # =================================================================================
        # mst_nodes.geojson uses enumerate(gdf_final_poles.itertuples()) => id = 0..N-1
        pole_index_to_node_id = {pole.Index: i for i, pole in enumerate(gdf_final_poles.itertuples())}

        associations_export_df = associations_df.copy()
        associations_export_df["pole_id"] = associations_export_df["pole_id"].map(pole_index_to_node_id)

        # Drop rows that could not be mapped (should be zero; if not, you want to know)
        if associations_export_df["pole_id"].isna().any():
            n_bad = int(associations_export_df["pole_id"].isna().sum())
            st.warning(f"{n_bad} association rows could not be mapped to node ids and were dropped.")
            associations_export_df = associations_export_df.dropna(subset=["pole_id"])

        associations_export_df["pole_id"] = associations_export_df["pole_id"].astype(int)

        # Prepare Associations CSV (export version!)
        associations_csv = associations_export_df.to_csv(index=False).encode("utf-8")

        # Allow user to download outputs
        st.subheader("Download Outputs")
        st.download_button(
            "Download Associations CSV",
            data=associations_csv,
            file_name="associations.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download Nodes GeoJSON",
            nodes_geojson,
            file_name="mst_nodes.geojson",
            mime="application/geo+json"
        )
        st.download_button(
            "Download Edges GeoJSON",
            edges_geojson,
            file_name="mst_edges.geojson",
            mime="application/geo+json"
        )
