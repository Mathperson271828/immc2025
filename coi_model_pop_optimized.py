# ==== STEP 1: load data and choose features ====
import pandas as pd
import numpy as np
import glob
import geopandas as gpd
import re

from libpysal.weights import Queen
from sklearn.preprocessing import StandardScaler
from spopt.region import Skater

CSV_PATH = "IMMC Data - FinalData.csv"   # make sure this is correct

df = pd.read_csv(CSV_PATH)

print("\n====== DEBUG: DF COLUMNS ======")
print(df.columns.tolist())

# Clean county strings
df["County"] = df["County"].astype(str).str.strip()

# Population as numeric
POP_COL = "2023 Population"
df[POP_COL] = pd.to_numeric(df[POP_COL], errors="coerce")

# log-transform it so it's stable in clustering
df["log_population"] = np.log(df[POP_COL])

print("Loaded rows from CSV:", len(df))
print("Columns:", df.columns.tolist())
print()

# These are the REAL column names from your dataset
feature_cols = [
    "Population Density",
    "Urban-Rural Classification",
    "Diversity Index",
    "Partisan Voting Index",
    "PerCapitaIncome_2022",
    "MedianListPrice_2024",
    "HS_Grad_or_Higher_2023_5yr_Pct",
    "Value (Percentage of People with a Bachelor's)",
    "HighPoint_Elevation_ft",
    "WaterArea_sqmi",
    "AgLand_Acres_2022",
    "Air Pollution (Micrograms per cm^3)",
    "Unemployment Rate (%) ",
    "Economic Diversity Index",
    "Mean Travel Time",
    "Mobility (%)",
    "Obesity %",
    "log_population",
]

print("Using", len(feature_cols), "features:")
for c in feature_cols:
    print("  -", c)

print("\n==== STEP 2: load shapefile and merge ====")

# look for any .shp file in these folders
shp_candidates = glob.glob("NC_Counties_2025/*.shp") + \
                 glob.glob("tl_2025_us_county/*.shp")

if not shp_candidates:
    raise FileNotFoundError("No .shp file found in NC_Counties_2025/ or tl_2025_us_county/")

SHP_PATH = shp_candidates[0]
print("Using shapefile:", SHP_PATH)

gdf = gpd.read_file(SHP_PATH)
print("Geo columns:", gdf.columns.tolist())

print("\n====== DEBUG: GDF COLUMNS (RAW) ======")
print(gdf.columns.tolist())

# try to find the county-name column
possible_name_cols = ["NAME", "County", "COUNTY", "COUNTY_NAM", "county"]
name_col = None
for c in possible_name_cols:
    if c in gdf.columns:
        name_col = c
        break

if name_col is None:
    raise ValueError("Could not find a county-name column in shapefile.")

print("Detected county-name column in shapefile:", name_col)

# rename to 'County' so it matches our CSV
gdf = gdf.rename(columns={name_col: "County"})
gdf["County"] = gdf["County"].astype(str).str.strip()

print("Unique counties in shapefile before split:", len(gdf["County"].unique()))

# ==== split Wake / Mecklenburg / Guilford in the shapefile
BASE_TO_SPLIT = ["Wake", "Mecklenburg", "Guilford"]

rows_to_add = []
bases_to_drop = set()

for base in BASE_TO_SPLIT:
    if base not in gdf["County"].values:
        print(f"WARNING: base county '{base}' not found in shapefile; skipping.")
        continue

    # If the base name itself still exists in the CSV, then you didn't split it -> skip
    if base in df["County"].values:
        print(f"INFO: '{base}' appears unchanged in CSV (not split); leaving as-is.")
        continue

    # Otherwise, look for CSV rows that contain the base name in their County string
    split_names = sorted([
        c for c in df["County"].unique()
        if (base in c) and (c != base)
    ])

    if not split_names:
        print(f"INFO: No split variants for '{base}' found in CSV; leaving as-is.")
        continue

    print(f"Splitting county '{base}' into:", split_names)

    base_row = gdf[gdf["County"] == base]
    if base_row.empty:
        print(f"WARNING: shapefile row for '{base}' not found when splitting; skipping.")
        continue

    for nn in split_names:
        r = base_row.copy()
        r["County"] = nn
        rows_to_add.append(r)

    bases_to_drop.add(base)

if rows_to_add:
    extra = pd.concat(rows_to_add, ignore_index=True)
    # drop the original unsplit rows
    if bases_to_drop:
        gdf = gdf[~gdf["County"].isin(bases_to_drop)]
    # add duplicated geometry rows with new names
    gdf = pd.concat([gdf, extra], ignore_index=True)

print("Unique counties in shapefile AFTER split logic:", len(gdf["County"].unique()))

# merge geometry with CSV attributes
gdf = gdf.merge(df, on="County", how="inner")
print("Rows after merging geometry + CSV:", len(gdf))

# sanity check: total population in df vs gdf
total_pop_df = df[POP_COL].sum()
total_pop_gdf = gdf[POP_COL].sum()
print("Total pop in df: ", total_pop_df)
print("Total pop in gdf:", total_pop_gdf)
if abs(total_pop_df - total_pop_gdf) > 1:
    print("⚠️ WARNING: population mismatch between df and gdf! Check split names.")
else:
    print("✅ Population totals match between df and gdf.")

print("\n==== STEP 3: clean feature columns, scale, and build adjacency ====")

# 3a. Convert all feature columns to numeric by extracting the numeric part
for col in feature_cols:
    s = gdf[col].astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s_num = s.str.extract(r"(-?\d+\.?\d*)", expand=False)
    gdf[col] = pd.to_numeric(s_num, errors="coerce")

# 3b. Now take only these numeric features
X = gdf[feature_cols].copy()

# Fill any missing values with column means
X = X.fillna(X.mean())

# 3c. Scale features (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Overwrite the feature columns in gdf with the scaled values
for i, col in enumerate(feature_cols):
    gdf[col] = X_scaled[:, i]

print("Feature matrix shape:", X_scaled.shape)

# 3d. Build Queen contiguity weights (adjacency)
w = Queen.from_dataframe(gdf)
print("Number of counties (units in model):", len(gdf))
print("Example neighbors for first county:", list(w.neighbors.items())[0])

print("\n==== STEP 4: run SKATER clustering ====")

K = 14  # number of districts you want; you can change this

print(f"Running SKATER with K = {K}...")
model = Skater(
    gdf,
    w,
    attrs_name=feature_cols,
    n_clusters=K,
)
model.solve()

labels = model.labels_
gdf["District_ID"] = labels

# also attach districts back to the plain pandas df
df = df.merge(gdf[["County", "District_ID"]], on="County", how="left")

print("Cluster labels (first few counties):")
print(df[["County", "District_ID"]].head())


# ==== STEP 4b: refine districts for population + COI similarity + split-county penalty ====

print("\n==== STEP 4b: population-balanced refinement ====")

if POP_COL not in gdf.columns:
    print(f"WARNING: '{POP_COL}' not found in gdf; available columns are:")
    print(gdf.columns.tolist())
    print("Skipping population refinement.")
else:
    total_pop = gdf[POP_COL].sum()
    target_pop = total_pop / K
    print("Total population (from gdf):", total_pop)
    print("Target pop per district:", target_pop)

    neighbors = w.neighbors  # dict: index -> list of neighbor indices

    # NEW: strongly penalize if split county pieces are in the same district
    PENALTY_PAIRS = [
        ["WakeN", "WakeS"],
        ["MecklenburgN", "MecklenburgS"],
        ["GuilfordE", "GuilfordW"],
    ]

    def compute_loss(gdf_local):
        """
        Combined feature + population loss
        + BIG penalty if both halves of a split county are in the same district.
        """
        loss_feat = 0.0
        loss_pop = 0.0

        # --- standard COI + population loss ---
        for d in gdf_local["District_ID"].unique():
            block = gdf_local[gdf_local["District_ID"] == d]
            X_block = block[feature_cols].values
            if len(X_block) == 0:
                continue
            mu = X_block.mean(axis=0)
            loss_feat += ((X_block - mu) ** 2).sum()

            pop = block[POP_COL].sum()
            loss_pop += ((pop - target_pop) / target_pop) ** 2

        # --- penalty for putting both halves of a split county together ---
        df_pairs = gdf_local[["County", "District_ID"]].copy()
        big_penalty = 0.0

        for pair in PENALTY_PAIRS:
            sub = df_pairs[df_pairs["County"].isin(pair)]
            # only penalize if ALL pieces exist AND they share the same district
            if len(sub) == len(pair) and sub["District_ID"].nunique() == 1:
                big_penalty += 1_000_000.0  # gigantic penalty per violating pair

        # weights: tweak these if you want to emphasize one part more
        lambda_feat = 0.02
        lambda_pop = 600

        return lambda_feat * loss_feat + lambda_pop * loss_pop + big_penalty

    def is_connected(idx_set):
        """Check if the subgraph induced by idx_set is connected."""
        if not idx_set:
            return True
        visited = set()
        stack = [next(iter(idx_set))]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for u in neighbors[v]:
                if u in idx_set and u not in visited:
                    stack.append(u)
        return visited == idx_set

    def district_index_sets(gdf_local):
        """Helper: indices of counties in each district."""
        d_sets = {}
        for d in gdf_local["District_ID"].unique():
            d_sets[d] = set(gdf_local.index[gdf_local["District_ID"] == d])
        return d_sets

    current_loss = compute_loss(gdf)
    print("Initial combined loss:", current_loss)

    max_iters = 5000  # you can increase if you want more refinement
    rng = np.random.default_rng(0)

    for it in range(max_iters):
        d_sets = district_index_sets(gdf)

        # pick a random county
        i = int(rng.integers(0, len(gdf)))
        d_cur = gdf.at[i, "District_ID"]

        neigh_districts = set(gdf.loc[neighbors[i], "District_ID"])
        neigh_districts.discard(d_cur)
        if not neigh_districts:
            continue

        best_loss = current_loss
        best_target = None

        for d_new in neigh_districts:
            if len(d_sets[d_cur]) <= 1:
                continue

            old_set = d_sets[d_cur].copy()
            new_set = d_sets[d_new].copy()

            old_set.remove(i)
            new_set.add(i)

            if not is_connected(old_set):
                continue
            if not is_connected(new_set):
                continue

            orig_label = gdf.at[i, "District_ID"]
            gdf.at[i, "District_ID"] = d_new
            trial_loss = compute_loss(gdf)
            gdf.at[i, "District_ID"] = orig_label

            if trial_loss < best_loss:
                best_loss = trial_loss
                best_target = d_new

        if best_target is not None:
            gdf.at[i, "District_ID"] = best_target
            current_loss = best_loss

        if (it + 1) % 500 == 0:
            print(f"Iteration {it+1}, current loss = {current_loss}")

    print("Final combined loss after refinement:", current_loss)

    # Update df to match refined labels
    df = df.drop(columns=["District_ID"], errors="ignore")
    df = df.merge(gdf[["County", "District_ID"]], on="County", how="left")

print("\n==== STEP 5: save results ====")

OUT_CSV = "nc_districts_skater.csv"
df_out_cols = ["County", "District_ID"] + feature_cols
df[df_out_cols].to_csv(OUT_CSV, index=False)

OUT_SHP = "nc_districts_skater.shp"
gdf.to_file(OUT_SHP)

print("✅ Done!")
print("  -> Wrote", OUT_CSV)
print("  -> Wrote", OUT_SHP)

print("\n===== POPULATION BY DISTRICT =====")
pop_by_dist = df.groupby("District_ID")[POP_COL].sum().sort_index()
print(pop_by_dist)

target = df[POP_COL].sum() / K
print("\nTarget population per district:", target)

print("\nPopulation deviation from target:")
print(pop_by_dist - target)
# ==== MAP PREVIEW ====
""" print("\n==== MAP PREVIEW ====")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Use categorical labels so legend is discrete
gdf["District_ID_cat"] = gdf["District_ID"].astype(int).astype("category")

gdf.plot(
    column="District_ID_cat",
    ax=ax,
    legend=True,
    cmap="tab20",          # gives 20 colors (plenty for K=14)
    edgecolor="black",
    linewidth=0.4
)

ax.set_title(f"NC COI Districts (K = {K})", fontsize=18)
ax.set_axis_off()
plt.tight_layout()

plt.show()

# Save map
fig.savefig("nc_coi_districts_map_K14.png", dpi=300)
print("Saved map image to nc_coi_districts_map_K14.png")
 """


#final map output:

# ==== STEP 6: MAP WITH SPECIAL TREATMENT FOR SPLIT COUNTIES ====
print("\n==== MAP PREVIEW WITH SPLIT-COUNTY BLENDS ====")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Work on a copy so we don't accidentally mess up gdf
gdf_plot = gdf.copy()

# Names of the sub-counties as they appear in your CSV
SPLIT_VARIANTS = {
    "Wake": ["WakeN", "WakeS"],
    "Mecklenburg": ["MecklenburgN", "MecklenburgS"],
    "Guilford": ["GuilfordE", "GuilfordW"],
}

# Flatten list of all split names
split_names_flat = [name for lst in SPLIT_VARIANTS.values() for name in lst]

# Base layer: all units that are NOT split (we color them normally)
base_gdf = gdf_plot[~gdf_plot["County"].isin(split_names_flat)].copy()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Build a consistent color for each district ID
cmap = plt.get_cmap("tab20")
unique_ids = sorted(int(d) for d in gdf_plot["District_ID"].unique())
id_to_color = {d: cmap(i % cmap.N) for i, d in enumerate(unique_ids)}

# Assign colors to base (non-split) counties
base_gdf["color"] = base_gdf["District_ID"].astype(int).map(id_to_color)

# Plot base counties
base_gdf.plot(
    ax=ax,
    color=base_gdf["color"],
    edgecolor="black",
    linewidth=0.4,
)

# Now handle the split counties: blend the colors of their districts
for base_name, variants in SPLIT_VARIANTS.items():
    part_gdf = gdf_plot[gdf_plot["County"].isin(variants)]
    if part_gdf.empty:
        # If you didn't actually split this county in the CSV, skip
        continue

    # Which districts do the sub-parts belong to?
    districts_here = sorted(int(d) for d in part_gdf["District_ID"].unique())
    cols = [id_to_color[d] for d in districts_here]

    # Blend the RGBA colors by averaging
    cols_arr = np.array(cols)
    blended = cols_arr.mean(axis=0)  # shape (4,): RGBA

    # The geometry is the union of the duplicated pieces (but they are identical)
    geom = part_gdf.geometry.unary_union

    # Plot this county as one polygon with the blended color
    gpd.GeoSeries([geom]).plot(
        ax=ax,
        color=[blended],
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

# Add a legend for the districts (using the base district colors)
handles = [
    mpatches.Patch(color=id_to_color[d], label=f"District {d}")
    for d in unique_ids
]
ax.legend(
    handles=handles,
    title="District_ID",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.0,
)

ax.set_title(f"NC COI Districts with Split Counties (K = {K})")
ax.set_axis_off()
plt.tight_layout()

# Show the map
plt.show()

# Save to file for your report/slides
fig.savefig("nc_coi_districts_map_splitgradient.png", dpi=300)
print("Saved map image to nc_coi_districts_map_splitgradient.png")
