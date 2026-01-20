"""
Geospatial Infrastructure System (ZIS) Data Processing and Export Module

This module implements a comprehensive workflow for processing, filtering, and exporting 
geospatial data related to broadband infrastructure planning and deployment. It integrates 
data from multiple sources including PostGIS databases, applies spatial and attribute-based 
funding classification, validates geometries, and exports results in standardized formats.

Author: [Author Name]
Institution: [University/Institution]
Date: [Date]
"""

import os
import json
import datetime
from typing import Dict, Optional, Set
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine
from dotenv import load_dotenv


# ============================================================================
# Configuration and Logging Setup
# ============================================================================

load_dotenv("config.env")

_user = os.getenv("DB_USER")
_password = os.getenv("DB_PASSWORD")
_host = os.getenv("DB_HOST")
_dbname = os.getenv("DB_NAME")

# Establish connection to PostgreSQL database with PostGIS extension
engine = create_engine(f"postgresql://{_user}:{_password}@{_host}/{_dbname}")

EXPORT_GPKG_PATH: Optional[str] = None
TARGET_EPSG: int = 3035
REGION = "3501-00_St. Nikolai im Sausal"


# ============================================================================
# SQL Query Templates
# ============================================================================

sql_templates = {
    "shaft_locations": """
        SELECT DISTINCT
            p.wkb::geometry AS geom,
            p.oop AS externalid,
            p.id AS name
        FROM prod_staging.stg_points p
        JOIN prod_marts.dim_baulose b ON ST_Intersects(p.wkb, b.wkb)
        WHERE current_date <@ p.dbt_valid_range
          AND p.physicalcategory = 'ShaftLocation'
          AND p.class = 'SDIBuilding'
          AND b.baulose_userlabel_aktuell = '{region_filter}'
    """,
    "fcp": """
        SELECT DISTINCT
            p.wkb::geometry AS geom,
            p.oop AS externalid,
            p.id AS name
        FROM prod_staging.stg_points p
        JOIN prod_marts.dim_baulose b ON ST_Intersects(p.wkb, b.wkb)
        WHERE current_date <@ p.dbt_valid_range
          AND p.physicalcategory = 'ConcentrationPoint'
          AND b.baulose_userlabel_aktuell = '{region_filter}'
    """,
    "fttx_locations": """
        SELECT DISTINCT
            p.wkb::geometry AS geom,
            p.oop AS externalid,
            p.id AS name
        FROM prod_staging.stg_points p
        JOIN prod_marts.dim_baulose b ON ST_Intersects(p.wkb, b.wkb)
        WHERE current_date <@ p.dbt_valid_range
          AND p.physicalcategory = 'FTTxCustomerLocation'
          AND p.class = 'SDIBuilding'
          AND p.executionstate != '99'
          AND b.baulose_userlabel_aktuell = '{region_filter}'
    """,
    "workorders": """
        SELECT DISTINCT
            f.dim_locations_id,
            d.externalid,
            d.wkb::geometry AS geom
        FROM prod_marts.fct_homes f
        JOIN prod_marts.dim_locations d ON f.dim_locations_id = d.dim_locations_id
        JOIN prod_marts.dim_baulose b ON ST_Intersects(d.wkb, b.wkb)
        WHERE f.services_active_anzahl = 1
          AND b.baulose_userlabel_aktuell = '{region_filter}'
    """,
    "subsidy_polygon": """
        SELECT
            programm,
            foerdernehmerin,
            shape::geometry as geom    
        FROM external_data.bb_atlas_gefoerderter_ausbau
    """,
    "trenches": """
        SELECT DISTINCT
            l.wkb::geometry AS geom,
            l.masteritem,
            l.cableready,
            l.requestoops
        FROM prod_staging.stg_lines l
        JOIN prod_marts.dim_baulose b ON ST_Intersects(l.wkb, b.wkb)
        WHERE current_date <@ l.dbt_valid_range
          AND l.executionstate IN ('5', '6')
          AND b.baulose_userlabel_aktuell = '{region_filter}'
    """,
}


# ============================================================================
# Utility Functions
# ============================================================================

def _is_non_empty(gdf: Optional[gpd.GeoDataFrame]) -> bool:
    """
    Check if a GeoDataFrame is non-empty and valid.
    
    Args:
        gdf: GeoDataFrame to validate
        
    Returns:
        bool: True if GeoDataFrame is not None and contains data, False otherwise
    """
    return gdf is not None and hasattr(gdf, 'empty') and not gdf.empty


def ensure_geometry_column(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    """
    Ensure proper geometry column naming and coordinate reference system (CRS).
    
    Operations performed:
    - Renames 'geom' column to 'geometry' if present
    - Sets geometry column as active geometry
    - Ensures CRS is set to TARGET_EPSG (ETRS89-LAEA)
    - Reprojects to TARGET_EPSG if different CRS is present
    
    Args:
        gdf: Input GeoDataFrame
        
    Returns:
        GeoDataFrame with standardized geometry column and CRS, or None if input is empty
    """
    if not _is_non_empty(gdf):
        return gdf
    
    data = gdf.copy()
    
    # Standardize geometry column naming
    if "geom" in data.columns and "geometry" not in data.columns:
        data = data.rename(columns={"geom": "geometry"})
    
    # Set active geometry column
    if "geometry" in data.columns:
        data = data.set_geometry("geometry", inplace=False)
    
    # Ensure CRS is set and matches TARGET_EPSG
    if data.crs is None:
        data = data.set_crs(epsg=TARGET_EPSG, allow_override=True)
    else:
        data = data.to_crs(epsg=TARGET_EPSG)
    
    return data


# ============================================================================
# Database Operations
# ============================================================================

def _read_postgis(sql: str) -> gpd.GeoDataFrame:
    """
    Execute a SQL query against the PostGIS database and return results as GeoDataFrame.
    
    Args:
        sql: SQL query string
        
    Returns:
        GeoDataFrame containing query results with standardized geometry
        
    Raises:
        Exception: If database query fails
    """
    try:
        print(f"[DEBUG] Executing SQL query...")
        # Wrap SQL to capture SRID
        wrapped_sql = f"SELECT t.*, ST_SRID(t.geom) AS srid FROM ({sql}) t"
        gdf = gpd.read_postgis(wrapped_sql, con=engine, geom_col="geom")
        
        # Detect and set CRS if not already set
        if gdf.crs is None and "srid" in gdf.columns and gdf["srid"].notna().any():
            try:
                srid = int(gdf["srid"].mode(dropna=True)[0])
                gdf = gdf.set_crs(epsg=srid, allow_override=True)
                print(f"[DEBUG] CRS detected: EPSG:{srid}")
            except Exception as e:
                print(f"[WARN] Could not detect CRS: {e}")
        
        # Transform to TARGET_EPSG if needed
        if gdf.crs is not None and gdf.crs.to_epsg() != TARGET_EPSG:
            print(f"[DEBUG] Transforming from EPSG:{gdf.crs.to_epsg()} to EPSG:{TARGET_EPSG}")
            gdf = gdf.to_crs(epsg=TARGET_EPSG)
        
        gdf = gdf.drop(columns=["srid"], errors="ignore")
        return ensure_geometry_column(gdf)
        
    except Exception as e:
        print(f"[ERROR] Database query failed: {e}")
        print(f"[DEBUG] SQL: {sql[:200]}...")
        raise


def load_layers(region_userlabel: str) -> Dict[str, gpd.GeoDataFrame]:
    """
    Load all geospatial layers for a specified region from the database.
    
    Args:
        region_userlabel: Regional identifier for filtering (e.g., "3501-00_St.Nikolai")
        
    Returns:
        Dictionary mapping layer names to GeoDataFrames
    """
    print(f"\n[INFO] ========== Loading all layers for region: {region_userlabel} ==========")
    layers: Dict[str, gpd.GeoDataFrame] = {}
    
    for name, tpl in sql_templates.items():
        sql = tpl.format(region_filter=region_userlabel)
        try:
            print(f"\n[INFO] Loading layer: {name}")
            gdf = _read_postgis(sql)
            layers[name] = gdf
            print(f"[INFO] ✓ {name}: {len(gdf)} features loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {type(e).__name__}: {str(e)[:100]}")
            continue
    
    print(f"\n[INFO] ========== All layers loaded. Total: {len(layers)}/{len(sql_templates)} layers ==========\n")
    return layers


# ============================================================================
# Funding Classification Functions
# ============================================================================

def is_gefoerdert_requestoops(requestoops) -> bool:
    """
    Determine if infrastructure is funded based on requestoops attribute.
    
    Args:
        requestoops: Raw requestoops value (string, JSON, list, or None)
        
    Returns:
        bool: True if funding indicator is present, False otherwise
    """
    if requestoops is None or (isinstance(requestoops, float) and pd.isna(requestoops)):
        return False
    
    try:
        parsed = json.loads(requestoops) if isinstance(requestoops, str) else requestoops
        if isinstance(parsed, (list, tuple, set)):
            return any(str(item).startswith("SDMFunding") for item in parsed)
        return str(parsed).startswith("SDMFunding")
    except (json.JSONDecodeError, TypeError):
        return str(requestoops).startswith("SDMFunding")


def is_kabel_cableready(cableready) -> bool:
    """
    Check if a trench feature is cable-ready.
    
    Args:
        cableready: Cable readiness indicator (boolean or None)
        
    Returns:
        bool: True if cable-ready, False otherwise
    """
    return bool(cableready)


def assign_rtrtyp(
    df: gpd.GeoDataFrame,
    typ: int,
    funded_mask: Optional[pd.Series] = None
) -> gpd.GeoDataFrame:
    """
    Assign infrastructure type codes (RTRTYP) with funding differentiation.
    
    Args:
        df: Input GeoDataFrame
        typ: Base type code (1-5 for different infrastructure types)
        funded_mask: Boolean Series indicating funded features; if None, all treated as unfunded
        
    Returns:
        GeoDataFrame with 'RTRTYP' column assigned
    """
    df = df.copy()
    if funded_mask is None:
        funded_mask = pd.Series([False] * len(df), index=df.index)
    
    df["RTRTYP"] = typ
    df.loc[funded_mask, "RTRTYP"] = typ + 10
    
    return df


# ============================================================================
# Geometry Validation and Cleaning
# ============================================================================

def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and clean geometries in a GeoDataFrame.
    
    Performs multi-stage validation:
    1. Removes invalid geometries
    2. Removes null and empty geometries
    3. Removes non-simple LineStrings/MultiLineStrings
    4. Removes zero-length lines
    5. Filters to allowed geometry types
    6. Removes duplicate geometries
    
    Args:
        gdf: Input GeoDataFrame to clean
        
    Returns:
        Cleaned GeoDataFrame
        
    Raises:
        ValueError: If no active geometry column is present
    """
    gdf = gdf.copy()
    
    if not hasattr(gdf, "geometry") or gdf.geometry is None:
        raise ValueError("GeoDataFrame must have an active geometry column.")
    
    geom_col = gdf.geometry.name
    
    # Stage 1: Remove invalid geometries
    gdf = gdf[gdf.geometry.is_valid]
    
    # Stage 2: Remove null and empty geometries
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    
    # Stage 3: Remove non-simple lines (self-intersecting)
    geom = gdf.geometry
    geom_type = geom.geom_type
    line_mask = geom_type.isin(["LineString", "MultiLineString"])
    nonsimple_mask = line_mask & (~geom.is_simple)
    if nonsimple_mask.any():
        gdf = gdf[~nonsimple_mask]
    
    # Stage 4: Remove zero-length lines
    geom = gdf.geometry
    geom_type = geom.geom_type
    line_mask = geom_type.isin(["LineString", "MultiLineString"])
    zero_length_mask = line_mask & (geom.length == 0)
    if zero_length_mask.any():
        gdf = gdf[~zero_length_mask]
    
    # Stage 5: Filter to allowed geometry types
    allowed_types: Set[str] = {"Point", "MultiPoint", "LineString", "MultiLineString"}
    geom_type = gdf.geometry.geom_type
    type_mask = geom_type.isin(allowed_types)
    gdf = gdf[type_mask]
    
    # Stage 6: Remove duplicate geometries
    gdf = gdf.drop_duplicates(subset=geom_col)
    
    return gdf


# ============================================================================
# Data Export Functions
# ============================================================================

def export_layer_by_funding(
    gdf: Optional[gpd.GeoDataFrame],
    output_prefix: str,
    method: str = "spatial",
    subsidy_polygon: Optional[gpd.GeoDataFrame] = None,
    attribute_column: Optional[str] = None,
    rtrtyp: int = 0,
) -> None:
    """
    Export infrastructure layer with spatial or attribute-based funding classification.
    
    Funding classification methods:
    - 'spatial': Uses spatial join with subsidy polygon to determine funded areas
    - 'attribute': Analyzes attribute values to determine funding status
    
    Args:
        gdf: Input GeoDataFrame to export
        output_prefix: Prefix for output layer names (e.g., 'Verteiler')
        method: Funding classification method ('spatial' or 'attribute')
        subsidy_polygon: GeoDataFrame with subsidy areas (required for 'spatial' method)
        attribute_column: Column name to analyze for funding (required for 'attribute' method)
        rtrtyp: Infrastructure type code (1-5)
        
    Raises:
        ValueError: If method is invalid or required parameters are missing
        Exception: If export to GeoPackage fails (attempts GeoJSON fallback)
    """
    global EXPORT_GPKG_PATH
    
    print(f"[INFO] Starting export for {output_prefix} using {method} method")
    
    # Check if input data is valid
    if not _is_non_empty(gdf):
        print(f"[WARN] Skipping export for {output_prefix}: input layer is empty")
        return
    
    # Initialize export file path if not already done
    if EXPORT_GPKG_PATH is None:
        os.makedirs("data", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        EXPORT_GPKG_PATH = os.path.join("data", f"ZIS_export_{ts}.gpkg")
        print(f"[INFO] Created new export file: {EXPORT_GPKG_PATH}")
    
    working = ensure_geometry_column(gdf.copy())
    
    # ---- Funding Classification ----
    if method == "spatial":
        if not _is_non_empty(subsidy_polygon):
            raise ValueError("subsidy_polygon is required for spatial method")
        
        subsidy = ensure_geometry_column(subsidy_polygon.copy())
        
        # Find features intersecting subsidy areas
        gefoerdert = gpd.sjoin(working, subsidy, how="inner", predicate="intersects")
        gefoerdert = gefoerdert.drop(columns=["index_right"], errors="ignore")
        
        # Find features NOT intersecting subsidy areas
        ungefoerdert = working.loc[~working.index.isin(gefoerdert.index)].copy()
        
        # Assign type codes
        gefoerdert = assign_rtrtyp(
            gefoerdert,
            rtrtyp,
            funded_mask=pd.Series([True] * len(gefoerdert), index=gefoerdert.index)
        )
        ungefoerdert = assign_rtrtyp(
            ungefoerdert,
            rtrtyp,
            funded_mask=pd.Series([False] * len(ungefoerdert), index=ungefoerdert.index)
        )
        
    elif method == "attribute":
        if attribute_column is None:
            raise ValueError("attribute_column is required for attribute method")
        
        working["_is_funded"] = working[attribute_column].apply(is_gefoerdert_requestoops)
        
        gefoerdert = working[working["_is_funded"]].copy()
        ungefoerdert = working[~working["_is_funded"]].copy()
        
        # Assign type codes
        gefoerdert = assign_rtrtyp(
            gefoerdert,
            rtrtyp,
            funded_mask=pd.Series([True] * len(gefoerdert), index=gefoerdert.index)
        )
        ungefoerdert = assign_rtrtyp(
            ungefoerdert,
            rtrtyp,
            funded_mask=pd.Series([False] * len(ungefoerdert), index=ungefoerdert.index)
        )
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'spatial' or 'attribute'")
    
    # ---- Export Funded and Unfunded Layers ----
    for df, tag in ((gefoerdert, "gefoerdert"), (ungefoerdert, "ungefoerdert")):
        if not _is_non_empty(df):
            print(f"[INFO] No features to export for {output_prefix}_{tag} (empty)")
            continue
        
        print(f"[INFO] Cleaning geometries for {output_prefix}_{tag}")
        df = clean_geometries(df)
        
        if not _is_non_empty(df):
            print(f"[INFO] No features left after geometry cleaning for {output_prefix}_{tag}")
            continue
        
        layer_name = f"{output_prefix}_{tag}"
        try:
            print(f"[INFO] Exporting {layer_name} ({len(df)} features) to GeoPackage")
            df.to_file(EXPORT_GPKG_PATH, layer=layer_name, driver="GPKG")
        except Exception as e:
            # Fallback to GeoJSON if GeoPackage export fails
            fallback = os.path.join("data", f"{layer_name}.geojson")
            try:
                df.to_file(fallback, driver="GeoJSON")
                print(f"[WARN] GeoPackage export failed for {layer_name} ({e}). Wrote GeoJSON fallback to {fallback}")
            except Exception as e2:
                print(f"[ERROR] Both GeoPackage and GeoJSON export failed for {layer_name}: {e2}")
    
    print(f"[INFO] Export {output_prefix} completed → {EXPORT_GPKG_PATH}")


# ============================================================================
# Processing Workflows
# ============================================================================

def process_point_layers(layers: Dict[str, gpd.GeoDataFrame], subsidy_polygon: Optional[gpd.GeoDataFrame]) -> None:
    """
    Process and export point-based infrastructure layers.
    
    Exports:
    - FCP (Concentration Points) as 'Verteiler'
    - FTTX Locations as 'Uebergabepunkte'
    - Shaft Locations as 'Schaechte'
    
    Args:
        layers: Dictionary of GeoDataFrames from load_layers()
        subsidy_polygon: GeoDataFrame with subsidy areas
    """
    print("[INFO] Starting processing of point layers")
    
    export_layer_by_funding(
        layers.get("fcp"),
        "Verteiler",
        method="spatial",
        subsidy_polygon=subsidy_polygon,
        rtrtyp=4
    )
    
    export_layer_by_funding(
        layers.get("fttx_locations"),
        "Uebergabepunkte",
        method="spatial",
        subsidy_polygon=subsidy_polygon,
        rtrtyp=1
    )
    
    export_layer_by_funding(
        layers.get("shaft_locations"),
        "Schaechte",
        method="spatial",
        subsidy_polygon=subsidy_polygon,
        rtrtyp=3
    )
    
    print("[INFO] Processing of point layers finished")


def process_trenches_and_cables(layers: Dict[str, gpd.GeoDataFrame]) -> None:
    """
    Process and export trench and cable infrastructure.
    
    Exports:
    - All trenches with funding classification as 'Rohre'
    - Cable-ready trenches and workorder-buffered trenches as 'Kabel'
    
    Args:
        layers: Dictionary of GeoDataFrames from load_layers()
    """
    print("[INFO] Starting processing of trenches and cables")
    
    trenches = layers.get("trenches")
    if not _is_non_empty(trenches):
        print("[WARN] No trenches available")
        return
    
    # Export all trenches
    export_layer_by_funding(
        trenches,
        "Rohre",
        method="attribute",
        attribute_column="requestoops",
        rtrtyp=2
    )
    
    # Process cables (cable-ready trenches)
    kabel = trenches[trenches["cableready"].apply(is_kabel_cableready)].copy()
    
    # Add trenches near workorders
    workorders = layers.get("workorders")
    if _is_non_empty(workorders):
        workorders_buffer = workorders.copy()
        workorders_buffer["geometry"] = workorders_buffer.geometry.buffer(5)
        workorders_buffer = workorders_buffer.set_geometry("geometry")
        workorders_buffer.crs = trenches.crs
        
        joined = gpd.sjoin(trenches, workorders_buffer, how="inner", predicate="intersects")
        kabel_from_workorders = trenches.loc[joined.index.unique()].copy()
        kabel = pd.concat([kabel, kabel_from_workorders]).drop_duplicates(subset="geometry")
    
    if kabel.empty:
        return
    
    export_layer_by_funding(
        kabel,
        "Kabel",
        method="attribute",
        attribute_column="requestoops",
        rtrtyp=5
    )
    
    print("[INFO] Processing of trenches and cables finished")


def main():
    """Main workflow orchestration."""
    print("[INFO] ========== Starting main workflow ==========")
    
    layers = load_layers(REGION)
    subsidy_polygon = layers.get("subsidy_polygon")
    
    process_point_layers(layers, subsidy_polygon)
    process_trenches_and_cables(layers)
    
    print("[INFO] ========== Main workflow finished ==========")

if __name__ == "__main__":
    main()