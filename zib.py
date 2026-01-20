"""
Master's Thesis: Geodata and automated GIS processes in the implementation of fiber optic infrastructure
=====================================================

This python script processes broadband infrastructure (FTTX) and address data (AGWR)
to categorize locations based on geometric proximity and attribute matching.

Author: Niklas Terler
Institution: University of Graz
Date: January 2026
"""

import os
import json
import requests
import pandas as pd
import geopandas as gpd
from typing import Optional, Dict
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv


# ============================================================================
# Configuration and Environment Setup
# ============================================================================

load_dotenv("config.env")

A10_GEBIETE: list[str] = ["3501-00_St.Nikolai"]
API_URL: Optional[str] = os.getenv("API_URL")
TARGET_EPSG: int = 3035
OUTPUT_EPSG: int = 3035
METRIC_CRS: str = "EPSG:3857"


# ============================================================================
# SQL Query Templates
# ============================================================================

sql_templates = {
    "fttx": """
        SELECT DISTINCT ON (df.dim_locations_id)
            fh.homes_passed_anzahl,
            df.executionstate,
            df.address,
            df.adrcd_subcd,
            df.lotnumber,
            df.zipcode,
            dg.katastralgemeinde_nr,
            ds.salesclusters_userlabel_aktuell,
            ds.wkb AS salescluster_wkb,
            df.objektnummer,
            df.aktuell_status,
            df.wkb AS geometry_fttx
        FROM prod_marts.fct_homes fh
        LEFT JOIN prod_marts.dim_locations df ON fh.dim_locations_id = df.dim_locations_id
        LEFT JOIN prod_marts.dim_salesclusters ds ON fh.dim_salesclusters_id = ds.dim_salesclusters_id
        LEFT JOIN prod_marts.dim_gemeinden dg ON fh.dim_gemeinden_id = dg.dim_gemeinden_id
        WHERE fh.homes_passed_anzahl = '1'
          AND df.aktuell_status = 'Aktuell'
          AND df.lotnumber IS NOT NULL
          AND dg.katastralgemeinde_nr IS NOT NULL
          AND ds.salesclusters_userlabel_aktuell = '{cluster}'
    """,
    "agwr": """
        SELECT 
            gwr.kgnr,
            gwr.objnr,
            gwr."adrcd-subcd" AS adrcd_subcd_agwr,
            gwr.plz,
            gwr.ort,
            gwr.strasse,
            gwr.hnr,
            gwr.gstnr,
            gwr.wkb AS geometry_agwr
        FROM autonom.gwr_rimo_table gwr
        WHERE ST_Within(
            gwr.wkb,
            (SELECT ST_Union(ds.wkb) AS geometry_salescluster
               FROM prod_marts.dim_salesclusters ds 
              WHERE ds.salesclusters_userlabel_aktuell = '{cluster}')
        )
    """,
}


# ============================================================================
# Database Connection
# ============================================================================

def _get_engine() -> Engine:
    """
    Creates and returns a PostgreSQL database connection engine.
    
    Reads database credentials from environment variables (config.env).
    
    Returns:
        sqlalchemy.engine.Engine: Configured database connection engine
        
    Raises:
        ValueError: If required environment variables are missing
    """
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME")
    
    if not all([user, password, host, dbname]):
        raise ValueError("Missing required database environment variables")
    
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(conn_str)


# ============================================================================
# Data Loading
# ============================================================================

def _load_gdf(sql: str, engine: Engine, geom_col: str = "wkb") -> gpd.GeoDataFrame:
    """
    Loads a GeoDataFrame from database using a SQL query.
    
    Automatically detects SRID from database and transforms to OUTPUT_EPSG.
    
    Args:
        sql: SQL query string to execute
        engine: SQLAlchemy database engine
        geom_col: Column name containing geometry (default: 'wkb')
    
    Returns:
        gpd.GeoDataFrame: Loaded data with CRS set to OUTPUT_EPSG
        
    Raises:
        Exception: If database query or CRS detection fails
    """
    try:
        print(f"[DEBUG] Executing SQL query on column '{geom_col}'...")
        wrapped_sql = f"SELECT t.*, ST_SRID(t.{geom_col}) AS srid FROM ({sql}) t"
        gdf = gpd.read_postgis(wrapped_sql, engine, geom_col=geom_col)
        
        # Detect and set CRS if not already set
        if gdf.crs is None and "srid" in gdf.columns and gdf["srid"].notna().any():
            try:
                srid = int(gdf["srid"].mode(dropna=True)[0])
                gdf = gdf.set_crs(epsg=srid, allow_override=True)
                print(f"[DEBUG] CRS detected: EPSG:{srid}")
            except Exception as e:
                print(f"[WARN] Could not detect CRS: {e}")
        
        # Transform to OUTPUT_EPSG if needed
        if gdf.crs is not None and gdf.crs.to_epsg() != OUTPUT_EPSG:
            print(f"[DEBUG] Transforming from EPSG:{gdf.crs.to_epsg()} to EPSG:{OUTPUT_EPSG}")
            gdf = gdf.to_crs(epsg=OUTPUT_EPSG)
        
        gdf = gdf.drop(columns=["srid"], errors="ignore")
        print(f"[DEBUG] Loaded {len(gdf)} features")
        return gdf
        
    except Exception as e:
        print(f"[ERROR] Database query failed: {e}")
        print(f"[DEBUG] SQL: {sql[:200]}...")
        raise


# ============================================================================
# Utility Functions - String Normalization
# ============================================================================

def _normalize_str(value: Optional[object]) -> str:
    """
    Normalizes string values by removing whitespace and handling null values.
    
    Args:
        value: Input value to normalize
    
    Returns:
        str: Normalized string (empty string if null/NA)
    """
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ["nan", "none", "null", ""]:
        return ""
    return s.replace(" ", "")


def _is_null_or_empty(value: Optional[object]) -> bool:
    """
    Check if a value is null or empty string.
    
    Args:
        value: Value to check
        
    Returns:
        bool: True if null/NA/empty, False otherwise
    """
    if value is None:
        return True
    s = str(value).strip().lower()
    return s in ["nan", "none", "null", ""]


# ============================================================================
# API Integration - Connection Point Lookups
# ============================================================================

def _fetch_api_point(objnr: str, cache: Dict[str, Optional[Point]]) -> Optional[Point]:
    """
    Fetches connection point coordinates from ZIB API using object number.
    
    Results are cached to avoid redundant API calls for identical object numbers.
    
    Args:
        objnr: Object number to query
        cache: Cache dictionary for API results (modified in place)
    
    Returns:
        Optional[Point]: Shapely Point geometry or None if not found
    """
    if _is_null_or_empty(objnr):
        return None
    
    if objnr in cache:
        return cache[objnr]
    
    try:
        print(f"[DEBUG] API query: object number '{objnr}'")
        resp = requests.get(API_URL, params={"objnr": str(objnr)}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("response", {}).get("docs", [])
        
        if not docs:
            cache[objnr] = None
            return None
        
        coords = docs[0].get("coordinates", {})
        wkt_str = coords.get("wkt")
        if not wkt_str:
            cache[objnr] = None
            return None
        
        # Remove SRID prefix if present (e.g., "SRID=4326;POINT(...)")
        if wkt_str.startswith("SRID="):
            wkt_str = wkt_str.split(";", 1)[1]
        
        point = wkt.loads(wkt_str)
        cache[objnr] = point
        print(f"[DEBUG] Point found for object '{objnr}'")
        return point
        
    except Exception as e:
        print(f"[WARN] API query failed for object '{objnr}': {e}")
        cache[objnr] = None
        return None


def _fetch_api_point_by_kggst(kg: str, gst: str, 
                              cache: Dict[str, Optional[Point]]) -> Optional[Point]:
    """
    Fetches connection point coordinates using cadastral data.
    
    Queries the API using cadastral municipality number and lot number.
    Cache key format: "KGGST:kg-gst"
    
    Args:
        kg: Cadastral municipality number
        gst: Lot number
        cache: Cache dictionary for API results (modified in place)
    
    Returns:
        Optional[Point]: Shapely Point geometry or None if not found
    """
    if not kg or not gst:
        return None
    
    key = f"KGGST:{kg}-{gst}"
    if key in cache:
        return cache[key]
    
    try:
        print(f"[DEBUG] API query: cadastral lookup '{key}'")
        resp = requests.get(API_URL, params={"kg": kg, "gstnr": gst}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("response", {}).get("docs", [])
        
        if not docs:
            cache[key] = None
            return None
        
        coords = docs[0].get("coordinates", {})
        wkt_str = coords.get("wkt")
        if not wkt_str:
            cache[key] = None
            return None
        
        # Remove SRID prefix if present
        if wkt_str.startswith("SRID="):
            wkt_str = wkt_str.split(";", 1)[1]
        
        point = wkt.loads(wkt_str)
        cache[key] = point
        print(f"[DEBUG] Point found for cadastral '{key}'")
        return point
        
    except Exception as e:
        print(f"[WARN] API query failed for cadastral '{key}': {e}")
        cache[key] = None
        return None


# ============================================================================
# Categorization Functions
# ============================================================================

def _category_1(row: pd.Series) -> Optional[str]:
    """
    Category 1: Exact object number match within 53m threshold.
    
    Requires:
    - Matching object numbers (objektnummer == objnr)
    - Matching address codes (adrcd_subcd == adrcd_subcd_agwr)
    - Distance < 53 meters
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 1" if criteria met, else None
    """
    obj_fttx = _normalize_str(row.get("objektnummer"))
    obj_agwr = _normalize_str(row.get("objnr"))
    
    if not obj_fttx or not obj_agwr or obj_fttx != obj_agwr:
        return None
    
    dist = row.get("dist_api_m")
    if dist is not None and dist < 53:
        if row.get("adrcd_subcd") == row.get("adrcd_subcd_agwr"):
            return "Category 1"
    
    return None


def _category_2(row: pd.Series) -> Optional[str]:
    """
    Category 2: Object number match within 53m but address code mismatch.
    
    Requires:
    - Matching object numbers (objektnummer == objnr)
    - Differing address codes (adrcd_subcd != adrcd_subcd_agwr)
    - Distance < 53 meters
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 2" if criteria met, else None
    """
    obj_fttx = _normalize_str(row.get("objektnummer"))
    obj_agwr = _normalize_str(row.get("objnr"))
    
    if not obj_fttx or not obj_agwr or obj_fttx != obj_agwr:
        return None
    
    dist = row.get("dist_api_m")
    if dist is not None and dist < 53:
        if row.get("adrcd_subcd") != row.get("adrcd_subcd_agwr"):
            return "Category 2"
    
    return None


def _category_3(row: pd.Series) -> Optional[str]:
    """
    Category 3: Address code match but different object numbers within 53m.
    
    Requires:
    - Matching address codes (adrcd_subcd == adrcd_subcd_agwr)
    - Different object numbers (objektnummer != objnr)
    - Distance < 53 meters
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 3" if criteria met, else None
    """
    if row.get("adrcd_subcd") != row.get("adrcd_subcd_agwr"):
        return None
    
    obj_fttx = _normalize_str(row.get("objektnummer"))
    obj_agwr = _normalize_str(row.get("objnr"))
    
    if not obj_fttx or not obj_agwr or obj_fttx == obj_agwr:
        return None
    
    dist = row.get("dist_api_m")
    if dist is not None and dist < 53:
        return "Category 3"
    
    return None


def _category_4(row: pd.Series) -> Optional[str]:
    """
    Category 4: No object number, found via KGGST lookup within 53m.
    
    Requires:
    - No object number (objektnummer is empty/null)
    - Found via KGGST cadastral lookup
    - Distance < 53 meters
    - Valid cadastral data (katastralgemeinde_nr and lotnumber present)
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 4" if criteria met, else None
    """
    if not _is_null_or_empty(row.get("objektnummer")):
        return None
    
    used_search = str(row.get("api_search_used", "") or "")
    if not used_search.startswith("KGGST"):
        return None
    
    dist = row.get("dist_api_m")
    if dist is None:
        return None
    
    try:
        dist_val = float(str(dist).replace(",", "."))
        if dist_val >= 53.0:
            return None
    except (ValueError, TypeError):
        return None
    
    kg_fttx = _normalize_str(row.get("katastralgemeinde_nr"))
    lot = _normalize_str(row.get("lotnumber"))
    if kg_fttx and lot:
        return "Category 4"
    
    return None


def _category_5(row: pd.Series) -> Optional[str]:
    """
    Category 5: Object number found but distance exceeds 53m threshold.
    
    Requires:
    - Valid object number (objektnummer present)
    - API point found via object number lookup
    - Distance > 53 meters
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 5" if criteria met, else None
    """
    if _is_null_or_empty(row.get("objektnummer")):
        return None
    
    if not row.get("api_found", False):
        return None
    
    used_search = str(row.get("api_search_used", "") or "")
    if not used_search.startswith("OBJ:"):
        return None
    
    dist = row.get("dist_api_m")
    if dist is None:
        return None
    
    try:
        dist_val = float(str(dist).replace(",", "."))
        if dist_val > 53.0:
            return "Category 5"
    except (ValueError, TypeError):
        pass
    
    return None


def _category_6(row: pd.Series) -> Optional[str]:
    """
    Category 6: Object number present but API lookup failed.
    
    Requires:
    - Valid object number (objektnummer present)
    - API point not found
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 6" if criteria met, else None
    """
    if not _is_null_or_empty(row.get("objektnummer")) and not row.get("api_found", False):
        return "Category 6"
    
    return None


def _category_7(row: pd.Series) -> Optional[str]:
    """
    Category 7: Found via object number, within 53m, no AGWR match.
    
    Requires:
    - Valid object number (objektnummer present)
    - API point found via object number
    - Distance < 53 meters
    - No AGWR object number match
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 7" if criteria met, else None
    """
    obj_fttx = _normalize_str(row.get("objektnummer"))
    if not obj_fttx:
        return None
    
    if not row.get("api_found", False):
        return None
    
    used = str(row.get("api_search_used", "") or "")
    if not used.startswith("OBJ:"):
        return None
    
    dist = row.get("dist_api_m")
    if dist is None:
        return None
    
    try:
        dist_val = float(str(dist).replace(",", "."))
        if dist_val >= 53.0:
            return None
    except (ValueError, TypeError):
        return None
    
    agwr_obj = _normalize_str(row.get("objnr"))
    if agwr_obj:
        return None
    
    return "Category 7"


def _category_8(row: pd.Series) -> Optional[str]:
    """
    Category 8: Found via KGGST lookup but distance exceeds 53m threshold.
    
    Requires:
    - No object number (objektnummer is empty/null)
    - Found via KGGST cadastral lookup
    - API point found
    - Distance >= 53 meters
    - Valid cadastral data present
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "Category 8" if criteria met, else None
    """
    if not _is_null_or_empty(row.get("objektnummer")):
        return None
    
    used_search = str(row.get("api_search_used", "") or "")
    if not used_search.startswith("KGGST:"):
        return None
    
    if not row.get("api_found", False):
        return None
    
    dist = row.get("dist_api_m")
    if dist is None:
        return None
    
    try:
        dist_val = float(str(dist).replace(",", "."))
        if dist_val <= 53.0:
            return None
    except (ValueError, TypeError):
        return None
    
    kg_fttx = _normalize_str(row.get("katastralgemeinde_nr"))
    lot = _normalize_str(row.get("lotnumber"))
    if kg_fttx and lot:
        return "Category 8"
    
    return None


def _category_none(row: pd.Series) -> Optional[str]:
    """
    Catch-all category for locations without object number.
    
    Assigns "No category" to rows where objektnummer is null or invalid.
    
    Args:
        row: Data row with required fields
    
    Returns:
        Optional[str]: "No category" if criteria met, else None
    """
    if _is_null_or_empty(row.get("objektnummer")):
        return "No category"
    
    return None


def _assign_category(row: pd.Series) -> str:
    """
    Assigns a category to a location by testing categorization rules in order.
    
    Categories are tested in order (1-8) until one matches. If no specific
    category matches, the location is assigned "No category".
    
    Args:
        row: Data row with all required fields
    
    Returns:
        str: Assigned category name
    """
    for func in [_category_1, _category_2, _category_3, _category_4, 
                 _category_5, _category_6, _category_7, _category_8, _category_none]:
        result = func(row)
        if result is not None:
            return result
    
    return "No category"


# ============================================================================
# Data Export - Export Value Generation
# ============================================================================

def _build_export_zib(row: pd.Series) -> str:
    """
    Builds the ZIB export value based on assigned category.
    
    Returns the appropriate identifier or action code based on the category:
    - Category 1: Object number
    - Category 2: "check adrcd-subcd"
    - Category 3: "check objektnummer"
    - Category 4: KGGST cadastral code (kg-lot)
    - Category 5: "check distance"
    - Category 6: "check objektnummer"
    - Category 7: Object number
    - Category 8: "check distance"
    - No category: Empty string
    
    Args:
        row: Data row with category assignment
    
    Returns:
        str: Export value to include in output
    """
    cat = row.get("categories_export")
    
    def _get_object():
        v = row.get("objektnummer")
        if v is None:
            return ""
        s = str(v).strip()
        return s if not _is_null_or_empty(s) else ""
    
    def _get_kggst():
        kg = _normalize_str(row.get("katastralgemeinde_nr"))
        lot = _normalize_str(row.get("lotnumber"))
        if kg and lot:
            return f"{kg}-{lot}"
        kgnr = _normalize_str(row.get("kgnr"))
        gst = _normalize_str(row.get("gstnr"))
        if kgnr and gst:
            return f"{kgnr}-{gst}"
        return ""
    
    category_mapping = {
        "Category 1": _get_object(),
        "Category 2": "check adrcd-subcd",
        "Category 3": "check objektnummer",
        "Category 4": _get_kggst(),
        "Category 5": "check distance",
        "Category 6": "check objektnummer",
        "Category 7": _get_object(),
        "Category 8": "check distance",
    }
    
    return category_mapping.get(cat, "")


# ============================================================================
# Cluster Processing
# ============================================================================

def _process_cluster(cluster: str, engine: Engine, 
                    cache: Dict[str, Optional[Point]]) -> gpd.GeoDataFrame:
    """
    Processes a single sales cluster, performing all data merging and 
    categorization steps.
    
    Steps:
    1. Load FTTX data (FTTX homes)
    2. Load AGWR data (official address register)
    3. Merge on object number
    4. Query API for connection points (both object and cadastral lookups)
    5. Calculate distances to API points
    6. Assign categories
    7. Generate export values
    
    Args:
        cluster: Cluster name/identifier
        engine: Database connection
        cache: Cache for API results (modified in place)
    
    Returns:
        gpd.GeoDataFrame: Processed features with categories and export values
    """
    print(f"\n[INFO] Loading data for cluster: {cluster}")
    fttx_sql = sql_templates["fttx"].format(cluster=cluster)
    agwr_sql = sql_templates["agwr"].format(cluster=cluster)

    # Load FTTX data
    try:
        fttx_gdf = _load_gdf(fttx_sql, engine, geom_col="geometry_fttx")
        print(f"[INFO] FTTX: {len(fttx_gdf)} features loaded")
    except Exception as e:
        print(f"[ERROR] Loading FTTX failed: {type(e).__name__}: {str(e)[:100]}")
        return gpd.GeoDataFrame()

    # Load AGWR data
    try:
        agwr_gdf = _load_gdf(agwr_sql, engine, geom_col="geometry_agwr")
        print(f"[INFO] AGWR: {len(agwr_gdf)} features loaded")
    except Exception as e:
        print(f"[WARN] Loading AGWR failed (continuing without): {str(e)[:100]}")
        agwr_gdf = gpd.GeoDataFrame()

    if fttx_gdf.empty:
        print(f"[WARN] No FTTX data available, skipping cluster")
        return gpd.GeoDataFrame()

    # Prepare data types
    fttx_gdf["objektnummer"] = fttx_gdf["objektnummer"].astype(str)
    fttx_gdf["adrcd_subcd"] = fttx_gdf["adrcd_subcd"].astype(str)

    # Merge with AGWR on object number
    print("[INFO] Merging FTTX with AGWR data...")
    if not agwr_gdf.empty:
        agwr_gdf["objnr"] = agwr_gdf["objnr"].astype(str)
        agwr_gdf["adrcd_subcd_agwr"] = agwr_gdf["adrcd_subcd_agwr"].astype(str)
        agwr_idx = agwr_gdf.drop_duplicates(subset=["objnr"], keep="first")
        merged = fttx_gdf.merge(
            agwr_idx,
            how="left",
            left_on="objektnummer",
            right_on="objnr",
            suffixes=("", "_agwr")
        )
    else:
        merged = fttx_gdf.copy()
        merged["objnr"] = None
        merged["adrcd_subcd_agwr"] = None

    # Set geometry and transform to metric CRS for distance calculations
    geom_col = "geometry_fttx" if "geometry_fttx" in merged.columns else merged.geometry.name
    merged = gpd.GeoDataFrame(merged, geometry=geom_col, crs=fttx_gdf.crs)
    
    print(f"[INFO] Transforming to metric CRS ({METRIC_CRS}) for distance calculations...")
    merged_metric = merged.to_crs(METRIC_CRS)

    # Query API for connection points
    print("[INFO] Querying API for connection points...")
    distances, api_wkts, api_found = [], [], []
    used_searches = []

    for idx, row in merged_metric.iterrows():
        objnr = _normalize_str(row.get("objektnummer"))
        api_point = None
        used_search = None
        
        # Try object number lookup first
        if objnr:
            api_point = _fetch_api_point(objnr, cache)
            used_search = f"OBJ:{objnr}"
        else:
            # Fall back to cadastral lookup
            kg_fttx = _normalize_str(row.get("katastralgemeinde_nr"))
            lot = _normalize_str(row.get("lotnumber"))
            if kg_fttx and lot:
                api_point = _fetch_api_point_by_kggst(kg_fttx, lot, cache)
                used_search = f"KGGST:{kg_fttx}-{lot}"
        
        # Calculate distance if point found
        if api_point:
            api_point_metric = gpd.GeoSeries([api_point], crs=METRIC_CRS).iloc[0]
            distances.append(merged_metric.geometry.iloc[idx].distance(api_point_metric))
            api_wkts.append(api_point.wkt)
            api_found.append(True)
        else:
            distances.append(None)
            api_wkts.append(None)
            api_found.append(False)
        
        used_searches.append(used_search)

    # Add API results to dataframe
    merged["dist_api_m"] = distances
    merged["api_geom_wkt"] = api_wkts
    merged["api_found"] = api_found
    merged["api_search_used"] = used_searches

    # Assign categories
    print("[INFO] Assigning categories...")
    merged["categories_export"] = merged.apply(_assign_category, axis=1)
    merged["export_zib"] = merged.apply(_build_export_zib, axis=1)

    # Print summary
    cat_dist = merged['categories_export'].value_counts().to_dict()
    print(f"[INFO] Category distribution: {cat_dist}")

    return merged


# ============================================================================
# Main Workflow - Data Export
# ============================================================================

def build_export_geojson(out_path: str = "export_categories_zib.geojson") -> Optional[str]:
    """
    Main processing function that orchestrates cluster processing and output.
    
    Steps:
    1. Initialize database connection
    2. Process each cluster (merging, API queries, categorization)
    3. Merge all results
    4. Transform to output CRS (EPSG:3035)
    5. Export to GeoJSON
    6. Print summary statistics
    
    Args:
        out_path: Output file path for categories GeoJSON
    
    Returns:
        Optional[str]: Path to output file, or None if no results
        
    Raises:
        Exception: If database connection or cluster processing fails
    """
    print("[INFO] ========== Starting ZIB categorization workflow ==========")
    
    try:
        engine = _get_engine()
        print("[INFO] Database connection established")
    except Exception as e:
        print(f"[ERROR] Failed to establish database connection: {e}")
        return None
    
    api_cache: Dict[str, Optional[Point]] = {}
    results = []

    # Process each cluster
    for i, cluster in enumerate(A10_GEBIETE):
        print(f"\n[INFO] [{i+1}/{len(A10_GEBIETE)}] Processing cluster: {cluster}")
        try:
            result = _process_cluster(cluster, engine, api_cache)
            if not result.empty:
                results.append(result)
                print(f"[INFO] ✓ Processed {len(result)} features from {cluster}")
            else:
                print(f"[WARN] No results for {cluster}")
            del result
        except Exception as e:
            print(f"[ERROR] Processing failed for {cluster}: {type(e).__name__}: {str(e)[:100]}")
            continue

    if not results:
        print("\n[ERROR] No results produced across all clusters!")
        return None

    # Merge all cluster results
    print(f"\n[INFO] Merging results from {len(results)} cluster(s)...")
    final_gdf = pd.concat(results, ignore_index=True)
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry="geometry_fttx", crs="EPSG:4326")
    print(f"[INFO] Merged dataset contains {len(final_gdf)} features")

    # Transform to output CRS
    print(f"[INFO] Transforming to output CRS (EPSG:{OUTPUT_EPSG})...")
    final_gdf = final_gdf.to_crs(epsg=OUTPUT_EPSG)

    # Select export columns
    export_cols = [
        "categories_export",
        "objektnummer",
        "objnr",
        "adrcd_subcd",
        "adrcd_subcd_agwr",
        "katastralgemeinde_nr",
        "kgnr",
        "lotnumber",
        "gstnr",
        "zipcode",
        "plz",
        "address",
        "ort",
        "strasse",
        "hnr",
        "api_search_used",
        "dist_api_m",
        "api_geom_wkt",
        "export_zib",
        "geometry_fttx",
    ]
    export_cols = [c for c in export_cols if c in final_gdf.columns]
    final_gdf = final_gdf[export_cols]

    # Export to GeoJSON
    print(f"[INFO] Exporting to GeoJSON: {out_path}")
    try:
        final_gdf.to_file(out_path, driver="GeoJSON")
        print(f"[INFO] ✓ Successfully exported {len(final_gdf)} features to {out_path}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        return None

    # Print summary statistics
    print("\n[INFO] ========== Export Summary ==========")
    print(f"[INFO] Total features exported: {len(final_gdf)}")
    print("[INFO] Category distribution:")
    print(final_gdf["categories_export"].value_counts(dropna=False))
    print("[INFO] ========== Workflow completed ==========\n")

    return out_path


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    build_export_geojson()
