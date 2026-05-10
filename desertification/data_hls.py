# -*- coding: utf-8 -*-
"""
desertification.data_hls
========================
HLS (Harmonized Landsat-Sentinel) and Sentinel-2 pixel-level data
extraction pipeline.  Replaces MODIS 500 m spatial-average vegetation
indices with 30 m per-pixel time series for dramatically higher
signal-to-noise ratio in ODE discovery.

Key improvements over MODIS-only pipeline
------------------------------------------
- 30 m native resolution (vs 500 m MODIS) → 278× more pixels per ROI
- Per-pixel time series preserves full NDVI dynamic range
- 16-day composites capture post-rainfall greening pulses
- Sentinel-2 red-edge and SWIR bands (not available from MODIS)
- Stratified pixel sampling keeps compute tractable on Colab

Usage
-----
    from desertification.data_hls import load_or_fetch_hls
    df = load_or_fetch_hls(roi, '2005-01-01', '2023-12-31',
                           'rajasthan_hls.csv', n_pixels=1000)
"""

import os
import numpy as np
import pandas as pd

from .config import (
    PIXEL_SAMPLE_COUNT, PIXEL_SAMPLE_SEED,
    TEMPORAL_COMPOSITE_DAYS, S2_MASK_VALUES,
    sentinel2_vars_map, vars_map,
)


# ── Pixel sampling ────────────────────────────────────────────

def sample_pixels(roi, n=None, seed=None):
    """
    Generate stratified random sample points within the ROI.

    Uses MODIS land-cover to stratify by vegetation class so that
    both irrigated and bare-soil pixels are represented.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    n : int, optional
        Number of pixels to sample (default from config).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ee.FeatureCollection
        Point features with 'pixel_id' and 'landcover' properties.
    """
    import ee

    n = n or PIXEL_SAMPLE_COUNT
    seed = seed if seed is not None else PIXEL_SAMPLE_SEED

    # Use MODIS land-cover for stratification
    lc = (ee.ImageCollection('MODIS/061/MCD12Q1')
          .sort('system:time_start', False)
          .first()
          .select('LC_Type1'))

    points = lc.stratifiedSample(
        numPoints=max(n // 10, 10),
        classBand='LC_Type1',
        region=roi,
        scale=30,
        seed=seed,
        geometries=True,
    )

    # If stratified sample yields fewer than n, pad with random points
    count = points.size().getInfo()
    if count < n:
        extra = ee.FeatureCollection.randomPoints(
            region=roi, points=n - count, seed=seed + 1
        )
        # Tag extra points with landcover
        extra = extra.map(lambda f: f.set(
            'LC_Type1',
            lc.reduceRegion(ee.Reducer.first(), f.geometry(), 30)
              .get('LC_Type1')
        ))
        points = points.merge(extra)

    # Add sequential pixel_id
    point_list = points.toList(n)
    def _add_id(i):
        return ee.Feature(point_list.get(i)).set('pixel_id', i)
    points = ee.FeatureCollection(
        ee.List.sequence(0, ee.Number(n).subtract(1)).map(_add_id)
    )

    print(f"  Sampled {n} pixels (stratified by MODIS land cover)")
    return points


# ── Sentinel-2 cloud masking ──────────────────────────────────

def _mask_s2_clouds(image):
    """Mask clouds/shadows using Scene Classification Layer (SCL)."""
    import ee
    scl = image.select('SCL')
    mask = scl.neq(ee.Image.constant(S2_MASK_VALUES[0]))
    for val in S2_MASK_VALUES[1:]:
        mask = mask.And(scl.neq(ee.Image.constant(val)))
    return image.updateMask(mask)


# ── HLS / Sentinel-2 extraction ──────────────────────────────

def _compute_ndvi(image, red_band, nir_band):
    """Compute NDVI from red and NIR bands."""
    red = image.select(red_band).toFloat()
    nir = image.select(nir_band).toFloat()
    return nir.subtract(red).divide(nir.add(red).add(1e-6)).rename('NDVI')


def _compute_evi(image, blue_band, red_band, nir_band):
    """Compute EVI from blue, red, and NIR bands."""
    blue = image.select(blue_band).toFloat()
    red = image.select(red_band).toFloat()
    nir = image.select(nir_band).toFloat()
    num = nir.subtract(red).multiply(2.5)
    den = nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    return num.divide(den.add(1e-6)).rename('EVI')


def _compute_ndmi(image, nir_band, swir_band):
    """Compute NDMI (Normalised Difference Moisture Index)."""
    nir = image.select(nir_band).toFloat()
    swir = image.select(swir_band).toFloat()
    return nir.subtract(swir).divide(nir.add(swir).add(1e-6)).rename('NDMI')


def fetch_sentinel2_pixel_timeseries(roi, pixels, start_date, end_date,
                                      composite_days=None):
    """
    Extract per-pixel vegetation index time series from Sentinel-2 at 10-20 m.

    Computes NDVI (10 m), Red-Edge NDVI (20 m), NDMI (20 m), and
    spatial standard deviation of NDVI across all pixels per composite.

    Parameters
    ----------
    roi : ee.Geometry
    pixels : ee.FeatureCollection
        Sample points from sample_pixels().
    start_date, end_date : str
    composite_days : int, optional
        Compositing window in days (default from config).

    Returns
    -------
    pd.DataFrame
        Columns: [date, pixel_id, NDVI, EVI, RedEdge_NDVI, NDMI,
                  NDVI_spatial_std, lat, lon]
    """
    import ee

    composite_days = composite_days or TEMPORAL_COMPOSITE_DAYS

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
          .map(_mask_s2_clouds))

    # Build composite date sequence
    start_ee = ee.Date(start_date)
    end_ee = ee.Date(end_date)
    n_composites = end_ee.difference(start_ee, 'day').divide(composite_days).ceil()
    date_seq = ee.List.sequence(0, n_composites.subtract(1))

    all_records = []

    def _process_composite(n):
        c_start = start_ee.advance(ee.Number(n).multiply(composite_days), 'day')
        c_end = c_start.advance(composite_days, 'day')
        window = s2.filterDate(c_start, c_end)

        # Compute indices on each image, then median-composite
        def _add_indices(img):
            ndvi = _compute_ndvi(img, 'B4', 'B8')
            evi = _compute_evi(img, 'B2', 'B4', 'B8')
            re_ndvi = img.select('B8A').subtract(img.select('B5')).divide(
                img.select('B8A').add(img.select('B5')).add(1e-6)
            ).rename('RedEdge_NDVI')
            ndmi = _compute_ndmi(img, 'B8', 'B11')
            return img.addBands([ndvi, evi, re_ndvi, ndmi])

        composite = window.map(_add_indices).median().select(
            ['NDVI', 'EVI', 'RedEdge_NDVI', 'NDMI']
        )

        # Spatial std of NDVI across entire ROI (for heterogeneity metric)
        ndvi_std = composite.select('NDVI').reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=roi, scale=30, maxPixels=1e8
        ).get('NDVI')

        # Extract values at sample pixels
        sampled = composite.reduceRegions(
            collection=pixels,
            reducer=ee.Reducer.first(),
            scale=30,
        )

        # Attach date and spatial std
        return sampled.map(lambda f: f.set({
            'date': c_start.format('YYYY-MM-dd'),
            'NDVI_spatial_std': ndvi_std,
        }))

    # Process in chunks to avoid GEE timeout
    n_comp_info = int(n_composites.getInfo())
    chunk_size = 12  # ~6 months per chunk
    print(f"  Extracting {n_comp_info} Sentinel-2 composites "
          f"({composite_days}-day) for {pixels.size().getInfo()} pixels...")

    for chunk_start in range(0, n_comp_info, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_comp_info)
        chunk_indices = list(range(chunk_start, chunk_end))

        for ci in chunk_indices:
            try:
                fc = _process_composite(ci).getInfo()
                for feat in fc.get('features', []):
                    props = feat.get('properties', {})
                    coords = feat.get('geometry', {}).get('coordinates', [None, None])
                    all_records.append({
                        'date': props.get('date'),
                        'pixel_id': props.get('pixel_id', -1),
                        'NDVI': props.get('NDVI'),
                        'EVI': props.get('EVI'),
                        'RedEdge_NDVI': props.get('RedEdge_NDVI'),
                        'NDMI': props.get('NDMI'),
                        'NDVI_spatial_std': props.get('NDVI_spatial_std'),
                        'lon': coords[0] if coords else None,
                        'lat': coords[1] if coords else None,
                    })
            except Exception as e:
                print(f"    Warning: composite {ci} failed: {e}")
                continue

        pct = 100 * chunk_end / n_comp_info
        print(f"    Progress: {chunk_end}/{n_comp_info} composites ({pct:.0f}%)")

    df = pd.DataFrame(all_records)
    if len(df) == 0:
        print("  WARNING: No Sentinel-2 data returned. Falling back to MODIS.")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    # Replace null / masked values
    for col in ['NDVI', 'EVI', 'RedEdge_NDVI', 'NDMI', 'NDVI_spatial_std']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Extracted {len(df)} pixel-time observations "
          f"({df['pixel_id'].nunique()} pixels × "
          f"{df['date'].nunique()} dates)")
    return df


# ── Merge with MODIS climate data ─────────────────────────────

# Standard MODIS cache filenames (same as MODIS-only pipeline)
_MODIS_CACHE_CANDIDATES = {
    'rajasthan': ['rajasthan_fixed.csv', 'rajasthan_modis.csv'],
    'gobi':      ['gobi_fixed.csv',      'gobi_modis.csv'],
}


def _guess_modis_cache(hls_cache_path):
    """Try to find an existing MODIS CSV cache based on the HLS cache name."""
    base = os.path.basename(hls_cache_path).lower()
    dirn = os.path.dirname(hls_cache_path) or '.'
    for key, candidates in _MODIS_CACHE_CANDIDATES.items():
        if key in base:
            for c in candidates:
                p = os.path.join(dirn, c)
                if os.path.exists(p):
                    return p
    return None


def fetch_modis_climate_monthly(roi, start_date, end_date,
                                modis_cache_path=None):
    """
    Fetch MODIS/CHIRPS/FLDAS/GRACE climate variables (spatially averaged).

    Strategy (to avoid GEE timeout on 228-month single call):
      1. If a MODIS CSV cache exists (e.g. rajasthan_fixed.csv from a
         previous MODIS-only run), load it directly — no GEE call needed.
      2. Otherwise, fetch in 3-year chunks and concatenate.

    Parameters
    ----------
    roi : ee.Geometry
    start_date, end_date : str
    modis_cache_path : str, optional
        Explicit path to an existing MODIS CSV cache.

    Returns
    -------
    pd.DataFrame
        Monthly time series with climate columns.
    """
    from .data import fetch_data_failsafe

    # ── Strategy 1: reuse existing MODIS cache ────────────────
    if modis_cache_path and os.path.exists(modis_cache_path):
        print(f"  Reusing existing MODIS cache: {modis_cache_path}")
        df_modis = pd.read_csv(modis_cache_path, parse_dates=['date'],
                               index_col='date')
        drop_cols = ['NDVI', 'EVI', 'Habitat_Patchiness']
        df_modis = df_modis.drop(
            columns=[c for c in drop_cols if c in df_modis.columns])
        return df_modis

    # ── Strategy 2: fetch in 3-year chunks ────────────────────
    import time
    from datetime import datetime

    start_year = int(start_date[:4])
    end_year   = int(end_date[:4])
    chunk_years = 3
    chunks = []

    for yr in range(start_year, end_year + 1, chunk_years):
        yr_end = min(yr + chunk_years - 1, end_year)
        c_start = f"{yr}-01-01"
        c_end   = f"{yr_end}-12-31"
        print(f"    Fetching MODIS {c_start} → {c_end} ...")

        retries = 3
        for attempt in range(retries):
            try:
                chunk_df = fetch_data_failsafe(roi, c_start, c_end)
                chunks.append(chunk_df)
                print(f"      ✓ {len(chunk_df)} months")
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"      Retry {attempt+1}/{retries} "
                          f"after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"      ✗ Failed after {retries} attempts: {e}")
                    raise

    df_modis = pd.concat(chunks, axis=0)
    df_modis = df_modis[~df_modis.index.duplicated(keep='first')]
    df_modis = df_modis.sort_index()

    drop_cols = ['NDVI', 'EVI', 'Habitat_Patchiness']
    df_modis = df_modis.drop(
        columns=[c for c in drop_cols if c in df_modis.columns])

    print(f"  MODIS climate: {len(df_modis)} months total")
    return df_modis


def merge_hls_modis(df_hls, df_modis):
    """
    Merge pixel-level HLS/S2 vegetation data with spatially-averaged
    MODIS climate data.

    Climate variables are broadcast to all pixels at matching dates
    (since they're coarser resolution anyway).

    Parameters
    ----------
    df_hls : pd.DataFrame
        Per-pixel vegetation data with 'date' and 'pixel_id' columns.
    df_modis : pd.DataFrame
        Monthly climate data with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Merged pixel-level DataFrame.
    """
    if df_hls.empty:
        return df_hls

    # 1. Data cleanup
    if 'EVI' in df_hls.columns:
        df_hls['EVI'] = df_hls['EVI'].clip(-1.0, 1.0)
    if 'NDVI' in df_hls.columns:
        # Filter noise (unmasked clouds/water)
        df_hls = df_hls[(df_hls['NDVI'] >= -0.95) & (df_hls['NDVI'] <= 0.95)]

    # 2. Resample HLS data to monthly to match MODIS cadence
    df_hls['year_month'] = df_hls['date'].dt.to_period('M')
    df_modis_copy = df_modis.copy()
    df_modis_copy['year_month'] = df_modis_copy.index.to_period('M')

    # 3. Aggregate across pixels to create regional monthly features
    veg_cols = ['NDVI', 'EVI', 'RedEdge_NDVI', 'NDMI', 'NDVI_spatial_std']
    veg_cols = [c for c in veg_cols if c in df_hls.columns]

    agg_funcs = {col: 'median' for col in veg_cols}
    if 'NDVI' in df_hls.columns:
        # Add distributional features
        agg_funcs['NDVI'] = [
            'median',
            lambda x: x.quantile(0.10),
            lambda x: x.quantile(0.90),
            'std'
        ]

    df_monthly = df_hls.groupby('year_month').agg(agg_funcs)

    # Flatten multi-index columns
    new_cols = []
    for c in df_monthly.columns:
        if c[0] == 'NDVI' and c[1] != 'median':
            if c[1] == 'std':
                new_cols.append('NDVI_std')
            else:
                # Handle lambdas: lambda_0 -> p10, lambda_1 -> p90
                # Using order to identify them: p10 is first lambda
                new_cols.append('NDVI_p10' if '<lambda_0>' in str(c[1]) or '<lambda>' in str(c[1]) and len(new_cols) == 1 else 'NDVI_p90')
        else:
            new_cols.append(c[0])
    
    # Let's cleanly rename using known indices to be safe
    # If 'NDVI' had ['median', lambda, lambda, 'std']
    # The columns will be:
    # ('NDVI', 'median') -> NDVI
    # ('NDVI', '<lambda_0>') -> NDVI_p10
    # ('NDVI', '<lambda_1>') -> NDVI_p90
    # ('NDVI', 'std') -> NDVI_std
    clean_cols = []
    for c in df_monthly.columns:
        if c[0] == 'NDVI':
            if c[1] == 'median': clean_cols.append('NDVI')
            elif c[1] == 'std': clean_cols.append('NDVI_std')
            elif clean_cols.count('NDVI_p10') == 0: clean_cols.append('NDVI_p10')
            else: clean_cols.append('NDVI_p90')
        else:
            clean_cols.append(c[0])
            
    df_monthly.columns = clean_cols
    df_monthly = df_monthly.reset_index()

    # 4. Merge climate variables
    merged = df_monthly.merge(
        df_modis_copy.reset_index()[
            ['year_month'] + [c for c in df_modis_copy.columns if c != 'year_month']
        ],
        on='year_month',
        how='left',
    )

    # Convert year_month back to datetime
    merged['date'] = merged['year_month'].dt.to_timestamp()
    merged = merged.drop(columns=['year_month'])
    merged = merged.set_index('date')

    # Interpolate missing values
    merged = merged.interpolate(method='linear').ffill().bfill()

    print(f"  Merged dataset: {len(merged)} regional monthly rows")
    return merged


# ── Main entry point ──────────────────────────────────────────

def load_or_fetch_hls(roi, start_date, end_date, cache_path,
                      n_pixels=None, force_refresh=False):
    """
    Load HLS/S2 pixel-level data from CSV cache, or fetch from GEE.

    This is the drop-in replacement for data.load_or_fetch() when
    USE_HLS is True.

    Parameters
    ----------
    roi : ee.Geometry
    start_date, end_date : str
    cache_path : str
        CSV cache file path.
    n_pixels : int, optional
        Number of pixels to sample (default from config).
    force_refresh : bool

    Returns
    -------
    pd.DataFrame
        Pixel-level monthly time series with vegetation + climate columns.
    """
    if not force_refresh and os.path.exists(cache_path):
        print(f"  Loading cached HLS data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
        return df

    n_pixels = n_pixels or PIXEL_SAMPLE_COUNT

    print(f"\n  [HLS] Fetching {n_pixels}-pixel Sentinel-2 time series...")
    pixels = sample_pixels(roi, n=n_pixels)
    df_hls = fetch_sentinel2_pixel_timeseries(
        roi, pixels, start_date, end_date
    )

    if df_hls.empty:
        print("  WARNING: HLS extraction returned no data. "
              "Falling back to MODIS-only pipeline.")
        from .data import fetch_data_failsafe
        df = fetch_data_failsafe(roi, start_date, end_date)
        df.to_csv(cache_path)
        return df

    print(f"\n  [HLS] Fetching MODIS climate variables (LST, GPP, Rain, etc.)...")
    # Try to reuse existing MODIS cache to avoid GEE timeout
    modis_cache = _guess_modis_cache(cache_path)
    if modis_cache:
        print(f"  Found existing MODIS cache: {modis_cache}")
    df_modis = fetch_modis_climate_monthly(roi, start_date, end_date,
                                           modis_cache_path=modis_cache)

    print(f"\n  [HLS] Merging pixel-level vegetation + climate data...")
    df_merged = merge_hls_modis(df_hls, df_modis)

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    df_merged.to_csv(cache_path)
    print(f"  Saved to {cache_path} ({len(df_merged)} rows)")
    return df_merged
