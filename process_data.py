import os
import shutil
import zipfile
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import datetime, timezone, timedelta

import eumdac
from satpy import Scene
from pyresample import create_area_def

OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EUMETSAT Data Store collection for Meteosat Second Generation (MSG) SEVIRI
# Level 1.5 rectified image data from the 0° service (full disk, every 15 min).
COLLECTION_ID = 'EO:EUM:DAT:MSG:HRSEVIRI'

MAX_FRAMES = 10  # rolling frame buffer per product

# Geographic extent: [west_lon, east_lon, south_lat, north_lat]
# This MUST match the imageBounds in site/index.html.
# MSG/SEVIRI sits over 0° longitude, so the natural coverage is Europe,
# Africa and the Atlantic. We crop to a Europe + Mediterranean window.
EXTENT = [-25, 45, 30, 72]

# Output grid resolution in degrees. SEVIRI is ~3 km at nadir / ~5 km over
# Europe; a coarser grid keeps the rendered PNGs small and fast to produce.
# Larger values => bigger pixels, smaller files, faster runs.
RESOLUTION_DEG = 0.04


# ---------------------------------------------------------------------------
# Custom colour maps
# ---------------------------------------------------------------------------

def _ir_colormap():
    """NWS-style rainbow IR enhancement.

    Maps brightness temperature (190 K → 310 K):
      Cold cloud tops  (190–220 K) → white / magenta / red / orange
      Moderate clouds  (230–260 K) → orange / green / cyan
      Warm clear sky   (270–310 K) → blue → dark blue → near-black
    """
    return LinearSegmentedColormap.from_list('ir_enhancement', [
        (0.00, '#ffffff'),  # 190 K  – white  (extreme cold tops)
        (0.07, '#dd00dd'),  # 199 K  – magenta
        (0.17, '#ff0000'),  # 210 K  – red
        (0.27, '#ff5500'),  # 222 K  – orange-red
        (0.37, '#ff8800'),  # 233 K  – orange (was amber, removed yellow cast)
        (0.45, '#44cc00'),  # 244 K  – green  (was yellow #ffff00 – removed)
        (0.53, '#00cc00'),  # 254 K  – green
        (0.62, '#00cccc'),  # 264 K  – cyan
        (0.72, '#0066ff'),  # 276 K  – blue
        (0.87, '#001177'),  # 294 K  – dark blue
        (1.00, '#060606'),  # 310 K  – near-black
    ])


def _wv_colormap():
    """Water-vapour enhancement colormap.

    Maps brightness temperature (195 K → 280 K):
      Cold / moist upper troposphere (195–225 K) → deep navy → royal blue
      Moderate moisture              (225–250 K) → medium blue → teal
      Warm / dry troposphere         (250–280 K) → green → orange → red
    """
    return LinearSegmentedColormap.from_list('wv_enhancement', [
        (0.00, '#00003c'),  # 195 K  – deep navy
        (0.18, '#0000cc'),  # 209 K  – royal blue
        (0.35, '#0066ee'),  # 222 K  – medium blue
        (0.50, '#00bbdd'),  # 233 K  – light blue / cyan
        (0.63, '#00bb66'),  # 242 K  – teal-green
        (0.74, '#22cc00'),  # 250 K  – green (was yellow-green #aadd00 – removed)
        (0.84, '#ff8800'),  # 258 K  – orange (was yellow #ffcc00 – removed)
        (0.92, '#ff5500'),  # 265 K  – deep orange
        (1.00, '#cc1100'),  # 280 K  – red-orange (warm / dry)
    ])


# ---------------------------------------------------------------------------
# Frame management
# ---------------------------------------------------------------------------

def shift_frames(product_base):
    """Shift existing frames back one slot to make room for a new _00 frame.

    _00 is always the newest frame; _{MAX_FRAMES-1} is the oldest.
    When the buffer is full the oldest frame is deleted before shifting.
    A legacy single-file (product.png) is migrated to the oldest slot on
    the first call so no historical imagery is lost.
    """
    legacy   = os.path.join(OUTPUT_DIR, f'{product_base}.png')
    frame_00 = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')

    # One-time migration: seed the oldest slot with the pre-frame-buffer image
    if os.path.exists(legacy) and not os.path.exists(frame_00):
        seed = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        os.rename(legacy, seed)
        print(f"  Migrated legacy {product_base}.png → {os.path.basename(seed)}")

    # Count how many frame files currently exist
    n_existing = sum(
        1 for i in range(MAX_FRAMES)
        if os.path.exists(os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png'))
    )

    # Drop the oldest frame only when the buffer is already at capacity
    if n_existing >= MAX_FRAMES:
        oldest = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        if os.path.exists(oldest):
            os.remove(oldest)

    # Shift _08→_09, _07→_08, …, _00→_01
    for i in range(MAX_FRAMES - 2, -1, -1):
        src = os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png')
        dst = os.path.join(OUTPUT_DIR, f'{product_base}_{i + 1:02d}.png')
        if os.path.exists(src):
            os.rename(src, dst)


# ---------------------------------------------------------------------------
# EUMETSAT Data Store access
# ---------------------------------------------------------------------------

def download_latest_seviri():
    """Download the most recent MSG/SEVIRI native (.nat) file.

    Authenticates against the EUMETSAT Data Store with credentials supplied
    via the EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET environment
    variables, searches the HRSEVIRI collection for the latest product, and
    extracts the .nat image file into /tmp.

    Returns the local path to the .nat file, or None on failure.
    """
    consumer_key    = os.environ.get('EUMETSAT_CONSUMER_KEY')
    consumer_secret = os.environ.get('EUMETSAT_CONSUMER_SECRET')
    if not consumer_key or not consumer_secret:
        print("  ERROR: EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET not set.")
        return None

    try:
        token     = eumdac.AccessToken((consumer_key, consumer_secret))
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(COLLECTION_ID)

        # SEVIRI full-disk scans arrive every 15 min; look back a few hours so
        # the job still succeeds if the very latest slot is briefly unavailable.
        now   = datetime.now(timezone.utc)
        products = collection.search(
            dtstart=now - timedelta(hours=3),
            dtend=now,
        )

        product = None
        for p in products:           # search returns newest first
            product = p
            break

        if product is None:
            print("  ERROR: No SEVIRI products found in the last 3 hours.")
            return None

        print(f"  Latest product: {product}")

        # Download the product (a ZIP archive) and extract the .nat image file.
        zip_path = os.path.join('/tmp', f'{product}.zip')
        with product.open() as fsrc, open(zip_path, 'wb') as fdst:
            print(f"  Downloading {fsrc.name}...")
            shutil.copyfileobj(fsrc, fdst)

        extract_dir = '/tmp/seviri'
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        os.remove(zip_path)

        nat_files = glob.glob(os.path.join(extract_dir, '**', '*.nat'), recursive=True)
        if not nat_files:
            print("  ERROR: No .nat file inside the downloaded product.")
            return None

        print(f"  Extracted: {os.path.basename(nat_files[0])}")
        return nat_files[0]

    except Exception as e:
        print(f"  ERROR downloading SEVIRI data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_channels(nat_file):
    """Read a SEVIRI .nat file and resample the needed channels to the Europe grid.

    Returns a dict of 2-D numpy arrays on a regular lat/lon grid plus the scene
    start time:
        {
            'VIS006': reflectance [0-1],
            'VIS008': reflectance [0-1],
            'WV_062': brightness temperature [K],
            'IR_108': brightness temperature [K],
            'start_time': datetime,
        }
    Off-disk / out-of-coverage pixels are NaN. Returns None on failure.
    """
    reflective = ['VIS006', 'VIS008']
    thermal    = ['WV_062', 'IR_108']

    try:
        scn = Scene(reader='seviri_l1b_native', filenames=[nat_file])
        scn.load(reflective, calibration='reflectance')
        scn.load(thermal, calibration='brightness_temperature')

        area = create_area_def(
            'europe',
            {'proj': 'longlat', 'datum': 'WGS84'},
            area_extent=(EXTENT[0], EXTENT[2], EXTENT[1], EXTENT[3]),  # W,S,E,N
            resolution=RESOLUTION_DEG,
            units='degrees',
        )
        local = scn.resample(area, resampler='nearest', radius_of_influence=50000)

        out = {'start_time': scn.start_time}
        # Reflectances come back in percent (0-100); rescale to 0-1.
        for name in reflective:
            out[name] = np.asarray(local[name].values, dtype=np.float32) / 100.0
        for name in thermal:
            out[name] = np.asarray(local[name].values, dtype=np.float32)
        return out

    except Exception as e:
        print(f"  ERROR loading channels: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _save_rgba(rgba, product_base):
    """Write an (H, W, 4) float RGBA array [0-1] as the newest frame PNG.

    The array is north-up (row 0 = north), matching the Leaflet imageBounds,
    so it is written directly with no flip.
    """
    shift_frames(product_base)
    output_path = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')
    plt.imsave(output_path, np.clip(rgba, 0.0, 1.0))
    print(f"  Saved: {output_path}")


def render_single_band(data, product_base, colormap, vmin, vmax, gamma=1.0):
    """Render a single channel through a colormap as a transparent PNG.

    NaN pixels (off-disk / no coverage) become fully transparent.
    gamma < 1 brightens the image (e.g. 0.5 = square-root stretch).
    """
    valid  = np.isfinite(data)
    normed = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    normed = np.where(valid, normed, 0.0)
    if gamma != 1.0:
        normed = np.power(normed, gamma)

    rgba = colormap(normed)              # (H, W, 4) float in [0, 1]
    rgba[..., 3] = valid.astype(np.float32)
    _save_rgba(rgba, product_base)


# ---------------------------------------------------------------------------
# GeoColor composite (day / night)
# ---------------------------------------------------------------------------

def process_geocolor(channels):
    """GeoColor-style RGB composite, switching between day and night.

    Daytime  – pseudo-natural colour from the SEVIRI visible channels
               (VIS006 / VIS008) with an IR cloud-top enhancement.
    Nighttime – IR cloud layer (IR_108) on a transparent background.
    """
    print("\n--- GeoColor RGB Composite ---")
    vis06 = channels['VIS006']
    vis08 = channels['VIS008']
    ir108 = channels['IR_108']

    # Day vs night: at night the visible reflectance collapses towards zero.
    mean_ref = float(np.nanmean(vis06))
    is_daytime = mean_ref > 0.03
    print(f"  VIS006 mean reflectance: {mean_ref:.4f}  →  "
          f"{'DAYTIME' if is_daytime else 'NIGHTTIME'}")

    if is_daytime:
        _render_geocolor_day(vis06, vis08, ir108)
    else:
        _render_geocolor_night(ir108)


def _render_geocolor_day(vis06, vis08, ir108):
    """Pseudo-natural colour composite for daytime.

    SEVIRI has no true blue channel, so natural colour is approximated from the
    0.6 µm (VIS006) and 0.8 µm (VIS008) reflectances. A vegetation index derived
    from the two channels greens up land; cold IR_108 cloud tops are blended
    towards white so storms and anvils stand out.
    """
    valid = np.isfinite(vis06)
    v6 = np.nan_to_num(vis06, nan=0.0)
    v8 = np.nan_to_num(vis08, nan=0.0)

    # NDVI-like greenness: vegetation reflects strongly at 0.8 µm vs 0.6 µm.
    veg = np.clip((v8 - v6) / (v8 + v6 + 1e-6), 0.0, 0.6)
    R = np.clip(v6 * (1.0 - 0.35 * veg), 0.0, 1.0)
    G = np.clip(v6 * (1.0 + 0.45 * veg), 0.0, 1.0)
    B = np.clip(v6 * 0.85, 0.0, 1.0)

    # Gamma correction for natural brightness (square-root stretch).
    gamma = 0.5
    R, G, B = R ** gamma, G ** gamma, B ** gamma

    # Cloud enhancement: pixels colder than ~265 K blend towards bright white.
    bt = np.where(np.isfinite(ir108), ir108, 320.0)
    cloud = np.clip((265.0 - bt) / 60.0, 0.0, 1.0)
    s = 0.85
    R = np.clip(R + cloud * (1.0 - R) * s, 0.0, 1.0)
    G = np.clip(G + cloud * (1.0 - G) * s, 0.0, 1.0)
    B = np.clip(B + cloud * (1.0 - B) * (s + 0.05), 0.0, 1.0)

    A = valid.astype(np.float32)
    _save_rgba(np.dstack([R, G, B, A]), 'geocolor')
    print("  (daytime composite)")


def _render_geocolor_night(ir108):
    """Nighttime GeoColor composite.

    Cloud layer derived from IR_108 (10.8 µm clean window): colder tops render
    as brighter blue-white cloud. Clear sky stays transparent so the dark
    basemap shows through. (SEVIRI has no day/night band, so there is no
    city-lights layer.)
    """
    bt = np.where(np.isfinite(ir108), ir108, 320.0)

    # 275 K → no cloud (transparent); 220 K → deep convection (opaque).
    cloud_opacity = np.clip((275.0 - bt) / 55.0, 0.0, 1.0)

    R = cloud_opacity * 0.80
    G = cloud_opacity * 0.88
    B = cloud_opacity * 1.00
    A = cloud_opacity

    _save_rgba(np.dstack([R, G, B, A]).astype(np.float32), 'geocolor')
    print("  (nighttime composite)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("MSG / SEVIRI (EUMETSAT) Satellite Image Processor")
    print("=" * 48)
    print(f"Extent:     {EXTENT}")
    print(f"Resolution: {RESOLUTION_DEG}° / pixel")
    print(f"Collection: {COLLECTION_ID}")

    nat_file = download_latest_seviri()
    if nat_file is None:
        print("\nERROR: Could not obtain SEVIRI data. Aborting.")
        raise SystemExit(1)

    try:
        channels = load_channels(nat_file)
    finally:
        # Clean up the (large) native file and its extraction directory.
        shutil.rmtree('/tmp/seviri', ignore_errors=True)

    if channels is None:
        print("\nERROR: Could not process SEVIRI channels. Aborting.")
        raise SystemExit(1)

    # VIS006 — Visible (0.6 µm)            reflectance [0.0 – 1.0]
    # 'gray' maps 0→black (clear sky) and 1→white (bright cloud).
    # gamma=0.5 (square-root stretch) matches conventional satellite display.
    print("\n--- Visible (VIS006) ---")
    render_single_band(channels['VIS006'], 'visible',
                       plt.get_cmap('gray'), vmin=0.0, vmax=1.0, gamma=0.5)

    # IR_108 — Clean IR window (10.8 µm)   brightness temp [K]
    # Custom NWS-style rainbow: cold tops → red/orange, warm surface → dark blue.
    print("\n--- Infrared (IR_108) ---")
    render_single_band(channels['IR_108'], 'infrared',
                       _ir_colormap(), vmin=190, vmax=310)

    # WV_062 — Upper-Level Water Vapour (6.2 µm)  brightness temp [K]
    # Custom enhancement: cold/moist → navy/blue, warm/dry → orange/red.
    print("\n--- Water Vapor (WV_062) ---")
    render_single_band(channels['WV_062'], 'water_vapor',
                       _wv_colormap(), vmin=195, vmax=280)

    # GeoColor — natural colour (day) or IR cloud composite (night)
    process_geocolor(channels)

    # Write a plain-text timestamp so the website can show freshness.
    # Prefer the satellite acquisition time over wall-clock time.
    sat_time = channels.get('start_time')
    if isinstance(sat_time, datetime):
        if sat_time.tzinfo is None:
            sat_time = sat_time.replace(tzinfo=timezone.utc)
        stamp = sat_time.astimezone(timezone.utc)
    else:
        stamp = datetime.now(timezone.utc)

    ts_path = os.path.join(OUTPUT_DIR, 'last_updated.txt')
    with open(ts_path, 'w') as f:
        f.write(stamp.strftime('%Y-%m-%d %H:%M UTC'))
    print(f"\nTimestamp written: {ts_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
