import io
import base64
import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astroalign


# --------------------------------------------------
# 1. FITS HEADER & WCS COORDINATE PARSER
# --------------------------------------------------

def parse_fits_header(fits_bytes_or_path):
    """
    Parses a FITS file or byte stream to extract metadata,
    World Coordinate System (WCS) celestial information (RA/Dec),
    and raw pixel array data.
    """
    if isinstance(fits_bytes_or_path, (bytes, bytearray)):
        hdul = fits.open(io.BytesIO(fits_bytes_or_path))
    else:
        hdul = fits.open(fits_bytes_or_path)

    try:
        header = hdul[0].header
        data = hdul[0].data

        if data is None and len(hdul) > 1:
            header = hdul[1].header
            data = hdul[1].data

        # Parse World Coordinate System
        wcs = WCS(header)
        
        # Calculate central RA/Dec coordinates if WCS is present
        ra_center, dec_center = None, None
        if wcs.has_celestial and data is not None:
            ny, nx = data.shape[-2:]
            ra_center, dec_center = wcs.all_pix2world(nx / 2, ny / 2, 0)
            ra_center, dec_center = float(ra_center), float(dec_center)

        metadata = {
            "target_name": header.get("OBJECT", "Unknown Target"),
            "telescope": header.get("TELESCOP", "SPHEREx"),
            "instrument": header.get("INSTRUME", "Infrared Spectral Channel"),
            "date_obs": header.get("DATE-OBS", header.get("DATE", "N/A")),
            "exposure_time": header.get("EXPTIME", 0.0),
            "ra_center": ra_center,
            "dec_center": dec_center,
            "filter": header.get("FILTER", "IR-Band"),
            "dimensions": list(data.shape) if data is not None else []
        }

        return data, metadata

    finally:
        hdul.close()


# --------------------------------------------------
# 2. ASTROALIGN STAR REGISTRATION ENGINE
# --------------------------------------------------

def align_images(source_array: np.ndarray, target_array: np.ndarray):
    """
    Aligns source_array (Frame B) to target_array (Frame A) using
    astroalign point-source triangulation matching.
    Returns the aligned image array and the transformation footprint.
    """
    source_clean = np.nan_to_num(source_array.astype(np.float64))
    target_clean = np.nan_to_num(target_array.astype(np.float64))

    try:
        aligned_image, footprint = astroalign.register(
            source_clean, 
            target_clean
        )
        return aligned_image, footprint
    except Exception as e:
        raise RuntimeError(f"Star registration failed: {str(e)}")


# --------------------------------------------------
# 3. CELESTIAL TARGET RESOLVER (GET /api/search)
# --------------------------------------------------

def resolve_celestial_target(query_name: str):
    """
    Resolves an astronomical target name (e.g., 'Messier 31', 'Crab Nebula')
    to exact RA and Dec coordinates using Astropy SkyCoord.
    """
    if not query_name or not query_name.strip():
        raise ValueError("Query target name cannot be empty.")

    try:
        coord = SkyCoord.from_name(query_name.strip())
        return {
            "query": query_name.strip(),
            "ra": float(coord.ra.deg),
            "dec": float(coord.dec.deg),
            "formatted": coord.to_string('hmsdms')
        }
    except Exception as e:
        raise RuntimeError(f"Could not resolve celestial target '{query_name}': {str(e)}")


# --------------------------------------------------
# 4. ARRAY & BASE64 SERIALIZERS FOR REST API
# --------------------------------------------------

def numpy_to_base64_png(array: np.ndarray) -> str:
    """
    Normalizes a 2D NumPy float array to uint8 (0-255) and encodes
    it into a Base64 PNG string for direct display in frontend UI canvas.
    """
    arr_clean = np.nan_to_num(array.astype(np.float32))
    
    min_val, max_val = np.min(arr_clean), np.max(arr_clean)
    if max_val > min_val:
        norm_arr = ((arr_clean - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_arr = np.zeros_like(arr_clean, dtype=np.uint8)

    img = Image.fromarray(norm_arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
