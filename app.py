import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="Satellite/UAV Multispectral Vegetation Index", layout="wide")
st.title("Satellite/ UAV Multispectral Imagery Vegetation Index Calculator")
st.write(
    "Upload available single-band GeoTIFFs (Red, Green, NIR, Red Edge, Blue) and/or an RGB photo. "
    "The app computes indices based on what you provide. "
    "Indices are computed at preview resolution for speed. No CRS required."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Preview settings")
    preview_max = st.slider("Preview max dimension (pixels)", 300, 3000, 300, 100)

    st.subheader("SAVI parameter")
    L = st.slider("SAVI L (soil adjustment)", 0.0, 1.0, 0.5, 0.05)
    st.caption("Typical SAVI uses L = 0.5")

    st.subheader("RGB index options")
    rgb_source_pref = st.radio(
        "RGB indices source (RGBVI, VARI)",
        ["Prefer multispectral (MS bands)", "Prefer RGB photo"],
        index=0
    )
    st.caption("If RGB photo is used, indices can shift with illumination and camera settings.")

# -----------------------------
# Helpers
# -----------------------------
def safe_div(numer, denom):
    numer = numer.astype(np.float32)
    denom = denom.astype(np.float32)
    denom_eps = max(float(np.nanpercentile(np.abs(denom), 95)) * 1e-3, 1e-3)
    out = np.full_like(numer, np.nan, dtype=np.float32)
    m = np.isfinite(denom) & (np.abs(denom) > denom_eps)
    out[m] = numer[m] / denom[m]
    return out

def percentile_limits(arr, lo=2, hi=98):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))

def show_index_map(arr, name, bounded=True):
    if bounded:
        zmin, zmax = -1.0, 1.0
        scale = "RdYlGn"
    else:
        zmin, zmax = percentile_limits(arr, 2, 98)
        scale = "Viridis"

    fig = px.imshow(
        arr,
        zmin=zmin,
        zmax=zmax,
        origin="upper",
        aspect="equal",
        color_continuous_scale=scale,
    )

    fig.update_layout(
        autosize=False,
        height=420,   # <- same height every time
        margin=dict(l=0, r=0, t=0, b=0),  # <- remove top padding
        coloraxis_colorbar=dict(title="") # <- remove colorbar title to keep consistent
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def show_hist(arr, title, bounded=True):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        st.info("No valid pixels.")
        return
    if bounded:
        vals = np.clip(vals, -1, 1)
    fig = px.histogram(vals, nbins=70, title=title)
    fig.update_layout(margin=dict(l=0, r=0, t=35, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def stretch_u8(arr):
    lo, hi = percentile_limits(arr, 2, 98)
    x = (arr - lo) / (hi - lo + 1e-9)
    x = np.clip(np.nan_to_num(x, nan=0.0), 0, 1)
    return (x * 255).astype(np.uint8)

def read_ms_band(uploaded_file, out_shape):
    data = uploaded_file.getvalue()
    mem = MemoryFile(data)
    ds = mem.open()
    arr = ds.read(1, out_shape=out_shape, resampling=Resampling.bilinear).astype(np.float32)
    nod = ds.nodata
    if nod is not None:
        arr = np.where(arr == nod, np.nan, arr)
    ds.close()
    mem.close()
    return arr

def read_rgb_photo(uploaded_file, out_shape_hw):
    img = Image.open(uploaded_file).convert("RGB")
    w, h = out_shape_hw[1], out_shape_hw[0]
    img = img.resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], img

def preview_shape_from_tif(uploaded_tif):
    data = uploaded_tif.getvalue()
    mem = MemoryFile(data)
    ds0 = mem.open()
    h0, w0 = ds0.height, ds0.width
    ds0.close()
    mem.close()
    scale = min(1.0, preview_max / max(h0, w0))
    return (int(h0 * scale), int(w0 * scale))

def preview_shape_from_rgb(uploaded_rgb):
    img = Image.open(uploaded_rgb)
    w0, h0 = img.size
    img.close()
    scale = min(1.0, preview_max / max(h0, w0))
    return (int(h0 * scale), int(w0 * scale))

# -----------------------------
# Index info
# -----------------------------
INDEX_INFO = {
    "EVI": {
        "eq": "EVI = 2.5 * (NIR - R) / (NIR + 6R - 7.5B + 1)",
        "desc": "Enhanced Vegetation Index; reduces NDVI saturation and improves sensitivity under high biomass.",
        "ref": "Huete (1995).",
        "needs": ["NIR", "R", "B"],
    },
    "EVI2": {
        "eq": "EVI2 = 2.5 * (NIR - R) / (NIR + 2.4R + 1)",
        "desc": "Two-band EVI variant that avoids Blue. Recommended alternative when Blue is missing.",
        "ref": "Jiang et al. (2007).",
        "needs": ["NIR", "R"],
    },
    "GNDVI": {
        "eq": "GNDVI = (NIR - G) / (NIR + G)",
        "desc": "Green NDVI; often sensitive to chlorophyll and canopy vigor.",
        "ref": "Gitelson & Merzlyak (1998).",
        "needs": ["NIR", "G"],
    },
    "NDRE": {
        "eq": "NDRE = (NIR - RE) / (NIR + RE)",
        "desc": "Red-edge index often used for chlorophyll and stress; can saturate less than NDVI in dense crops.",
        "ref": "Rodriguez et al. (2006).",
        "needs": ["NIR", "RE"],
    },
    "NDVI": {
        "eq": "NDVI = (NIR - R) / (NIR + R)",
        "desc": "Classic greenness index; higher values usually indicate more live green vegetation.",
        "ref": "Rouse et al. (1973).",
        "needs": ["NIR", "R"],
    },
    "NGRDI": {
        "eq": "NGRDI = (G - R) / (G + R)",
        "desc": "Visible greenness contrast; more sensitive to illumination than NIR-based indices.",
        "ref": "Gitelson et al. (2002).",
        "needs": ["G", "R"],
    },
    "NIRRENDVI": {
        "eq": "NIRRENDVI = ( ((NIR + RE)/2) - R ) / ( ((NIR + RE)/2) + R )",
        "desc": "Hybrid index combining NIR and Red Edge into a composite NIR term, then contrasting with Red.",
        "ref": "Xiang et al. (2019).",
        "needs": ["NIR", "RE", "R"],
    },
    "RENDVI": {
        "eq": "RENDVI = (RE - R) / (RE + R)",
        "desc": "Red-edge normalized difference using Red Edge and Red.",
        "ref": "Sims & Gamon (2002).",
        "needs": ["RE", "R"],
    },
    "RGBVI": {
        "eq": "RGBVI = (G^2 - B*R) / (G^2 + B*R)",
        "desc": "RGB-based vegetation index used with UAV RGB imagery; influenced by illumination and camera settings.",
        "ref": "Bendig et al. (2015).",
        "needs": ["R", "G", "B"],
    },
    "SAVI": {
        "eq": "SAVI = (1 + L) * (NIR - R) / (NIR + R + L),  L ~ 0.5",
        "desc": "Soil-adjusted index to reduce soil background influence vs NDVI, useful in sparse canopies.",
        "ref": "Huete (1988).",
        "needs": ["NIR", "R"],
    },
    "VARI": {
        "eq": "VARI = (G - R) / (G + R - B)",
        "desc": "Visible Atmospherically Resistant Index; RGB-only and aims to reduce illumination sensitivity.",
        "ref": "Gitelson et al. (2002).",
        "needs": ["R", "G", "B"],
    },
}

INDEX_ORDER = ["EVI", "EVI2", "GNDVI", "NDRE", "NDVI", "NGRDI", "NIRRENDVI", "RENDVI", "RGBVI", "SAVI", "VARI"]
BOUNDED = {"NDVI", "NDRE", "GNDVI", "NGRDI", "RENDVI", "NIRRENDVI", "RGBVI", "VARI", "SAVI"}
UNBOUNDED = {"EVI", "EVI2"}

# -----------------------------
# Upload UI
# -----------------------------
st.markdown("## Upload inputs")

c1, c2, c3 = st.columns(3)
with c1:
    up_r = st.file_uploader("MS_R.TIF (Red)", type=["tif", "tiff"], key="ms_r")
    up_g = st.file_uploader("MS_G.TIF (Green)", type=["tif", "tiff"], key="ms_g")
with c2:
    up_nir = st.file_uploader("MS_NIR.TIF (NIR)", type=["tif", "tiff"], key="ms_nir")
    up_re  = st.file_uploader("MS_RE.TIF (Red Edge)", type=["tif", "tiff"], key="ms_re")
with c3:
    up_b  = st.file_uploader("MS_B.TIF (Blue band)", type=["tif", "tiff"], key="ms_b")
    up_rgb = st.file_uploader("RGB photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="rgb")

# Require at least something to start
if all(u is None for u in [up_r, up_g, up_nir, up_re, up_b, up_rgb]):
    st.info("Upload at least one band GeoTIFF and/or an RGB photo to begin.")
    st.stop()

# Determine preview shape from first available source
first_tif = next((u for u in [up_r, up_g, up_nir, up_re, up_b] if u is not None), None)
if first_tif is not None:
    out_shape = preview_shape_from_tif(first_tif)
else:
    out_shape = preview_shape_from_rgb(up_rgb)

# Read multispectral bands if present
R_ms  = read_ms_band(up_r,  out_shape) if up_r  is not None else None
G_ms  = read_ms_band(up_g,  out_shape) if up_g  is not None else None
NIR   = read_ms_band(up_nir, out_shape) if up_nir is not None else None
RE    = read_ms_band(up_re,  out_shape) if up_re  is not None else None
B_ms  = read_ms_band(up_b,   out_shape) if up_b   is not None else None

# Read RGB photo if present
R_rgb = G_rgb = B_rgb = None
rgb_img = None
if up_rgb is not None:
    R_rgb, G_rgb, B_rgb, rgb_img = read_rgb_photo(up_rgb, out_shape)

# -----------------------------
# Previews
# -----------------------------
st.markdown("## Preview images")
left, right = st.columns(2, vertical_alignment="top")

with left:
    st.subheader("RGB photo (reference)")
    if rgb_img is not None:
        st.image(rgb_img, use_container_width=True)
    else:
        st.info("No RGB photo uploaded.")

with right:
    st.subheader("False color composite (NIR-R-G)")
    if (NIR is not None) and (R_ms is not None) and (G_ms is not None):
        fc = np.dstack([stretch_u8(NIR), stretch_u8(R_ms), stretch_u8(G_ms)])
        st.image(fc, caption="False color: NIR as Red, Red as Green, Green as Blue", use_container_width=True)
    else:
        st.info("Upload NIR, Red, and Green GeoTIFFs to see false color preview.")

# -----------------------------
# Compute indices (only when possible) + track source
# -----------------------------
indices = {k: None for k in INDEX_ORDER}
source = {k: "" for k in INDEX_ORDER}

def set_idx(name, arr, src):
    indices[name] = arr
    source[name] = src

# Multispectral-driven indices
if (NIR is not None) and (R_ms is not None):
    set_idx("NDVI", safe_div((NIR - R_ms), (NIR + R_ms)), "MS")
    set_idx("SAVI", (1.0 + float(L)) * safe_div((NIR - R_ms), (NIR + R_ms + float(L))), "MS")
    set_idx("EVI2", 2.5 * safe_div((NIR - R_ms), (NIR + 2.4 * R_ms + 1.0)), "MS")

if (NIR is not None) and (RE is not None):
    set_idx("NDRE", safe_div((NIR - RE), (NIR + RE)), "MS")

if (NIR is not None) and (G_ms is not None):
    set_idx("GNDVI", safe_div((NIR - G_ms), (NIR + G_ms)), "MS")

if (G_ms is not None) and (R_ms is not None):
    set_idx("NGRDI", safe_div((G_ms - R_ms), (G_ms + R_ms)), "MS")

if (RE is not None) and (R_ms is not None):
    set_idx("RENDVI", safe_div((RE - R_ms), (RE + R_ms)), "MS")

if (NIR is not None) and (RE is not None) and (R_ms is not None):
    set_idx("NIRRENDVI", safe_div((((NIR + RE) / 2.0) - R_ms), (((NIR + RE) / 2.0) + R_ms)), "MS")

# EVI (strict MS) if blue exists
if (NIR is not None) and (R_ms is not None) and (B_ms is not None):
    set_idx("EVI", 2.5 * safe_div((NIR - R_ms), (NIR + 6.0 * R_ms - 7.5 * B_ms + 1.0)), "MS")

# RGBVI / VARI based on preference and availability
has_rgb = (R_rgb is not None) and (G_rgb is not None) and (B_rgb is not None)
has_ms_blue = (B_ms is not None) and (R_ms is not None) and (G_ms is not None)
use_ms_for_rgb = (rgb_source_pref == "Prefer multispectral (MS bands)")

if has_ms_blue and (use_ms_for_rgb or not has_rgb):
    set_idx("RGBVI", safe_div((G_ms**2 - B_ms * R_ms), (G_ms**2 + B_ms * R_ms)), "MS (R,G,B)")
    set_idx("VARI",  safe_div((G_ms - R_ms), (G_ms + R_ms - B_ms)), "MS (R,G,B)")
elif has_rgb:
    set_idx("RGBVI", safe_div((G_rgb**2 - B_rgb * R_rgb), (G_rgb**2 + B_rgb * R_rgb)), "RGB photo (R,G,B)")
    set_idx("VARI",  safe_div((G_rgb - R_rgb), (G_rgb + R_rgb - B_rgb)), "RGB photo (R,G,B)")

# If NGRDI missing from MS, allow RGB fallback (optional)
if indices["NGRDI"] is None and has_rgb:
    set_idx("NGRDI", safe_div((G_rgb - R_rgb), (G_rgb + R_rgb)), "RGB photo (R,G)")

# -----------------------------
# Outputs
# -----------------------------
st.markdown("## Outputs (map | histogram | description)")



for name in INDEX_ORDER:
    arr = indices.get(name)

    c1, c2, c3 = st.columns([2.0, 1.8, 1.2], vertical_alignment="top")

    # ----- Map -----
    with c1:
        st.subheader(name)

        # Special handling for EVI when missing MS blue: offer alternatives
        if name == "EVI" and arr is None and (NIR is not None) and (R_ms is not None) and (B_ms is None):
            st.info("Not available: EVI needs MS Blue aligned with multispectral data.")

            if indices["EVI2"] is not None:
                show_evi2_alt = st.checkbox(
                    "Show EVI2 alternative here (recommended when Blue is missing)",
                    value=False,
                    key="show_evi2_alt"
                )
                if show_evi2_alt:
                    show_index_map(indices["EVI2"], "EVI2 (alternative)", bounded=False)

            # Optional approximate EVI using RGB photo blue (ask user)
            if has_rgb:
                do_evi_rgb = st.checkbox(
                    "Compute EVI using RGB photo Blue (approx, use with caution)",
                    value=False,
                    key="do_evi_rgb"
                )
                if do_evi_rgb:
                    evi_rgb = 2.5 * safe_div((NIR - R_ms), (NIR + 6.0 * R_ms - 7.5 * B_rgb + 1.0))
                    show_index_map(evi_rgb, "EVI (RGB Blue approx)", bounded=False)
                    arr = evi_rgb  # allow histogram to show
        elif arr is None:
            st.info("Not available (missing required bands).")
        else:
            bounded = (name in BOUNDED) and (name not in UNBOUNDED)
            title = f"{name} ({source[name]})" if source.get(name) else name
            show_index_map(arr, title, bounded=bounded)

    # ----- Histogram -----
    with c2:
        if arr is None:
            st.empty()
        else:
            bounded = (name in BOUNDED) and (name not in UNBOUNDED)
            show_hist(arr, f"{name} histogram (full image)", bounded=bounded)

    # ----- Description -----
    with c3:
        info = INDEX_INFO[name]
        st.markdown("**Equation**")
        st.code(info["eq"])

        st.markdown("**Description**")
        st.write(info["desc"])
        st.markdown(f"Reference: {info['ref']}")
        st.markdown(f"Required bands: {', '.join(info['needs'])}")

        # Source + cautions
        if name in {"RGBVI", "VARI", "NGRDI"} and "RGB photo" in source.get(name, ""):
            st.warning("Calculated from RGB photo channels. Results can shift with illumination, shadows, and camera settings.")
        elif name in {"RGBVI", "VARI"} and "MS" in source.get(name, ""):
            st.caption("Calculated from multispectral bands (R,G,B).")

        


    st.divider()
