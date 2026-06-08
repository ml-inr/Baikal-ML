"""Interactive UMAP plot generators using Three.js WebGL.

Both 2D and 3D use the same Three.js PointCloud template for consistent performance
(WebGL handles 500k+ points smoothly). CDN dependencies: three.js, OrbitControls, pako.

Functions:
    plot_umap_2d(df, coords_2d, save_path, title)
    plot_umap_3d(df, coords_3d, save_path, title)
"""

import base64
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_COLORS: dict[str, str] = {
    "muatm_2020":        "#1f77b4",
    "nuatm_2020":        "#ff7f0e",
    "nuatm_conv_2020":   "#ff7f0e",
    "nuatm_prompt_2020": "#2ca02c",
    "nue2_2020":         "#d62728",
    "exp_high":          "#9467bd",
    "exp_low":           "#8c564b",
}
_DEFAULT_COLOR = "#aec7e8"

_THREEJS_CDN  = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"
_ORBIT_CDN    = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"
_PAKO_CDN     = "https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _pack_data(coords: np.ndarray, labels: list[str]) -> str:
    """Serialise coords + labels → zlib-compressed base64 JSON."""
    payload = {
        "coords": coords.astype(np.float32).tolist(),
        "labels": labels,
    }
    raw  = json.dumps(payload, separators=(",", ":")).encode()
    comp = zlib.compress(raw, level=6)
    return base64.b64encode(comp).decode()


def _build_html(
    coords: np.ndarray,
    labels: list[str],
    class_colors: dict[str, str],
    title: str,
    is_3d: bool,
) -> str:
    data_b64 = _pack_data(coords, labels)
    unique_classes = list(dict.fromkeys(labels))  # preserve order

    # Build color map JS object
    color_entries = ", ".join(
        f'"{cls}": [{", ".join(f"{v:.4f}" for v in _hex_to_rgb01(class_colors.get(cls, _DEFAULT_COLOR)))}]'
        for cls in unique_classes
    )
    color_map_js = f"{{ {color_entries} }}"

    # Build checkbox HTML
    checkboxes_html = "".join(
        f'<label style="display:block;margin:4px 0;cursor:pointer;">'
        f'<input type="checkbox" id="cb_{i}" checked '
        f'onchange="toggleClass({i})">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{class_colors.get(cls, _DEFAULT_COLOR)};'
        f'border-radius:2px;margin:0 5px 0 4px;vertical-align:middle;"></span>'
        f'{cls}</label>'
        for i, cls in enumerate(unique_classes)
    )

    camera_setup = (
        "const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth/canvas.clientHeight, 0.01, 10000);"
        "camera.position.set(0, 0, 50);"
        if is_3d else
        "const camera = new THREE.OrthographicCamera(-50, 50, 50, -50, 0.01, 10000);"
        "camera.position.set(0, 0, 100);"
    )

    orbit_rotate = (
        "controls.enableRotate = true;"
        if is_3d else
        "controls.enableRotate = false;"
        "controls.enableZoom   = true;"
        "controls.enablePan    = true;"
    )

    coord_access = (
        "pts[i*3]=c[0]; pts[i*3+1]=c[1]; pts[i*3+2]=c[2];"
        if is_3d else
        "pts[i*3]=c[0]; pts[i*3+1]=c[1]; pts[i*3+2]=0;"
    )

    axis_label_js = """
        const axLabels = [
          { text: 'UMAP-1', pos: new THREE.Vector3(maxR*1.1, 0, 0) },
          { text: 'UMAP-2', pos: new THREE.Vector3(0, maxR*1.1, 0) },
        """ + (
        "  { text: 'UMAP-3', pos: new THREE.Vector3(0, 0, maxR*1.1) },"
        if is_3d else ""
    ) + """
        ];
        axLabels.forEach(({text, pos}) => {
          const div = document.createElement('div');
          div.textContent = text;
          div.style.cssText = 'position:absolute;color:#ccc;font:13px monospace;pointer-events:none';
          div.dataset.worldPos = JSON.stringify([pos.x, pos.y, pos.z]);
          document.getElementById('labels').appendChild(div);
          div._worldPos = pos;
        });
    """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin:0; background:#111; overflow:hidden; font-family:monospace; color:#ddd; }}
  #sidebar {{
    position:absolute; top:10px; left:10px; z-index:10;
    background:rgba(30,30,30,.85); padding:10px 14px;
    border-radius:6px; max-height:95vh; overflow-y:auto; min-width:180px;
  }}
  #sidebar h3 {{ margin:0 0 8px; font-size:13px; color:#fff; }}
  #sidebar label {{ font-size:12px; color:#ccc; }}
  #btns {{ margin-top:8px; }}
  #btns button {{
    font-size:11px; padding:3px 8px; margin-right:4px; cursor:pointer;
    background:#333; border:1px solid #555; color:#ccc; border-radius:3px;
  }}
  #title {{
    position:absolute; top:10px; right:14px; z-index:10;
    font-size:14px; color:#aaa;
  }}
  #labels {{ position:absolute; top:0; left:0; pointer-events:none; }}
  canvas {{ display:block; }}
</style>
</head>
<body>
<div id="sidebar">
  <h3>{title}</h3>
  {checkboxes_html}
  <div id="btns">
    <button onclick="setAll(true)">All</button>
    <button onclick="setAll(false)">None</button>
  </div>
</div>
<div id="labels"></div>
<script src="{_PAKO_CDN}"></script>
<script src="{_THREEJS_CDN}"></script>
<script src="{_ORBIT_CDN}"></script>
<script>
const B64 = "{data_b64}";
const COLOR_MAP = {color_map_js};
const CLASS_NAMES = {json.dumps(unique_classes)};

// Decode data
const raw    = pako.inflate(Uint8Array.from(atob(B64), c => c.charCodeAt(0)));
const parsed = JSON.parse(new TextDecoder().decode(raw));
const allCoords = parsed.coords;
const allLabels = parsed.labels;
const N = allLabels.length;

// Group by class
const classData = {{}};
CLASS_NAMES.forEach((cn, ci) => classData[cn] = {{ idx:[], ci }});
for (let i = 0; i < N; i++) classData[allLabels[i]].idx.push(i);

// Three.js scene
const canvas   = document.createElement('canvas');
document.body.appendChild(canvas);
canvas.style.width  = '100vw';
canvas.style.height = '100vh';

const renderer = new THREE.WebGLRenderer({{ canvas, antialias:true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

{camera_setup}

const controls = new THREE.OrbitControls(camera, renderer.domElement);
{orbit_rotate}
controls.update();

// Compute range for axis labels
let maxR = 0;
for (let i = 0; i < N; i++) {{
  const c = allCoords[i];
  for (let d = 0; d < c.length; d++) maxR = Math.max(maxR, Math.abs(c[d]));
}}

// Build one PointCloud per class
const meshes = {{}};
CLASS_NAMES.forEach(cn => {{
  const indices = classData[cn].idx;
  const pts = new Float32Array(indices.length * 3);
  const rgb = COLOR_MAP[cn] || [0.7, 0.7, 0.7];
  indices.forEach((gi, i) => {{
    const c = allCoords[gi];
    {coord_access}
  }});
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  const mat = new THREE.PointsMaterial({{
    color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
    size: 0.35, sizeAttenuation: true,
  }});
  const mesh = new THREE.Points(geo, mat);
  scene.add(mesh);
  meshes[cn] = mesh;
}});

// Axis lines
const axDirs = [
  [maxR*1.05, 0, 0, 0xff4444],
  [0, maxR*1.05, 0, 0x44ff44],
  {("[0, 0, maxR*1.05, 0x4444ff]," if is_3d else "")}
];
axDirs.forEach(([x,y,z,col]) => {{
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0,0,0), new THREE.Vector3(x,y,z)
  ]);
  scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({{color: col, opacity:0.5, transparent:true}})));
}});

{axis_label_js}

window.toggleClass = function(ci) {{
  const cn = CLASS_NAMES[ci];
  const cb = document.getElementById('cb_' + ci);
  meshes[cn].visible = cb.checked;
}};
window.setAll = function(v) {{
  CLASS_NAMES.forEach((cn, ci) => {{
    meshes[cn].visible = v;
    document.getElementById('cb_' + ci).checked = v;
  }});
}};

// Update CSS2D-style axis labels
const labelEls = Array.from(document.getElementById('labels').children);
function updateLabels() {{
  labelEls.forEach(el => {{
    const wp = el._worldPos;
    const v  = wp.clone().project(camera);
    const x  = ( v.x * 0.5 + 0.5) * window.innerWidth;
    const y  = (-v.y * 0.5 + 0.5) * window.innerHeight;
    el.style.left = x + 'px';
    el.style.top  = y + 'px';
  }});
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  updateLabels();
}}
animate();

window.addEventListener('resize', () => {{
  renderer.setSize(window.innerWidth, window.innerHeight);
  {'camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix();' if is_3d else
   'const s=Math.min(window.innerWidth,window.innerHeight)/100; camera.left=-s*window.innerWidth/window.innerHeight; camera.right=s*window.innerWidth/window.innerHeight; camera.top=s; camera.bottom=-s; camera.updateProjectionMatrix();'}
}});
</script>
</body>
</html>"""


def plot_umap_2d(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP 2D",
    class_colors: dict[str, str] | None = None,
) -> None:
    """Save an interactive 2D UMAP plot as a Three.js WebGL HTML file."""
    colors = {**CLASS_COLORS, **(class_colors or {})}
    html = _build_html(
        coords   = coords_2d,
        labels   = df["data_class"].tolist(),
        class_colors = colors,
        title    = title,
        is_3d    = False,
    )
    Path(save_path).write_text(html, encoding="utf-8")


def plot_umap_3d(
    df: pd.DataFrame,
    coords_3d: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP 3D",
    class_colors: dict[str, str] | None = None,
) -> None:
    """Save an interactive 3D UMAP plot as a Three.js WebGL HTML file."""
    colors = {**CLASS_COLORS, **(class_colors or {})}
    html = _build_html(
        coords   = coords_3d,
        labels   = df["data_class"].tolist(),
        class_colors = colors,
        title    = title,
        is_3d    = True,
    )
    Path(save_path).write_text(html, encoding="utf-8")


# ── Plotly interactive HTML plots (self-contained, no CDN) ────────────────────

def plot_umap_2d_html(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP 2D",
    class_colors: dict[str, str] | None = None,
) -> None:
    """Save an interactive 2D UMAP plot as self-contained HTML (Plotly Scattergl/WebGL)."""
    import plotly.graph_objects as go

    colors = {**CLASS_COLORS, **(class_colors or {})}
    fig = go.Figure()
    for cls in df["data_class"].unique():
        mask = (df["data_class"] == cls).values
        fig.add_trace(go.Scattergl(
            x=coords_2d[mask, 0].tolist(),
            y=coords_2d[mask, 1].tolist(),
            mode="markers",
            marker=dict(size=3, color=colors.get(cls, _DEFAULT_COLOR), opacity=0.6),
            name=f"{cls} ({mask.sum():,})",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        plot_bgcolor="#111",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#ddd"),
        legend=dict(itemsizing="constant"),
        yaxis=dict(scaleanchor="x"),
    )
    fig.write_html(str(save_path), include_plotlyjs=True)


def plot_umap_3d_html(
    df: pd.DataFrame,
    coords_3d: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP 3D",
    class_colors: dict[str, str] | None = None,
) -> None:
    """Save an interactive 3D UMAP plot as self-contained HTML (Plotly Scatter3d/WebGL)."""
    import plotly.graph_objects as go

    colors = {**CLASS_COLORS, **(class_colors or {})}
    fig = go.Figure()
    for cls in df["data_class"].unique():
        mask = (df["data_class"] == cls).values
        fig.add_trace(go.Scatter3d(
            x=coords_3d[mask, 0].tolist(),
            y=coords_3d[mask, 1].tolist(),
            z=coords_3d[mask, 2].tolist(),
            mode="markers",
            marker=dict(size=2, color=colors.get(cls, _DEFAULT_COLOR), opacity=0.6),
            name=f"{cls} ({mask.sum():,})",
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
            bgcolor="#111",
        ),
        paper_bgcolor="#1a1a1a",
        font=dict(color="#ddd"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(save_path), include_plotlyjs=True)


# ── Matplotlib PNG plots ───────────────────────────────────────────────────────

def plot_umap_2d_png(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP 2D",
    class_colors: dict[str, str] | None = None,
) -> None:
    """Save a static 2D UMAP scatter plot as PNG (matplotlib, no CDN deps)."""
    import matplotlib.pyplot as plt

    colors = {**CLASS_COLORS, **(class_colors or {})}
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in df["data_class"].unique():
        mask = (df["data_class"] == cls).values
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=colors.get(cls, _DEFAULT_COLOR),
            s=3, alpha=0.5, linewidths=0,
            label=f"{cls} ({mask.sum():,})",
            rasterized=True,
        )
    ax.legend(markerscale=4, fontsize=9, loc="best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_3d_png(
    df: pd.DataFrame,
    coords_3d: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP 3D",
    class_colors: dict[str, str] | None = None,
) -> None:
    """Save a static 3D UMAP scatter plot as PNG (matplotlib, no CDN deps)."""
    import matplotlib.pyplot as plt

    colors = {**CLASS_COLORS, **(class_colors or {})}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for cls in df["data_class"].unique():
        mask = (df["data_class"] == cls).values
        ax.scatter(
            coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
            c=colors.get(cls, _DEFAULT_COLOR),
            s=2, alpha=0.4, linewidths=0,
            label=f"{cls} ({mask.sum():,})",
            depthshade=False,
        )
    ax.legend(markerscale=4, fontsize=9, loc="best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
