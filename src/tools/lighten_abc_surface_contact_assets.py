"""Convert the ABC surface-contact demo assets to a light paper background.

The paper gallery reuses two source types from
``paper_true_mesh_surface_contact_abc_run_id``:

* MP4 frames for the first row of the gallery figure.
* Plotly/HTML screenshots for the second row.

This utility updates the source assets themselves so future figure generation
and demo reuse stay visually consistent.  Original dark-theme files are copied
to timestamped backup directories before overwrite.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "src/MyDemo/paper_true_mesh_surface_contact_abc_run_id"
ABC_SCREENSHOT_DIR = Path(
    os.environ.get(
        "P2CCCD_ABC_SCREENSHOT_DIR",
        str(CASE_DIR / "abc_screenshots_not_bundled"),
    )
)
DEFAULT_MP4S = [
    CASE_DIR / "paper_true_mesh_surface_contact_abc.mp4",
    CASE_DIR / "collision_zoom_wireframe.mp4",
]
DEFAULT_PREVIEWS = [
    CASE_DIR / "collision_zoom_wireframe_preview.png",
    CASE_DIR / "surface_clean.png",
    CASE_DIR / "surface_preview_three_methods.png",
]
HTML_PATH = CASE_DIR / "collision_zoom_wireframe_interactive.html"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup_file(path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, backup_dir / path.name)


def _lighten_rgb(rgb: np.ndarray, *, plotly: bool) -> np.ndarray:
    out = rgb.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    if plotly:
        if float(np.median(val)) > 170.0:
            return out
        neutral_bg = (val >= 170) & (val < 240) & (sat < 55)
        dark_bg = (val < 120) & (sat < 155)
        mid_grid = (val >= 120) & (val < 188) & (sat < 105)
        white_wire = (sat < 55) & (val >= 185)
        very_dark = (val < 58) & (sat < 190)
        out[neutral_bg] = (239, 243, 247)
        out[dark_bg] = (239, 243, 247)
        out[mid_grid] = (218, 225, 232)
        out[very_dark] = (232, 237, 242)
        out[white_wire] = (88, 99, 112)
    else:
        dark_bg = (val < 118) & (sat < 150)
        mid_bg = (val >= 118) & (val < 190) & (sat < 95)
        dark_lines = (val < 74) & (sat < 190)
        out[dark_bg] = (246, 248, 250)
        out[mid_bg] = (225, 230, 235)
        out[dark_lines] = (208, 214, 220)

    return out


def lighten_png(path: Path, *, plotly: bool, backup_dir: Path | None) -> None:
    if not path.exists():
        return
    if backup_dir is not None:
        _backup_file(path, backup_dir)
    img = Image.open(path).convert("RGB")
    arr = _lighten_rgb(np.asarray(img), plotly=plotly)
    Image.fromarray(arr).save(path)


def lighten_mp4(path: Path, *, backup_dir: Path) -> None:
    if not path.exists():
        return
    _backup_file(path, backup_dir)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    with tempfile.TemporaryDirectory(prefix="p2cccd_light_mp4_") as td:
        tmp_dir = Path(td)
        frame_dir = tmp_dir / "frames"
        frame_dir.mkdir()
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            light = _lighten_rgb(rgb, plotly=False)
            out_bgr = cv2.cvtColor(light, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_dir / f"frame_{idx:06d}.png"), out_bgr)
            idx += 1
        cap.release()
        if idx == 0:
            raise RuntimeError(f"No frames decoded from video: {path}")

        out_mp4 = tmp_dir / "light.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(frame_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)
        shutil.copy2(out_mp4, path)

    print(f"lightened video: {path} ({idx}/{frame_count or idx} frames, {fps:.3f} fps)")


LIGHT_HTML_PATCH = r"""
<!-- P2CCCD_LIGHT_BACKGROUND_PATCH_BEGIN -->
<style>
  html, body { background: #eff3f7 !important; }
  .js-plotly-plot, .plot-container, .svg-container { background: #eff3f7 !important; }
</style>
<script>
(function () {
  function applyP2CCCDLightTheme() {
    document.documentElement.style.background = "#eff3f7";
    document.body.style.background = "#eff3f7";
    const graphs = document.querySelectorAll(".js-plotly-plot");
    graphs.forEach(function (graph) {
      if (!window.Plotly) return;
      window.Plotly.relayout(graph, {
        "paper_bgcolor": "#eff3f7",
        "plot_bgcolor": "#eff3f7",
        "scene.bgcolor": "#eff3f7",
        "scene.xaxis.backgroundcolor": "#eff3f7",
        "scene.yaxis.backgroundcolor": "#eff3f7",
        "scene.zaxis.backgroundcolor": "#eff3f7",
        "scene.xaxis.gridcolor": "#d4d9df",
        "scene.yaxis.gridcolor": "#d4d9df",
        "scene.zaxis.gridcolor": "#d4d9df",
        "scene.xaxis.color": "#46505a",
        "scene.yaxis.color": "#46505a",
        "scene.zaxis.color": "#46505a"
      });
    });
  }
  window.addEventListener("load", function () {
    setTimeout(applyP2CCCDLightTheme, 200);
    setTimeout(applyP2CCCDLightTheme, 1000);
  });
})();
</script>
<!-- P2CCCD_LIGHT_BACKGROUND_PATCH_END -->
"""


def patch_html(path: Path, *, backup_dir: Path) -> None:
    if not path.exists():
        return
    _backup_file(path, backup_dir)
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"\n?<!-- P2CCCD_LIGHT_BACKGROUND_PATCH_BEGIN -->.*?<!-- P2CCCD_LIGHT_BACKGROUND_PATCH_END -->\n?",
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub("\n" + LIGHT_HTML_PATCH + "\n", text)
    elif "</body>" in text:
        text = text.replace("</body>", LIGHT_HTML_PATCH + "\n</body>", 1)
    else:
        text += "\n" + LIGHT_HTML_PATCH + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"patched html light theme: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--skip-abc-screenshots", action="store_true")
    args = parser.parse_args()

    stamp = _timestamp()
    case_backup = CASE_DIR / "old" / f"dark_theme_backup_{stamp}"
    abc_backup = ABC_SCREENSHOT_DIR / f"dark_theme_backup_{stamp}"

    if not args.skip_video:
        for mp4 in DEFAULT_MP4S:
            lighten_mp4(mp4, backup_dir=case_backup)
        for png in DEFAULT_PREVIEWS:
            lighten_png(png, plotly=False, backup_dir=case_backup)

    if not args.skip_html:
        patch_html(HTML_PATH, backup_dir=case_backup)

    if not args.skip_abc_screenshots:
        for name in ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png"]:
            lighten_png(ABC_SCREENSHOT_DIR / name, plotly=True, backup_dir=abc_backup)
        print(f"lightened ABC screenshot sources: {ABC_SCREENSHOT_DIR}")

    print(f"backup: {case_backup}")
    print(f"ABC backup: {abc_backup}")


if __name__ == "__main__":
    main()
