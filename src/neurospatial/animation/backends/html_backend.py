"""Standalone HTML player backend with instant scrubbing."""

from __future__ import annotations

import base64
import html as html_module
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def render_html(
    env: Environment,
    fields: list[NDArray[np.float64]],
    save_path: str | None,
    *,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    dpi: int = 100,
    image_format: str = "png",
    max_html_frames: int = 500,
    title: str = "Spatial Field Animation",
    **kwargs,  # Accept other parameters gracefully
) -> Path:
    """Generate standalone HTML player with instant scrubbing.

    Pre-renders all frames as base64-encoded images embedded in HTML.
    JavaScript provides play/pause/scrub controls with zero latency.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure
    fields : list of arrays
        Fields to animate
    save_path : str or None
        Output path. If None, defaults to 'animation.html'
    fps : int, default=30
        Frames per second for playback
    cmap : str, default="viridis"
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits. If None, computed from all fields.
    frame_labels : list of str, optional
        Frame labels for each frame
    dpi : int, default=100
        Resolution for rendering
    image_format : str, default="png"
        Image format (png or jpeg). PNG is lossless, JPEG is smaller.
    max_html_frames : int, default=500
        Maximum number of frames allowed. Prevents creating huge HTML files
        that crash browsers.
    title : str, default="Spatial Field Animation"
        Animation title
    **kwargs
        Other parameters (accepted gracefully for compatibility)

    Returns
    -------
    save_path : Path
        Path to saved HTML file

    Raises
    ------
    ValueError
        If number of frames exceeds max_html_frames

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(20)]
        path = env.animate_fields(fields, backend="html", save_path="output.html")
        print(f"HTML player saved to {path}")

    Notes
    -----
    The HTML player includes:
    - Play/pause buttons
    - Frame scrubbing slider
    - Speed control (0.25x to 4x)
    - Keyboard shortcuts (space = play/pause, arrows = step)
    - Frame counter and labels

    For large datasets, consider:
    - Reducing DPI to decrease file size
    - Using image_format='jpeg' for lossy compression
    - Subsampling frames before export
    - Using video or napari backend instead

    See Also
    --------
    render_video : Export as video file
    render_napari : Interactive GPU viewer
    """
    from neurospatial.animation.rendering import (
        compute_global_colormap_range,
        render_field_to_image_bytes,
    )

    n_frames = len(fields)

    # Normalize image_format
    image_format = image_format.lower()
    if image_format not in ("png", "jpeg"):
        raise ValueError(f"image_format must be 'png' or 'jpeg', got '{image_format}'")

    # Estimate file size BEFORE rendering
    # JPEG compression effectiveness depends on image size and content:
    # - Large images (DPI >= 100): 5-10x smaller than PNG
    # - Medium images (DPI 75-100): 2-5x smaller
    # - Small images (DPI < 75): PNG may be smaller due to JPEG overhead
    # Use conservative estimate that works across range
    size_factor = 0.05 if image_format == "jpeg" else 0.1
    estimated_mb = n_frames * size_factor * (dpi / 100) ** 2

    # Hard limit check
    if n_frames > max_html_frames:
        raise ValueError(
            f"HTML backend supports max {max_html_frames} frames (got {n_frames}).\n"
            f"Estimated file size: {estimated_mb:.0f} MB\n"
            f"\n"
            f"Options:\n"
            f"  1. Subsample frames:\n"
            f"     fields_subset = fields[::10]  # Every 10th frame\n"
            f"     env.animate_fields(fields_subset, backend='html', ...)\n"
            f"\n"
            f"  2. Use video backend:\n"
            f"     env.animate_fields(fields, backend='video', save_path='output.mp4')\n"
            f"\n"
            f"  3. Use Napari for interactive viewing:\n"
            f"     env.animate_fields(fields, backend='napari')\n"
            f"\n"
            f"  4. Override limit (NOT RECOMMENDED):\n"
            f"     env.animate_fields(fields, backend='html', max_html_frames={n_frames})\n"
        )

    # Warn about large files (with JPEG recommendation)
    if estimated_mb > 50:
        jpeg_hint = (
            ""
            if image_format == "jpeg"
            else "  - Use image_format='jpeg' (typically 2-10x smaller, best for DPI >= 100)\n"
        )
        warnings.warn(
            f"\nHTML export will create a large file (~{estimated_mb:.0f} MB with {image_format.upper()}).\n"
            f"Consider:\n"
            f"  - Reduce DPI: dpi=50 (current: {dpi})\n"
            f"  - Subsample frames: fields[::5]\n"
            f"{jpeg_hint}",
            UserWarning,
            stacklevel=2,
        )

    # Compute global color scale
    vmin_computed, vmax_computed = compute_global_colormap_range(fields, vmin, vmax)
    vmin = vmin if vmin is not None else vmin_computed
    vmax = vmax if vmax is not None else vmax_computed

    # Pre-render all frames to base64
    print(f"Rendering {n_frames} frames to {image_format.upper()}...")
    frames_b64 = []

    for field in tqdm(fields, desc="Encoding frames"):
        image_bytes = render_field_to_image_bytes(
            env, field, cmap, vmin, vmax, dpi, image_format=image_format
        )

        # Convert to base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        frames_b64.append(b64)

    # Generate frame labels
    if frame_labels is None:
        frame_labels = [f"Frame {i + 1}" for i in range(len(fields))]

    # Create HTML
    html = _generate_html_player(
        frames_b64=frames_b64,
        frame_labels=frame_labels,
        fps=fps,
        title=title,
        image_format=image_format,
    )

    # Set default save path if None
    if save_path is None:
        save_path = "animation.html"

    # Write to file
    output_path = Path(save_path)
    output_path.write_text(html)

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"✓ HTML saved to {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def _generate_html_player(
    frames_b64: list[str],
    frame_labels: list[str],
    fps: int,
    title: str,
    image_format: str,
) -> str:
    """Generate HTML with embedded frames and JavaScript controls.

    Parameters
    ----------
    frames_b64 : list of str
        Base64-encoded frame images
    frame_labels : list of str
        Frame labels for display
    fps : int
        Frames per second for playback
    title : str
        Animation title
    image_format : str
        Image format (png or jpeg)

    Returns
    -------
    html : str
        Complete HTML document with embedded player

    Notes
    -----
    The generated HTML includes:
    - Responsive CSS layout
    - Play/pause/prev/next buttons
    - Range slider for frame scrubbing
    - Speed control dropdown
    - Frame counter display
    - Keyboard shortcuts (space, arrows)
    - ARIA labels for accessibility

    Security:
    - Title is HTML-escaped to prevent injection attacks
    - Frame labels are JSON-encoded and safely inserted via textContent
    """
    n_frames = len(frames_b64)

    # Sanitize title to prevent HTML/JS injection
    safe_title = html_module.escape(title)

    # JSON-encode data for JavaScript
    frames_json = json.dumps(frames_b64)
    labels_json = json.dumps(frame_labels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 900px;
        }}
        h1 {{
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #333;
        }}
        #frame {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
        }}
        #controls {{
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .button-row {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #007bff;
            color: white;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #0056b3;
        }}
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        #slider {{
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            border-radius: 3px;
            outline: none;
        }}
        #slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
        }}
        #slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .info {{
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 14px;
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .speed-control label {{
            font-size: 14px;
            color: #666;
        }}
        .speed-control select {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{safe_title}</h1>

        <img id="frame" src="" alt="Animation frame" />

        <div id="controls">
            <div class="button-row">
                <button id="play" aria-label="Play animation">▶ Play</button>
                <button id="pause" aria-label="Pause animation">⏸ Pause</button>
                <button id="prev" aria-label="Previous frame">⏮ Prev</button>
                <button id="next" aria-label="Next frame">⏭ Next</button>
                <div class="speed-control">
                    <label for="speed">Speed:</label>
                    <select id="speed" aria-label="Playback speed">
                        <option value="0.25">0.25x</option>
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
            </div>

            <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0" aria-label="Frame slider" />

            <div class="info">
                <span id="label" aria-live="polite"></span>
                <span id="frame-counter" aria-live="polite"></span>
            </div>
        </div>
    </div>

    <script>
        // Frame data (embedded)
        const frames = {frames_json};
        const labels = {labels_json};
        const baseFPS = {fps};

        // State
        let currentFrame = 0;
        let playing = false;
        let interval = null;
        let speedMultiplier = 1.0;

        // Elements
        const img = document.getElementById('frame');
        const slider = document.getElementById('slider');
        const labelSpan = document.getElementById('label');
        const counterSpan = document.getElementById('frame-counter');
        const playBtn = document.getElementById('play');
        const pauseBtn = document.getElementById('pause');
        const prevBtn = document.getElementById('prev');
        const nextBtn = document.getElementById('next');
        const speedSelect = document.getElementById('speed');

        // Initialize
        function init() {{
            updateFrame(0);
            updateControls();
        }}

        function updateFrame(idx) {{
            if (idx < 0 || idx >= frames.length) return;

            currentFrame = idx;

            // Update image (instant - just changes src)
            // Use dynamic MIME type based on format
            img.src = 'data:image/{image_format};base64,' + frames[idx];

            // Update UI
            slider.value = idx;
            labelSpan.textContent = labels[idx];
            counterSpan.textContent = `${{idx + 1}} / ${{frames.length}}`;
        }}

        function updateControls() {{
            playBtn.disabled = playing;
            pauseBtn.disabled = !playing;
        }}

        function play() {{
            if (playing) return;
            playing = true;
            updateControls();

            const frameDelay = (1000 / baseFPS) / speedMultiplier;

            interval = setInterval(() => {{
                currentFrame = (currentFrame + 1) % frames.length;
                updateFrame(currentFrame);
            }}, frameDelay);
        }}

        function pause() {{
            if (!playing) return;
            playing = false;
            updateControls();
            clearInterval(interval);
        }}

        function stepForward() {{
            pause();
            updateFrame(Math.min(frames.length - 1, currentFrame + 1));
        }}

        function stepBackward() {{
            pause();
            updateFrame(Math.max(0, currentFrame - 1));
        }}

        // Event listeners
        slider.oninput = (e) => {{
            pause();
            updateFrame(parseInt(e.target.value));
        }};

        playBtn.onclick = play;
        pauseBtn.onclick = pause;
        nextBtn.onclick = stepForward;
        prevBtn.onclick = stepBackward;

        speedSelect.onchange = (e) => {{
            speedMultiplier = parseFloat(e.target.value);
            if (playing) {{
                pause();
                play();  // Restart with new speed
            }}
        }};

        // Keyboard shortcuts
        document.onkeydown = (e) => {{
            if (e.key === ' ') {{
                e.preventDefault();
                playing ? pause() : play();
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                stepForward();
            }}
        }};

        // Start
        init();
    </script>
</body>
</html>"""

    return html
