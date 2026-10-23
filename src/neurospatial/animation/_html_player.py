"""Standalone HTML players for pre-rendered animation frames.

Frames are complete images; overlays are drawn into them before encoding.
The player supplies playback controls only: play/pause, frame stepping, a
scrubbing slider, a speed menu and keyboard shortcuts.
"""

from __future__ import annotations

import base64
import html as html_module
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray


def encode_frame(frame: NDArray[np.uint8], image_format: str) -> bytes:
    """Encode an RGB frame as PNG (lossless) or JPEG (quality 85).

    Parameters
    ----------
    frame : ndarray of shape (height, width, 3), dtype uint8
        RGB image.
    image_format : {"png", "jpeg"}
        Encoded image format.

    Returns
    -------
    bytes
        The encoded image file.
    """
    from PIL import Image

    buffer = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(frame))
    if image_format == "jpeg":
        image.save(buffer, format="JPEG", quality=85, optimize=True)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_frame_b64(frame: NDArray[np.uint8], image_format: str) -> str:
    """Base64 text of an encoded frame, for an embedded player."""
    return base64.b64encode(encode_frame(frame, image_format)).decode("ascii")


def frame_filename(index: int, image_format: str, *, digits: int = 5) -> str:
    """Name of frame ``index`` in an external frames directory."""
    return f"frame_{index:0{digits}d}.{image_format}"


def _generate_html_player(
    frames_b64: list[str],
    frame_labels: list[str],
    fps: float,
    title: str,
    image_format: str,
) -> str:
    """Generate a self-contained player with base64-embedded frames.

    Parameters
    ----------
    frames_b64 : list of str
        Base64-encoded frame images.
    frame_labels : list of str
        Label shown with each frame.
    fps : float
        Playback frames per second at 1x speed.
    title : str
        Page title; HTML-escaped.
    image_format : {"png", "jpeg"}
        Encoding of ``frames_b64``.

    Returns
    -------
    str
        Complete HTML document.
    """
    source = f"""        // Frame data (embedded)
        const frames = {json.dumps(frames_b64)};
        function frameSrc(i) {{
            return 'data:image/{image_format};base64,' + frames[i];
        }}
        function prefetchFrames(centerIdx) {{}}
"""
    return _player_document(len(frames_b64), frame_labels, fps, title, source)


def _generate_non_embedded_html_player(
    frames_dir: Path,
    n_frames: int,
    frame_labels: list[str],
    fps: float,
    title: str,
    image_format: str,
    *,
    output_path: Path | None = None,
    digits: int = 5,
) -> str:
    """Generate a player that loads frames from a directory on demand.

    Parameters
    ----------
    frames_dir : Path
        Directory containing ``frame_00000.<image_format>``, ... files.
    n_frames : int
        Number of frames.
    frame_labels : list of str
        Label shown with each frame.
    fps : float
        Playback frames per second at 1x speed.
    title : str
        Page title; HTML-escaped.
    image_format : {"png", "jpeg"}
        Frame file extension.
    output_path : Path, optional
        HTML file used to resolve the frame directory's relative URL. Omit
        only for a frame directory beside the HTML file.
    digits : int, default=5
        Zero-padded width of the frame numbers in the file names.

    Returns
    -------
    str
        Complete HTML document referencing the frames by relative URL.
    """
    # Resolve URLs from the HTML file, including nested or external directories.
    relative_dir = (
        os.path.relpath(frames_dir.resolve(), output_path.resolve().parent)
        if output_path is not None
        else frames_dir.name
    )
    frames_dir_json = json.dumps(quote(Path(relative_dir).as_posix(), safe="/"))
    source = f"""        // Frame data (disk-backed)
        const framesDir = {frames_dir_json};
        function zpad(i, digits={digits}) {{ return (""+i).padStart(digits,"0"); }}
        const frameUrls = Array.from({{length: N}}, (_,i) => `${{framesDir}}/frame_${{zpad(i)}}.{image_format}`);
        function frameSrc(i) {{ return frameUrls[i]; }}

        // Preload +/-5 frames for smoother scrubbing
        const prefetchSize = 5;
        const imageCache = new Map();
        function prefetchFrames(centerIdx) {{
            for (let offset = -prefetchSize; offset <= prefetchSize; offset++) {{
                const idx = centerIdx + offset;
                if (idx >= 0 && idx < N && !imageCache.has(idx)) {{
                    const image = new Image();
                    image.src = frameUrls[idx];
                    imageCache.set(idx, image);
                }}
            }}
            if (imageCache.size > prefetchSize * 4) {{
                for (const [key] of imageCache) {{
                    if (Math.abs(key - centerIdx) > prefetchSize * 2) {{
                        imageCache.delete(key);
                    }}
                }}
            }}
        }}
"""
    return _player_document(n_frames, frame_labels, fps, title, source)


def _player_document(
    n_frames: int, frame_labels: list[str], fps: float, title: str, source: str
) -> str:
    """Assemble the player page around a script defining ``frameSrc``.

    The title is HTML-escaped and labels are JSON-encoded and inserted via
    ``textContent``, so neither can inject markup.
    """
    safe_title = html_module.escape(title)
    labels_json = json.dumps(frame_labels)
    return f"""<!DOCTYPE html>
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
        .frame-container {{
            position: relative;
            width: 100%;
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
        /* Restore a visible keyboard-focus indicator (the slider clears the
           default outline above; buttons/select have border:none). */
        #slider:focus-visible {{
            outline: 2px solid #0056b3;
            outline-offset: 2px;
            box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.35);
        }}
        button:focus-visible,
        .speed-control select:focus-visible {{
            outline: 2px solid #0056b3;
            outline-offset: 2px;
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

        <div class="frame-container">
            <img id="frame" src="" alt="Animation frame" />
        </div>

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
        const N = {n_frames};
        const labels = {labels_json};
        const baseFPS = {fps};
{source}
        // State
        let currentFrame = 0;
        let playing = false;
        let animationId = null;
        let lastTime = 0;
        let accumulator = 0;
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

        // Toggle screen-reader announcements for frame changes. Disabled during
        // autoplay so every frame is not announced (screen-reader spam);
        // enabled while paused so manual navigation (step/seek) is announced.
        function setAnnouncements(enabled) {{
            const politeness = enabled ? 'polite' : 'off';
            labelSpan.setAttribute('aria-live', politeness);
            counterSpan.setAttribute('aria-live', politeness);
        }}

        // Initialize
        function init() {{
            updateFrame(0);
            updateControls();
        }}

        function updateFrame(idx) {{
            if (idx < 0 || idx >= N) return;

            currentFrame = idx;
            img.src = frameSrc(idx);
            prefetchFrames(idx);

            // Update UI
            slider.value = idx;
            labelSpan.textContent = labels[idx];
            counterSpan.textContent = `${{idx + 1}} / ${{N}}`;
        }}

        function updateControls() {{
            playBtn.disabled = playing;
            pauseBtn.disabled = !playing;
        }}

        // Animation loop using requestAnimationFrame
        function animate(timestamp) {{
            if (!playing) return;

            const delta = timestamp - lastTime;
            lastTime = timestamp;
            accumulator += delta;

            const frameDelay = (1000 / baseFPS) / speedMultiplier;

            while (accumulator >= frameDelay) {{
                currentFrame = (currentFrame + 1) % N;
                updateFrame(currentFrame);
                accumulator -= frameDelay;
            }}

            animationId = requestAnimationFrame(animate);
        }}

        function play() {{
            if (playing) return;
            playing = true;
            setAnnouncements(false);  // silence per-frame announcements while playing
            updateControls();
            lastTime = performance.now();
            accumulator = 0;
            animationId = requestAnimationFrame(animate);
        }}

        function pause() {{
            if (!playing) return;
            playing = false;
            setAnnouncements(true);  // re-enable announcements for manual navigation
            updateControls();
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
        }}

        function stepForward() {{
            pause();
            updateFrame(Math.min(N - 1, currentFrame + 1));
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
            // No need to restart - accumulator handles speed changes gracefully
        }};

        // Keyboard shortcuts
        document.onkeydown = (e) => {{
            // If an interactive control has focus, let it handle the key
            // (e.g. Space activates the focused button, arrows move the slider
            // or change the speed select) instead of hijacking playback.
            if (e.target.closest('button, input, select, textarea, [tabindex]')) {{
                return;
            }}
            if (e.key === ' ') {{
                e.preventDefault();
                playing ? pause() : play();
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                stepForward();
            }} else if (e.key === 'Home') {{
                e.preventDefault();
                pause();
                updateFrame(0);
            }} else if (e.key === 'End') {{
                e.preventDefault();
                pause();
                updateFrame(N - 1);
            }} else if (e.key === ',') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === '.') {{
                e.preventDefault();
                stepForward();
            }}
        }};

        // Start
        init();
    </script>
</body>
</html>"""
