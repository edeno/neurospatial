"""Mouse interactions scoped to temporary assisted annotation points.

Left/right positive/negative prompting follows CVAT's interaction design:
https://github.com/cvat-ai/cvat. Pinned workflow references are recorded in
docs/user-guide/video-annotation.md#design-credit-and-references.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np

from neurospatial.annotation.segmentation import _inside_pixels

if TYPE_CHECKING:
    import napari
    from napari.layers import Points
    from numpy.typing import NDArray


class PromptTools:
    """Add inside/outside points without changing tools between clicks.

    ``activate`` selects a prompt layer in pan/zoom mode. Left and right clicks
    add inside and outside points respectively; Alt-click removes a nearby
    point. Drags retain napari's camera interaction. Selecting another layer or
    choosing a native point tool leaves this interaction until the next explicit
    activation. No viewer-wide mouse or keyboard bindings are installed.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        inside: Points,
        outside: Points,
        frame_shape: tuple[int, ...],
        *,
        remove_radius: float = 12.0,
    ) -> None:
        if len(frame_shape) < 2 or min(frame_shape[:2]) <= 0:
            raise ValueError("frame_shape must include a positive height and width.")
        if inside is outside or inside.ndim != 2 or outside.ndim != 2:
            raise ValueError("Use two distinct, two-dimensional prompt point layers.")
        if not np.isfinite(remove_radius) or remove_radius <= 0:
            raise ValueError("remove_radius must be a positive canvas distance.")
        self._viewer = viewer
        self._layers = (inside, outside)
        self._shape = np.asarray(frame_shape[:2])
        self._remove_radius = remove_radius
        # Left-click adds inside points unless the user chooses outside, for
        # mice and trackpads without a convenient right button.
        self.left_click_outside = False
        self._active = False
        self._disposed = False
        self._connections: list[tuple[Any, Any]] = []
        self._cursors: dict[Points, str] = {}
        # IDs belong to rows rather than coordinates, so moving a point does
        # not make Undo target an earlier point at its former position.
        self._next_id = 0
        self._ids: dict[Points, list[int]] = {}
        self._snapshots: dict[Points, NDArray[np.float64]] = {}
        self._history: list[tuple[Points, int]] = []
        self._removing: dict[Points, tuple[int, ...]] = {}
        for layer in self._layers:
            self._ids[layer] = []
            self._snapshots[layer] = np.empty((0, 2))
            self._sync_points(layer)
            layer.mouse_drag_callbacks.append(self._mouse_drag)
            self._connect(layer.events.data, self._data_changed)
            self._connect(layer.events.mode, self._mode_changed)
        self._connect(viewer.layers.selection.events.active, self._selection_changed)
        self._connect(viewer.layers.events.removed, self._layer_removed)

    @property
    def active(self) -> bool:
        """Whether assisted clicks currently own the selected prompt layer."""
        return self._active and not self._disposed

    def _connect(self, signal: Any, callback: Any) -> None:
        signal.connect(callback)
        self._connections.append((signal, callback))

    def activate(self, layer: Points | None = None) -> None:
        """Show both prompt sets and start assisted clicks on ``layer``."""
        if self._disposed:
            return
        if layer is None:
            layer = self._layers[0]
        if layer not in self._layers:
            raise ValueError("Activate an inside or outside prompt layer.")
        self.deactivate()
        for prompt_layer in self._layers:
            if prompt_layer not in self._viewer.layers:
                self._viewer.add_layer(prompt_layer)
            prompt_layer.visible = True
            # Native add/select callbacks would otherwise also handle the click.
            prompt_layer.mode = "pan_zoom"
            self._cursors[prompt_layer] = prompt_layer.cursor
            prompt_layer.cursor = "cross"
        self._viewer.layers.selection.active = layer
        self._active = True

    def deactivate(self) -> None:
        """Stop assisted clicks without changing the selected annotation tool."""
        self._active = False
        for layer, cursor in self._cursors.items():
            if layer.cursor == "cross":
                layer.cursor = cursor
        self._cursors.clear()

    def _selection_changed(self, _event: Any) -> None:
        if self._viewer.layers.selection.active not in self._layers:
            self.deactivate()

    def _mode_changed(self, event: Any) -> None:
        if event.source.mode != "pan_zoom":
            self.deactivate()

    def _layer_removed(self, event: Any) -> None:
        if event.value in self._layers:
            self.deactivate()
            # Deleting a prompt layer discards its points. The data event
            # invalidates any prediction made with them.
            if len(event.value.data):
                event.value.data = np.empty((0, 2))

    def _mouse_drag(self, layer: Points, event: Any) -> Generator[None, None, None]:
        if (
            not self.active
            or self._viewer.layers.selection.active is not layer
            or layer.mode != "pan_zoom"
            or event.button not in (1, 2)
            or any(key in event.modifiers for key in ("Shift", "Control", "Meta"))
        ):
            return
        button = event.button
        remove = "Alt" in event.modifiers
        start = np.asarray(event.pos, dtype=float).copy()
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged |= bool(np.linalg.norm(np.asarray(event.pos) - start) >= 5)
            if not dragged:
                event.handled = True
            yield
        dragged |= bool(np.linalg.norm(np.asarray(event.pos) - start) >= 5)
        if (
            dragged
            or not self.active
            or self._viewer.layers.selection.active is not layer
            or layer.mode != "pan_zoom"
        ):
            return
        if remove:
            self.remove_nearest(event.position, zoom=event.camera_zoom)
        else:
            outside = (button == 2) != self.left_click_outside
            target = self._layers[1 if outside else 0]
            # napari events carry world coordinates, not layer pixel coordinates.
            point = np.asarray(target.world_to_data(event.position), dtype=float)
            height, width = self._shape
            # Data coordinates are (row, column); the bounds check takes (x, y).
            if _inside_pixels(point[None, ::-1], (0, 0, width, height)):
                for prompt_layer in self._layers:
                    prompt_layer.selected_data = set()
                target.add(point)
        event.handled = True

    def remove_nearest(self, position: Any, *, zoom: float = 1.0) -> bool:
        """Remove a point within ``remove_radius`` canvas pixels of world position."""
        if self._disposed or not np.isfinite(zoom) or zoom <= 0:
            return False
        best: tuple[Points, int] | None = None
        distance = self._remove_radius
        position = np.asarray(position, dtype=float)[-2:]
        for layer in self._layers:
            for index, point in enumerate(layer.data):
                world = np.asarray(layer.data_to_world(point))[-2:]
                candidate = float(np.linalg.norm(world - position) * zoom)
                if candidate <= distance:
                    best = layer, index
                    distance = candidate
        if best is None:
            return False
        best[0].remove([best[1]])
        return True

    def remove_selected(self) -> int:
        """Delete selected prompts in either set, returning the number removed."""
        if self._disposed:
            return 0
        count = 0
        for layer in self._layers:
            selected = sorted(
                i for i in layer.selected_data if 0 <= i < len(layer.data)
            )
            count += len(selected)
            layer.remove(selected)
        return count

    def undo_last(self) -> bool:
        """Remove the last added point that is still present, even after edits."""
        if self._disposed:
            return False
        while self._history:
            layer, identity = self._history.pop()
            if identity in self._ids[layer]:
                layer.remove([self._ids[layer].index(identity)])
                return True
        return False

    def _data_changed(self, event: Any) -> None:
        layer = event.source
        action = str(getattr(event, "action", "changed"))
        if action == "removing":
            self._removing[layer] = tuple(event.data_indices)
        elif action in ("added", "removed", "changed"):
            if action == "removed":
                indices = self._removing.pop(layer, ())
                keep = [i for i in range(len(self._ids[layer])) if i not in indices]
                self._ids[layer] = [self._ids[layer][i] for i in keep]
                self._snapshots[layer] = self._snapshots[layer][keep]
            self._sync_points(layer)

    def _sync_points(self, layer: Points) -> None:
        old = self._snapshots[layer]
        data = np.asarray(layer.data, dtype=float)
        old_ids = self._ids[layer]
        identities: list[int | None] = [None] * len(data)
        available = set(range(len(old)))
        # Retain identities through a direct .data reassignment/reordering too.
        for index, point in enumerate(data):
            match = next(
                (i for i in sorted(available) if np.array_equal(old[i], point)), None
            )
            if match is not None:
                identities[index] = old_ids[match]
                available.remove(match)
        if len(data) == len(old) and any(identity is None for identity in identities):
            # A move can land on another point's coordinates. In that case
            # preserve row identities instead of mistaking it for a reorder.
            self._snapshots[layer] = data.copy()
            return
        for index in range(len(data)):
            if identities[index] is None:
                if index in available:
                    identities[index] = old_ids[index]
                    available.remove(index)
                else:
                    self._next_id += 1
                    identities[index] = self._next_id
                    self._history.append((layer, self._next_id))
        self._ids[layer] = [
            int(identity) for identity in identities if identity is not None
        ]
        self._snapshots[layer] = data.copy()

    def dispose(self) -> None:
        """Disconnect only this controller's callbacks; repeated calls are safe."""
        if self._disposed:
            return
        self.deactivate()
        self._disposed = True
        for layer in self._layers:
            if self._mouse_drag in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.remove(self._mouse_drag)
        for signal, callback in self._connections:
            signal.disconnect(callback)
        self._connections.clear()
