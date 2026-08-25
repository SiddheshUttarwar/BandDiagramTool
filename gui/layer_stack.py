"""
LayerStackPanel: a vertical, drag-to-reorder list of layer "cards" mirroring
the device's physical cross-section (top layer's card drawn at the top of
the window, bottom layer's card at the bottom) even though the underlying
DeviceModel.layers list stays in the codebase's bottom->top convention
(index 0 = substrate/bottom, matching devices.device.AlGaNDevice).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional

from devices.layer import AbruptLayer, GradedLayer
from gui.models import DeviceModel

_CARD_BG = "#ffffff"
_CARD_BG_SELECTED = "#e8f0fe"
_CARD_BORDER = "#d0d5dd"
_CARD_BORDER_SELECTED = "#4285f4"
_HANDLE_FG = "#9aa0a6"


def _layer_summary(layer) -> tuple[str, str]:
    """Return (title, subtitle) describing a layer for its card."""
    if isinstance(layer, AbruptLayer):
        title = f"Al{layer.x_Al:.2f}Ga{1 - layer.x_Al:.2f}N" if layer.x_Al > 0 else "GaN"
        bits = [f"{layer.thickness_nm:g} nm"]
        if layer.n_doping > 0:
            bits.append(f"n={layer.n_doping:.1e}")
        if layer.p_doping > 0:
            bits.append(f"p={layer.p_doping:.1e}")
        return title, "  ·  ".join(bits)
    if isinstance(layer, GradedLayer):
        title = f"Graded Al{layer.x_Al_start:.2f}→{layer.x_Al_end:.2f}GaN"
        bits = [f"{layer.thickness_nm:g} nm", layer.profile]
        if layer.n_doping > 0:
            bits.append(f"n={layer.n_doping:.1e}")
        if layer.p_doping > 0:
            bits.append(f"p={layer.p_doping:.1e}")
        return title, "  ·  ".join(bits)
    return "Layer", ""


class LayerStackPanel(ttk.Frame):
    """on_select(index_or_None) fires when a card is clicked or dropped
    after a drag (index is in the model's bottom->top list space)."""

    def __init__(self, parent, model: DeviceModel,
                 on_select: Callable[[Optional[int]], None], **kwargs):
        super().__init__(parent, **kwargs)
        self.model = model
        self.on_select = on_select
        self.selected_index: Optional[int] = None
        self._cards: List[tk.Frame] = []
        self._drag_index: Optional[int] = None

        self._build_header()
        self._build_scroll_area()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_header(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(header, text="Device Stack (bottom → top)",
                  font=("Segoe UI", 10, "bold")).pack(side="left")

        add_btn = ttk.Menubutton(header, text="+ Add Layer")
        menu = tk.Menu(add_btn, tearoff=False)
        menu.add_command(label="Abrupt layer", command=self._add_abrupt)
        menu.add_command(label="Graded layer", command=self._add_graded)
        add_btn["menu"] = menu
        add_btn.pack(side="right")

    def _build_scroll_area(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        self._canvas = tk.Canvas(outer, highlightthickness=0, bg="#f5f6f8")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._inner = ttk.Frame(self._canvas)
        self._inner_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>",
                          lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                           lambda e: self._canvas.itemconfigure(self._inner_id, width=e.width))

        # Only capture the mousewheel while the pointer is actually over
        # this panel, so it doesn't steal scrolling from other widgets.
        self._canvas.bind("<Enter>",
                           lambda e: self._canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    # ------------------------------------------------------------------
    def _add_abrupt(self):
        self.model.add_layer(AbruptLayer(x_Al=0.0, thickness_nm=10.0))
        self.select(len(self.model.layers) - 1)

    def _add_graded(self):
        self.model.add_layer(GradedLayer(x_Al_start=0.0, x_Al_end=0.2, thickness_nm=10.0))
        self.select(len(self.model.layers) - 1)

    # ------------------------------------------------------------------
    def select(self, index: Optional[int]):
        self.selected_index = index
        self.refresh()
        self.on_select(index)

    def refresh(self):
        for w in self._inner.winfo_children():
            w.destroy()
        self._cards = []

        n = len(self.model.layers)
        # Visual order is reversed vs. the bottom->top model list: the top
        # layer (last in the list) is drawn first (at the top of the panel).
        for model_index in range(n - 1, -1, -1):
            self._build_card(model_index)

        if n == 0:
            ttk.Label(self._inner, text="No layers yet — click “+ Add Layer” to start.",
                      foreground="#888888").pack(pady=20)

    def _build_card(self, model_index: int):
        layer = self.model.layers[model_index]
        selected = (model_index == self.selected_index)

        card = tk.Frame(self._inner, bg=_CARD_BG_SELECTED if selected else _CARD_BG,
                         highlightbackground=_CARD_BORDER_SELECTED if selected else _CARD_BORDER,
                         highlightthickness=2, bd=0)
        card.pack(fill="x", padx=6, pady=4)

        handle = tk.Label(card, text="≡", fg=_HANDLE_FG,
                           bg=card["bg"], font=("Segoe UI", 12), cursor="fleur")
        handle.pack(side="left", padx=(8, 4), pady=8)

        title, subtitle = _layer_summary(layer)
        text_frame = tk.Frame(card, bg=card["bg"])
        text_frame.pack(side="left", fill="both", expand=True, pady=6)
        tk.Label(text_frame, text=title, bg=card["bg"], anchor="w",
                 font=("Segoe UI", 10, "bold")).pack(fill="x")
        if subtitle:
            tk.Label(text_frame, text=subtitle, bg=card["bg"], anchor="w",
                     fg="#666666", font=("Segoe UI", 8)).pack(fill="x")

        del_btn = tk.Label(card, text="✕", fg="#b3261e", bg=card["bg"],
                            cursor="hand2", padx=8)
        del_btn.pack(side="right")
        del_btn.bind("<Button-1>", lambda e, i=model_index: self._delete(i))

        clickable = [card, text_frame] + list(text_frame.winfo_children())
        for widget in clickable:
            widget.bind("<Button-1>", lambda e, i=model_index: self.select(i))

        handle.bind("<ButtonPress-1>", lambda e, i=model_index: self._drag_start(i))
        handle.bind("<B1-Motion>", self._drag_motion)
        handle.bind("<ButtonRelease-1>", self._drag_end)

        self._cards.append(card)

    def _delete(self, model_index: int):
        self.model.remove_layer(model_index)
        if self.selected_index == model_index:
            self.selected_index = None
            self.on_select(None)
        elif self.selected_index is not None and self.selected_index > model_index:
            self.selected_index -= 1
        self.refresh()

    # ------------------------------------------------------------------
    # Drag-to-reorder. Visual position and model index are mirror images
    # of each other (visual_pos = n-1-model_index), since the card list is
    # rendered top-of-screen = top-of-device.
    # ------------------------------------------------------------------
    def _drag_start(self, model_index: int):
        self._drag_index = model_index

    def _drag_motion(self, event):
        if self._drag_index is None:
            return
        target_visual_pos = self._card_at_y(event.y_root)
        if target_visual_pos is None:
            return
        n = len(self.model.layers)
        target_index = n - 1 - target_visual_pos
        if target_index == self._drag_index:
            return
        self.model.move_layer(self._drag_index, target_index)
        self._drag_index = target_index
        self.selected_index = target_index
        self.refresh()

    def _drag_end(self, event):
        if self._drag_index is not None:
            self.on_select(self._drag_index)
        self._drag_index = None

    def _card_at_y(self, y_root: int) -> Optional[int]:
        for visual_pos, card in enumerate(self._cards):
            top = card.winfo_rooty()
            bottom = top + card.winfo_height()
            if top <= y_root <= bottom:
                return visual_pos
        return None
