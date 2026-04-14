"""
phase5_review.py — Phase 5: Keyboard-driven photo review UI.

Loads the top-800 photos selected by phase 4, presents them full-screen with
score metadata and person-tag support, accepts keyboard actions, and saves
review progress after every action.  On completion, copies approved photos
and person-tagged photos to the output folder tree.

Usage:
    python phase5_review.py
    python phase5_review.py --input selected_photos_candid.json

Keyboard shortcuts (main review window):
    A / D        Previous / Next photo
    Space        Cycle status: unreviewed → approved → unreviewed
                 (if rejected → goes to approved)
    Backspace    Reject current photo
    G            Toggle Gautam tag
    <letter>     Toggle any configured additional-person tag
    F            Toggle filter sidebar
"""

import argparse
import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

import config
from utils import copy_to_output, setup_logging


# ── UI constants ──────────────────────────────────────────────────────────────

# Colours cycled for additional persons (index 0 = first added person)
_TAG_COLOR_CYCLE = ["#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

_BORDER_APPROVED  = "#00C851"
_BORDER_REJECTED  = "#FF4444"
_CANVAS_BG        = "#2B2B2B"
_RESIZE_DEBOUNCE  = 60    # ms to wait after a canvas resize before redrawing

# Single-letter keys that cannot be assigned to a person tag
_RESERVED_LETTERS = frozenset("ADGF")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _load_top800(input_path: str) -> list:
    """Load selected_photos_candid.json and return in_top800 records sorted by rank.

    Args:
        input_path: Path to the phase 4 output JSON.

    Returns:
        List of record dicts sorted by selection_rank ascending.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    photos = [r for r in records if r.get("in_top800")]
    photos.sort(key=lambda r: r.get("selection_rank", 9999))
    return photos


def _save_progress(path: str, data: dict, logger) -> None:
    """Atomically write review progress JSON to disk via a temp file.

    Writes to a .tmp sidecar first, then os.replace() so a crash mid-write
    never produces a truncated progress file.

    Args:
        path:   Destination path for review_progress.json.
        data:   Full progress dict (version, persons, reviews).
        logger: Logger instance for error reporting.
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        if logger:
            logger.error("Progress save failed: %s", exc)


# ── Startup dialog ────────────────────────────────────────────────────────────

class StartupDialog:
    """Modal person-configuration dialog shown before the review window opens.

    Displays the photo count, a fixed G=Gautam tag, and input fields for
    configuring up to MAX_ADDITIONAL_PERSONS extra person tags.

    Attributes:
        result: List of person config dicts on success, or None if cancelled.
    """

    def __init__(self, root: tk.Tk, photo_count: int, logger) -> None:
        """Build and run the modal dialog, blocking until the user clicks Start.

        Args:
            root:        Parent Tk root window.
            photo_count: Total photos to review (displayed in header).
            logger:      Logger instance.
        """
        self.result  = None
        self._logger = logger
        self._persons = [
            {
                "letter": "G",
                "name":   "gautam",
                "color":  config.PERSON_TAG_COLORS.get("gautam", "#4A90D9"),
            }
        ]

        self._dlg = tk.Toplevel(root)
        self._dlg.title("Review Setup")
        self._dlg.resizable(False, False)
        self._dlg.transient(root)
        self._dlg.grab_set()
        self._dlg.protocol("WM_DELETE_WINDOW", self._dlg.destroy)

        self._build(photo_count)

        # Centre on screen
        self._dlg.update_idletasks()
        w  = self._dlg.winfo_reqwidth()
        h  = self._dlg.winfo_reqheight()
        sx = (self._dlg.winfo_screenwidth()  - w) // 2
        sy = (self._dlg.winfo_screenheight() - h) // 2
        self._dlg.geometry(f"+{sx}+{sy}")

        root.wait_window(self._dlg)

    # ── Builder ───────────────────────────────────────────────────────────────

    def _build(self, photo_count: int) -> None:
        """Construct all widgets in the dialog."""
        outer_pad = {"padx": 16, "pady": 6}

        # Header
        ttk.Label(
            self._dlg,
            text=f"{photo_count} photos to review",
            font=("Segoe UI", 12, "bold"),
        ).pack(**outer_pad)

        # ── Person tags section ───────────────────────────────────────────────
        section = ttk.LabelFrame(self._dlg, text=" Person Tags ", padding=10)
        section.pack(fill=tk.X, **outer_pad)

        # Fixed G = Gautam row (not editable)
        fixed_row = ttk.Frame(section)
        fixed_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            fixed_row, text="G", font=("Segoe UI", 10, "bold"), width=3
        ).pack(side=tk.LEFT)
        ttk.Label(
            fixed_row, text="→  Gautam  (reserved)",
            foreground="#888888",
        ).pack(side=tk.LEFT)

        ttk.Separator(section, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(section, text="Add another person?").pack(anchor="w")

        # Add-person input row
        add_row = ttk.Frame(section)
        add_row.pack(fill=tk.X, pady=4)

        self._letter_var = tk.StringVar()
        self._name_var   = tk.StringVar()

        ttk.Label(add_row, text="Letter:").pack(side=tk.LEFT)
        letter_ent = ttk.Entry(add_row, textvariable=self._letter_var, width=3)
        letter_ent.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(add_row, text="Name:").pack(side=tk.LEFT)
        ttk.Entry(add_row, textvariable=self._name_var, width=18).pack(
            side=tk.LEFT, padx=(4, 12)
        )

        self._add_btn = ttk.Button(add_row, text="Add", command=self._add_person)
        self._add_btn.pack(side=tk.LEFT)

        # Limit letter field to one character
        def _limit_one_char(*_):
            v = self._letter_var.get()
            if len(v) > 1:
                self._letter_var.set(v[-1])

        self._letter_var.trace_add("write", _limit_one_char)

        # Configured-persons list (rebuilt on every add/remove)
        self._list_frame = ttk.Frame(section)
        self._list_frame.pack(fill=tk.X, pady=(8, 0))
        self._refresh_person_list()

        # ── Start button ──────────────────────────────────────────────────────
        ttk.Button(
            self._dlg,
            text="Start Review",
            command=self._start,
            padding=(20, 6),
        ).pack(pady=(6, 16))

    # ── Person list helpers ───────────────────────────────────────────────────

    def _refresh_person_list(self) -> None:
        """Rebuild the displayed list of configured person tags."""
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        for i, person in enumerate(self._persons):
            row = ttk.Frame(self._list_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(
                row,
                text=f"  {person['letter'].upper()}  —  {person['name']}",
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT)
            if i > 0:   # Gautam (index 0) cannot be removed
                idx = i
                ttk.Button(
                    row,
                    text="Remove",
                    width=7,
                    command=lambda i=idx: self._remove_person(i),
                ).pack(side=tk.RIGHT, padx=4)

        extra = len(self._persons) - 1   # Gautam does not count toward the limit
        at_limit = extra >= config.MAX_ADDITIONAL_PERSONS
        self._add_btn.config(state=tk.DISABLED if at_limit else tk.NORMAL)

    def _add_person(self) -> None:
        """Validate input and append a new person to the configured list."""
        letter = self._letter_var.get().strip().upper()
        name   = self._name_var.get().strip().lower()

        if not letter or len(letter) != 1 or not letter.isalpha():
            messagebox.showerror(
                "Invalid Letter", "Enter a single letter (A–Z).", parent=self._dlg
            )
            return
        if letter in _RESERVED_LETTERS:
            messagebox.showerror(
                "Reserved Key",
                f"'{letter}' is reserved (A / D / G / F are navigation keys).",
                parent=self._dlg,
            )
            return
        existing = {p["letter"].upper() for p in self._persons}
        if letter in existing:
            messagebox.showerror(
                "Duplicate", f"'{letter}' is already assigned.", parent=self._dlg
            )
            return
        if not name:
            messagebox.showerror("Missing Name", "Enter a name.", parent=self._dlg)
            return

        extra_index = len(self._persons) - 1   # 0-based index into colour cycle
        color = _TAG_COLOR_CYCLE[extra_index % len(_TAG_COLOR_CYCLE)]
        self._persons.append({"letter": letter, "name": name, "color": color})
        self._letter_var.set("")
        self._name_var.set("")
        self._refresh_person_list()

    def _remove_person(self, idx: int) -> None:
        """Remove the person at idx (Gautam at index 0 is protected).

        Args:
            idx: Index into self._persons to remove (must be > 0).
        """
        if idx <= 0:
            return
        self._persons.pop(idx)
        self._refresh_person_list()

    def _start(self) -> None:
        """Store the configured person list and close the dialog."""
        self.result = self._persons
        self._dlg.destroy()


# ── Main review application ───────────────────────────────────────────────────

class ReviewApp:
    """Keyboard-driven photo review UI.

    Three-column layout:
        Column 0 — photo canvas (weight 7, shrinks when filter shown)
        Column 1 — filter sidebar (weight 2, hidden by default)
        Column 2 — info / navigation panel (fixed ~280 px)

    Images are loaded in background threads; the current PIL image is cached
    so window resize re-renders without re-reading from disk.
    """

    def __init__(
        self,
        root: tk.Tk,
        photos: list,
        persons: list,
        progress: dict | None,
        logger,
    ) -> None:
        """Initialise state and build the full UI.

        Args:
            root:     Tk root window (already realised).
            photos:   Sorted list of in_top800 photo records.
            persons:  Person config dicts from StartupDialog (or loaded from progress).
            progress: Loaded review_progress.json, or None for a fresh session.
            logger:   Logger instance.
        """
        self.root    = root
        self.photos  = photos
        self.persons = persons
        self.logger  = logger

        # Review state: path → {"status": ..., "person_tags": [...]}
        if progress and "reviews" in progress:
            self.reviews: dict = progress["reviews"]
        else:
            self.reviews = {}

        # Active (filtered) photo list and cursor
        self.active_photos: list = list(photos)
        self.active_idx: int     = 0

        # Image loading state
        self._pil_cache: dict     = {}          # path → PIL.Image (raw, unresized)
        self._current_pil         = None        # PIL Image for current photo
        self._current_tk          = None        # ImageTk ref (prevents GC)
        self._load_cancel         = threading.Event()
        self._resize_after_id     = None        # debounce handle

        self._filter_visible = False

        self._build_layout()
        self._bind_keys()
        self._show_photo(0)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Create the three-column main grid inside the root window."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._main = ttk.Frame(self.root)
        self._main.grid(row=0, column=0, sticky="nsew")
        self._main.columnconfigure(0, weight=7, minsize=400)
        self._main.columnconfigure(1, weight=0, minsize=0)   # filter (hidden)
        self._main.columnconfigure(2, weight=0, minsize=280)
        self._main.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_filter_panel()
        self._build_right_panel()

    def _build_left_panel(self) -> None:
        """Build the photo canvas that fills column 0."""
        self._left = ttk.Frame(self._main)
        self._left.grid(row=0, column=0, sticky="nsew")
        self._left.rowconfigure(0, weight=1)
        self._left.columnconfigure(0, weight=1)

        self.photo_canvas = tk.Canvas(
            self._left,
            bg=_CANVAS_BG,
            highlightthickness=0,
            cursor="arrow",
        )
        self.photo_canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.photo_canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_right_panel(self) -> None:
        """Build the info / navigation panel in column 2."""
        self._right = ttk.Frame(self._main, padding=10)
        self._right.grid(row=0, column=2, sticky="nsew")
        self._right.columnconfigure(0, weight=1)

        r = 0

        # Filename (wraps if long)
        self._lbl_filename = ttk.Label(
            self._right, text="", font=("Segoe UI", 10, "bold"), wraplength=260
        )
        self._lbl_filename.grid(row=r, column=0, sticky="w")
        r += 1

        # Shot type + moment label
        self._lbl_meta = ttk.Label(self._right, text="", foreground="#666666")
        self._lbl_meta.grid(row=r, column=0, sticky="w", pady=(0, 4))
        r += 1

        ttk.Separator(self._right, orient="horizontal").grid(
            row=r, column=0, sticky="ew", pady=4)
        r += 1

        # Score rows
        for label_text, attr_name in [
            ("NIMA",     "_lbl_nima"),
            ("Emotion",  "_lbl_emotion"),
            ("Memorab.", "_lbl_memorab"),
        ]:
            row_frame = ttk.Frame(self._right)
            row_frame.grid(row=r, column=0, sticky="ew")
            r += 1
            ttk.Label(row_frame, text=f"{label_text}:", width=10, anchor="w").pack(
                side=tk.LEFT
            )
            lbl = ttk.Label(row_frame, text="—", anchor="e")
            lbl.pack(side=tk.RIGHT)
            setattr(self, attr_name, lbl)

        # Final score (bold)
        final_row = ttk.Frame(self._right)
        final_row.grid(row=r, column=0, sticky="ew")
        r += 1
        ttk.Label(
            final_row, text="Final:", width=10, anchor="w",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT)
        self._lbl_final = ttk.Label(
            final_row, text="—", anchor="e", font=("Segoe UI", 9, "bold")
        )
        self._lbl_final.pack(side=tk.RIGHT)

        # Modifiers line
        self._lbl_mods = ttk.Label(
            self._right, text="", foreground="#888888",
            font=("Segoe UI", 8), wraplength=260,
        )
        self._lbl_mods.grid(row=r, column=0, sticky="w", pady=(2, 0))
        r += 1

        ttk.Separator(self._right, orient="horizontal").grid(
            row=r, column=0, sticky="ew", pady=6)
        r += 1

        # Status indicator
        self._lbl_status = ttk.Label(
            self._right, text="○ unreviewed", font=("Segoe UI", 10, "bold")
        )
        self._lbl_status.grid(row=r, column=0, sticky="w")
        r += 1

        # Person tag display
        self._lbl_tags = ttk.Label(
            self._right, text="Tags: (none)", foreground="#555555"
        )
        self._lbl_tags.grid(row=r, column=0, sticky="w", pady=(2, 0))
        r += 1

        ttk.Separator(self._right, orient="horizontal").grid(
            row=r, column=0, sticky="ew", pady=6)
        r += 1

        # Photo counter
        self._lbl_nav = ttk.Label(
            self._right, text="Photo 1 / 800", font=("Segoe UI", 9)
        )
        self._lbl_nav.grid(row=r, column=0, sticky="w")
        r += 1

        # Approval progress bar
        self._progress_var = tk.IntVar(value=0)
        self._progressbar = ttk.Progressbar(
            self._right,
            variable=self._progress_var,
            maximum=len(self.photos),
            length=240,
        )
        self._progressbar.grid(row=r, column=0, sticky="ew", pady=(4, 2))
        r += 1

        # Counts (approved / rejected / unreviewed)
        self._lbl_counts = ttk.Label(
            self._right, text="✓0  ✗0  ?800", font=("Segoe UI", 9)
        )
        self._lbl_counts.grid(row=r, column=0, sticky="w")
        r += 1

        ttk.Separator(self._right, orient="horizontal").grid(
            row=r, column=0, sticky="ew", pady=6)
        r += 1

        # Prev / Next buttons
        nav_row = ttk.Frame(self._right)
        nav_row.grid(row=r, column=0, sticky="ew")
        r += 1
        ttk.Button(nav_row, text="◀ Prev", command=lambda: self._navigate(-1)).pack(
            side=tk.LEFT
        )
        ttk.Button(nav_row, text="Next ▶", command=lambda: self._navigate(+1)).pack(
            side=tk.RIGHT
        )

        # Done button
        ttk.Button(
            self._right, text="Done", command=self._show_completion_dialog
        ).grid(row=r, column=0, sticky="ew", pady=(8, 0))
        r += 1

        # Keyboard hint
        ttk.Label(
            self._right,
            text=(
                "A/D  navigate    Space  approve\n"
                "Backspace  reject    G  tag Gautam\n"
                "F  toggle filters"
            ),
            foreground="#999999",
            font=("Segoe UI", 7),
            justify=tk.LEFT,
        ).grid(row=r, column=0, sticky="w", pady=(10, 0))

    def _build_filter_panel(self) -> None:
        """Build the filter sidebar (column 1).  Hidden by default."""
        self._filter = ttk.LabelFrame(self._main, text=" Filters ", padding=8)
        # Not gridded here — shown/hidden by _toggle_filter_panel

        f = self._filter

        # ── Shot type ─────────────────────────────────────────────────────────
        ttk.Label(f, text="Shot type", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._filter_shot_vars: dict[str, tk.BooleanVar] = {}
        for shot in ("solo_portrait", "group", "couple", "candid"):
            var = tk.BooleanVar(value=True)
            self._filter_shot_vars[shot] = var
            ttk.Checkbutton(f, text=shot, variable=var).pack(anchor="w")

        ttk.Separator(f, orient="horizontal").pack(fill=tk.X, pady=6)

        # ── Score range ───────────────────────────────────────────────────────
        ttk.Label(f, text="Score range", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._filter_min_score = tk.DoubleVar(value=0.0)
        self._filter_max_score = tk.DoubleVar(value=1.0)
        self._lbl_score_range  = ttk.Label(f, text="0.00 – 1.00", foreground="#555555")
        self._lbl_score_range.pack(anchor="w")

        for label, var in (("Min", self._filter_min_score), ("Max", self._filter_max_score)):
            ttk.Label(f, text=label).pack(anchor="w")
            ttk.Scale(
                f, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                variable=var, command=self._on_score_slider,
            ).pack(fill=tk.X)

        ttk.Separator(f, orient="horizontal").pack(fill=tk.X, pady=6)

        # ── Status ────────────────────────────────────────────────────────────
        ttk.Label(f, text="Status", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._filter_status_vars: dict[str, tk.BooleanVar] = {}
        for status in ("approved", "rejected", "unreviewed"):
            var = tk.BooleanVar(value=True)
            self._filter_status_vars[status] = var
            ttk.Checkbutton(f, text=status, variable=var).pack(anchor="w")

        ttk.Separator(f, orient="horizontal").pack(fill=tk.X, pady=6)

        # ── Person tags ───────────────────────────────────────────────────────
        ttk.Label(f, text="Person tags", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self._filter_tag_vars: dict[str, tk.BooleanVar] = {}
        for person in self.persons:
            var = tk.BooleanVar(value=False)
            self._filter_tag_vars[person["name"]] = var
            ttk.Checkbutton(f, text=person["name"].title(), variable=var).pack(anchor="w")

        ttk.Separator(f, orient="horizontal").pack(fill=tk.X, pady=6)

        # Count label + buttons
        self._lbl_filter_count = ttk.Label(
            f,
            text=f"Showing {len(self.photos)}/{len(self.photos)}",
            foreground="#555555",
            font=("Segoe UI", 8),
        )
        self._lbl_filter_count.pack(anchor="w")

        btn_row = ttk.Frame(f)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_row, text="Apply", command=self._apply_filters).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Clear", command=self._clear_filters).pack(
            side=tk.LEFT, padx=4
        )

    # ── Key bindings ──────────────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        """Register all keyboard shortcuts on the root window."""
        root = self.root
        root.bind("<Key-a>",     lambda e: self._navigate(-1))
        root.bind("<Key-A>",     lambda e: self._navigate(-1))
        root.bind("<Key-d>",     lambda e: self._navigate(+1))
        root.bind("<Key-D>",     lambda e: self._navigate(+1))
        root.bind("<space>",     lambda e: self._cycle_status())
        root.bind("<BackSpace>", lambda e: self._set_status("rejected"))
        root.bind("<Key-g>",     lambda e: self._toggle_person_tag("gautam"))
        root.bind("<Key-G>",     lambda e: self._toggle_person_tag("gautam"))
        root.bind("<Key-f>",     lambda e: self._toggle_filter_panel())
        root.bind("<Key-F>",     lambda e: self._toggle_filter_panel())

        for person in self.persons:
            if person["letter"].upper() == "G":
                continue
            lo   = person["letter"].lower()
            hi   = person["letter"].upper()
            name = person["name"]
            root.bind(f"<Key-{lo}>", lambda e, n=name: self._toggle_person_tag(n))
            root.bind(f"<Key-{hi}>", lambda e, n=name: self._toggle_person_tag(n))

    # ── Photo display ─────────────────────────────────────────────────────────

    def _show_photo(self, idx: int) -> None:
        """Navigate to and display the photo at position idx in active_photos.

        Starts a background thread for image loading.  Shows a placeholder
        immediately while loading proceeds.  Pre-loads the two neighbours.

        Args:
            idx: Index into self.active_photos.
        """
        if not self.active_photos:
            self._draw_placeholder("No photos match current filters")
            self._update_info_panel()
            self._update_progress_bar()
            return

        self.active_idx = max(0, min(idx, len(self.active_photos) - 1))
        record = self.active_photos[self.active_idx]
        path   = record["path"]

        # Cancel any in-flight load for a previous photo
        self._load_cancel.set()
        self._load_cancel = threading.Event()
        cancel = self._load_cancel

        self._draw_placeholder("Loading…")
        self._update_info_panel()
        self._update_progress_bar()

        if path in self._pil_cache:
            self._draw_photo(self._pil_cache[path])
        else:
            threading.Thread(
                target=self._load_image_worker,
                args=(path, cancel, True),
                daemon=True,
            ).start()

        # Preload neighbours into cache (display=False)
        for delta in (+1, -1):
            ni = self.active_idx + delta
            if 0 <= ni < len(self.active_photos):
                npath = self.active_photos[ni]["path"]
                if npath not in self._pil_cache:
                    threading.Thread(
                        target=self._load_image_worker,
                        args=(npath, cancel, False),
                        daemon=True,
                    ).start()

    def _load_image_worker(
        self, path: str, cancel: threading.Event, display: bool
    ) -> None:
        """Open an image file in a background thread and optionally display it.

        Caches the raw PIL.Image (up to 5 entries).  Posts _draw_photo back to
        the main thread via root.after when display=True.

        Args:
            path:    Full path to the photo file.
            cancel:  Shared cancellation event; result discarded if set.
            display: If True, call _draw_photo on the main thread when done.
        """
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            self.logger.warning("Could not load %s: %s", path, exc)
            img = None

        if cancel.is_set():
            return

        if img is not None:
            # LRU trim: evict oldest entry when cache is full
            if len(self._pil_cache) >= 5:
                oldest = next(iter(self._pil_cache))
                del self._pil_cache[oldest]
            self._pil_cache[path] = img

        if display and not cancel.is_set():
            self.root.after(0, lambda: self._draw_photo(img))

    def _draw_photo(self, pil_img) -> None:
        """Resize a PIL image to fit the canvas and draw it with border and badges.

        Args:
            pil_img: PIL.Image to display, or None for an error placeholder.
        """
        if pil_img is None:
            self._draw_placeholder("Could not load photo")
            return

        w = self.photo_canvas.winfo_width()
        h = self.photo_canvas.winfo_height()
        if w < 2 or h < 2:
            # Canvas not yet laid out — retry after next layout pass
            self.root.after(50, lambda: self._draw_photo(pil_img))
            return

        self._current_pil = pil_img
        img = pil_img.copy()
        img.thumbnail((w - 8, h - 8), Image.LANCZOS)
        self._current_tk = ImageTk.PhotoImage(img)

        self.photo_canvas.delete("all")
        self.photo_canvas.create_image(
            w // 2, h // 2, anchor=tk.CENTER, image=self._current_tk
        )
        self._update_border()
        self._draw_badges()

    def _draw_placeholder(self, text: str = "Loading…") -> None:
        """Draw a grey placeholder message centred on the canvas.

        Args:
            text: Message to display.
        """
        self.photo_canvas.delete("all")
        self._current_pil = None
        self._current_tk  = None
        w = self.photo_canvas.winfo_width()  or 600
        h = self.photo_canvas.winfo_height() or 400
        self.photo_canvas.create_text(
            w // 2, h // 2,
            text=text, fill="#666666", font=("Segoe UI", 12),
        )
        self.photo_canvas.config(highlightthickness=0)

    def _draw_badges(self) -> None:
        """Draw person-tag badges in the top-right corner of the canvas."""
        self.photo_canvas.delete("badge")
        if not self.active_photos:
            return

        path = self.active_photos[self.active_idx]["path"]
        tags = self._get_tags(path)
        if not tags:
            return

        badge_w, badge_h, gap = 24, 22, 4
        x = self.photo_canvas.winfo_width() - 10
        y = 10

        for person in reversed(self.persons):
            if person["name"] not in tags:
                continue
            x0, y0 = x - badge_w, y
            x1, y1 = x,           y + badge_h
            self.photo_canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=person["color"], outline="white", width=1, tags="badge",
            )
            self.photo_canvas.create_text(
                (x0 + x1) // 2, (y0 + y1) // 2,
                text=person["letter"].upper(),
                fill="white", font=("Segoe UI", 9, "bold"),
                tags="badge",
            )
            x -= badge_w + gap

    def _update_border(self) -> None:
        """Set the canvas highlight colour to reflect the current photo's status."""
        if not self.active_photos:
            self.photo_canvas.config(highlightthickness=0)
            return
        path   = self.active_photos[self.active_idx]["path"]
        status = self._get_status(path)
        if status == "approved":
            self.photo_canvas.config(
                highlightthickness=4, highlightbackground=_BORDER_APPROVED
            )
        elif status == "rejected":
            self.photo_canvas.config(
                highlightthickness=4, highlightbackground=_BORDER_REJECTED
            )
        else:
            self.photo_canvas.config(highlightthickness=0)

    def _on_canvas_resize(self, _event=None) -> None:
        """Debounced handler — redraws the current photo after the window settles."""
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(_RESIZE_DEBOUNCE, self._redraw_on_resize)

    def _redraw_on_resize(self) -> None:
        """Redraw the current photo at the new canvas dimensions after debounce."""
        self._resize_after_id = None
        if self._current_pil is not None:
            self._draw_photo(self._current_pil)

    # ── Info panel ────────────────────────────────────────────────────────────

    def _update_info_panel(self) -> None:
        """Refresh all right-panel labels for the current photo."""
        if not self.active_photos:
            self._lbl_filename.config(text="")
            self._lbl_meta.config(text="No photos match filters")
            self._lbl_nav.config(text="Photo — / —")
            return

        record = self.active_photos[self.active_idx]
        path   = record["path"]

        self._lbl_filename.config(text=os.path.basename(path))

        shot   = record.get("primary_shot_type") or "—"
        moment = record.get("moment_label") or "unknown"
        self._lbl_meta.config(text=f"{shot}  •  {moment}")

        self._lbl_nima.config(    text=f"{record.get('nima_score', 0):.2f}")
        self._lbl_emotion.config( text=f"{record.get('emotion_score', 0):.2f}")
        self._lbl_memorab.config( text=f"{record.get('memorability_score', 0):.2f}")
        self._lbl_final.config(   text=f"{record.get('final_score', 0):.3f}")

        sc   = record.get("score_components") or {}
        mods = []
        if sc.get("candid_bonus"):        mods.append("candid +10%")
        if sc.get("soft_blur_penalty"):   mods.append("soft blur −5%")
        if sc.get("closed_eyes_penalty"): mods.append("closed eyes −15%")
        self._lbl_mods.config(text="  ".join(mods) if mods else "")

        status = self._get_status(path)
        _text  = {"approved": "● APPROVED", "rejected": "✗ REJECTED",
                  "unreviewed": "○ unreviewed"}
        _color = {"approved": "#00C851", "rejected": "#FF4444",
                  "unreviewed": "#888888"}
        self._lbl_status.config(text=_text[status], foreground=_color[status])

        tags = self._get_tags(path)
        if tags:
            letters = "  ".join(
                p["letter"].upper() for p in self.persons if p["name"] in tags
            )
            self._lbl_tags.config(text=f"Tags: {letters}")
        else:
            self._lbl_tags.config(text="Tags: (none)")

        total = len(self.active_photos)
        suffix = (
            f"  (of {len(self.photos)} total)"
            if total != len(self.photos) else ""
        )
        self._lbl_nav.config(text=f"Photo {self.active_idx + 1} / {total}{suffix}")

    def _update_progress_bar(self) -> None:
        """Refresh the progress bar and approved / rejected / unreviewed counts."""
        approved   = sum(1 for p in self.photos if self._get_status(p["path"]) == "approved")
        rejected   = sum(1 for p in self.photos if self._get_status(p["path"]) == "rejected")
        unreviewed = len(self.photos) - approved - rejected
        self._progress_var.set(approved)
        self._lbl_counts.config(text=f"✓{approved}  ✗{rejected}  ?{unreviewed}")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _navigate(self, delta: int) -> None:
        """Move delta steps in the active photo list, wrapping at the ends.

        Args:
            delta: +1 for next, -1 for previous.
        """
        if not self.active_photos:
            return
        self._show_photo((self.active_idx + delta) % len(self.active_photos))

    def _cycle_status(self) -> None:
        """Space key handler: cycle status between approved and unreviewed.

        If the current status is rejected, Space moves to approved rather than
        unreviewed so the reviewer can recover a rejected photo in one keystroke.
        """
        if not self.active_photos:
            return
        path   = self.active_photos[self.active_idx]["path"]
        status = self._get_status(path)
        self._set_status("unreviewed" if status == "approved" else "approved")

    def _set_status(self, status: str) -> None:
        """Set the review status of the current photo and persist progress.

        Args:
            status: One of "approved", "rejected", "unreviewed".
        """
        if not self.active_photos:
            return
        path = self.active_photos[self.active_idx]["path"]
        if path not in self.reviews:
            self.reviews[path] = {"status": "unreviewed", "person_tags": []}
        self.reviews[path]["status"] = status
        self._save_progress_now()
        self._update_info_panel()
        self._update_progress_bar()
        self._update_border()

    def _toggle_person_tag(self, name: str) -> None:
        """Toggle a person tag on the current photo and persist progress.

        Args:
            name: Person name to toggle (must match a name in self.persons).
        """
        if not self.active_photos:
            return
        path = self.active_photos[self.active_idx]["path"]
        if path not in self.reviews:
            self.reviews[path] = {"status": "unreviewed", "person_tags": []}
        tags = self.reviews[path]["person_tags"]
        if name in tags:
            tags.remove(name)
        else:
            tags.append(name)
        self._save_progress_now()
        self._update_info_panel()
        self._draw_badges()

    # ── Filter panel ──────────────────────────────────────────────────────────

    def _toggle_filter_panel(self) -> None:
        """Show or hide the filter sidebar (column 1), shrinking the photo panel."""
        self._filter_visible = not self._filter_visible
        if self._filter_visible:
            self._main.columnconfigure(1, weight=2, minsize=180)
            self._filter.grid(row=0, column=1, sticky="nsew", padx=(0, 2), pady=2)
        else:
            self._filter.grid_remove()
            self._main.columnconfigure(1, weight=0, minsize=0)

    def _on_score_slider(self, _=None) -> None:
        """Update the score range label when either slider moves."""
        lo = self._filter_min_score.get()
        hi = self._filter_max_score.get()
        self._lbl_score_range.config(text=f"{lo:.2f} – {hi:.2f}")

    def _apply_filters(self) -> None:
        """Rebuild active_photos to include only photos matching all active filters."""
        shot_types = {s for s, v in self._filter_shot_vars.items()   if v.get()}
        statuses   = {s for s, v in self._filter_status_vars.items() if v.get()}
        tag_names  = {n for n, v in self._filter_tag_vars.items()    if v.get()}
        lo = self._filter_min_score.get()
        hi = self._filter_max_score.get()

        filtered = []
        for record in self.photos:
            if shot_types and record.get("primary_shot_type") not in shot_types:
                continue
            score = record.get("final_score") or 0.0
            if not (lo <= score <= hi):
                continue
            if statuses and self._get_status(record["path"]) not in statuses:
                continue
            if tag_names:
                photo_tags = set(self._get_tags(record["path"]))
                if not tag_names.intersection(photo_tags):
                    continue
            filtered.append(record)

        self.active_photos = filtered
        self._lbl_filter_count.config(
            text=f"Showing {len(filtered)}/{len(self.photos)}"
        )

        if filtered:
            self.active_idx = min(self.active_idx, len(filtered) - 1)
            self._show_photo(self.active_idx)
        else:
            self.active_idx = 0
            self._draw_placeholder("No photos match filters")
            self._update_info_panel()
            self._update_progress_bar()

    def _clear_filters(self) -> None:
        """Reset all filter controls to defaults (show all photos)."""
        for v in self._filter_shot_vars.values():
            v.set(True)
        for v in self._filter_status_vars.values():
            v.set(True)
        for v in self._filter_tag_vars.values():
            v.set(False)
        self._filter_min_score.set(0.0)
        self._filter_max_score.set(1.0)
        self._lbl_score_range.config(text="0.00 – 1.00")
        self._apply_filters()

    # ── Progress persistence ──────────────────────────────────────────────────

    def _build_progress_data(self) -> dict:
        """Build the full progress dict ready for JSON serialisation.

        Returns:
            Dict with keys: version, persons, reviews.
        """
        return {
            "version": 1,
            "persons": self.persons,
            "reviews": self.reviews,
        }

    def _save_progress_now(self) -> None:
        """Atomically write current state to review_progress.json."""
        _save_progress(
            config.REVIEW_PROGRESS_FILE,
            self._build_progress_data(),
            self.logger,
        )

    def _get_status(self, path: str) -> str:
        """Return the review status of a photo path.

        Args:
            path: Photo file path (used as key in self.reviews).

        Returns:
            One of "approved", "rejected", "unreviewed".
        """
        return self.reviews.get(path, {}).get("status", "unreviewed")

    def _get_tags(self, path: str) -> list:
        """Return the person tags applied to a photo.

        Args:
            path: Photo file path.

        Returns:
            List of person name strings (may be empty).
        """
        return self.reviews.get(path, {}).get("person_tags", [])

    # ── Completion and export ─────────────────────────────────────────────────

    def _show_completion_dialog(self) -> None:
        """Show a summary dialog offering export or continued review."""
        approved   = [p for p in self.photos if self._get_status(p["path"]) == "approved"]
        rejected   = [p for p in self.photos if self._get_status(p["path"]) == "rejected"]
        unreviewed = [p for p in self.photos if self._get_status(p["path"]) == "unreviewed"]

        tag_counts = {
            p["name"]: sum(
                1 for ph in self.photos if p["name"] in self._get_tags(ph["path"])
            )
            for p in self.persons
        }

        dlg = tk.Toplevel(self.root)
        dlg.title("Review Complete")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(
            dlg, text="Review Summary", font=("Segoe UI", 13, "bold")
        ).pack(padx=16, pady=(14, 4))
        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=4)

        for label, val in [
            ("Total photos",  len(self.photos)),
            ("✓ Approved",    len(approved)),
            ("✗ Rejected",    len(rejected)),
            ("? Unreviewed",  len(unreviewed)),
        ]:
            row = ttk.Frame(dlg)
            row.pack(fill=tk.X, padx=16)
            ttk.Label(row, text=label, width=16, anchor="w").pack(side=tk.LEFT)
            ttk.Label(row, text=str(val), font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        if any(tag_counts.values()):
            ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=4)
            ttk.Label(
                dlg, text="Per-person tags:", font=("Segoe UI", 9, "bold")
            ).pack(anchor="w", padx=16)
            for person in self.persons:
                ct  = tag_counts.get(person["name"], 0)
                row = ttk.Frame(dlg)
                row.pack(fill=tk.X, padx=24)
                ttk.Label(row, text=person["name"].title(), width=14, anchor="w").pack(
                    side=tk.LEFT
                )
                ttk.Label(row, text=f"{ct} photos").pack(side=tk.LEFT)

        # Warning if approved > 200
        if len(approved) > 200:
            ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=4)
            ttk.Label(
                dlg,
                text=(
                    f"You approved {len(approved)} photos.\n"
                    "Album target is 200.  All approved photos will be exported.\n"
                    "Consider going back to refine, or export all."
                ),
                foreground="#E67E22",
                font=("Segoe UI", 9),
                justify=tk.LEFT,
                wraplength=280,
            ).pack(padx=16, pady=4)

        ttk.Separator(dlg, orient="horizontal").pack(fill=tk.X, padx=16, pady=4)

        btn_row = ttk.Frame(dlg)
        btn_row.pack(padx=16, pady=(0, 14))

        def do_export():
            dlg.destroy()
            self._export(approved)

        ttk.Button(
            btn_row, text="Export & Close", command=do_export, padding=(12, 4)
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn_row, text="Continue Reviewing", command=dlg.destroy, padding=(12, 4)
        ).pack(side=tk.LEFT)

        # Centre over main window
        dlg.update_idletasks()
        rx = self.root.winfo_x() + (self.root.winfo_width()  - dlg.winfo_reqwidth())  // 2
        ry = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{rx}+{ry}")

    def _export(self, approved: list) -> None:
        """Copy approved and person-tagged photos to output folders.

        Runs in a background thread with a progress dialog on the main thread.
        Uses utils.copy_to_output (preserves filename, skips existing, never
        modifies originals).

        Args:
            approved: List of photo records with status "approved".
        """
        # Build per-person tagged lists
        tagged_map: dict[str, list] = {}
        for person in self.persons:
            tagged = [
                p for p in self.photos
                if person["name"] in self._get_tags(p["path"])
            ]
            if tagged:
                tagged_map[person["name"]] = tagged

        total_ops = len(approved) + sum(len(v) for v in tagged_map.values())

        # Progress dialog
        prog = tk.Toplevel(self.root)
        prog.title("Exporting…")
        prog.resizable(False, False)
        prog.transient(self.root)
        prog.grab_set()

        ttk.Label(
            prog, text="Copying photos…", font=("Segoe UI", 10)
        ).pack(padx=20, pady=(14, 4))

        pb_var = tk.IntVar(value=0)
        pb = ttk.Progressbar(
            prog, variable=pb_var, maximum=max(1, total_ops), length=320
        )
        pb.pack(padx=20, pady=4)

        status_lbl = ttk.Label(prog, text="", foreground="#555555", font=("Segoe UI", 8))
        status_lbl.pack(padx=20, pady=(0, 14))

        prog.update_idletasks()
        rx = self.root.winfo_x() + (self.root.winfo_width()  - prog.winfo_reqwidth())  // 2
        ry = self.root.winfo_y() + (self.root.winfo_height() - prog.winfo_reqheight()) // 2
        prog.geometry(f"+{rx}+{ry}")

        def worker() -> None:
            errors = 0
            step   = 0

            # Copy approved → album_approved/
            for record in approved:
                result = copy_to_output(record["path"], config.ALBUM_APPROVED_FOLDER)
                if result is None:
                    errors += 1
                step += 1
                fname = os.path.basename(record["path"])
                self.root.after(0, lambda s=step: pb_var.set(s))
                self.root.after(0, lambda n=fname: status_lbl.config(text=n))

            # Copy tagged → person_highlights/{name}/
            for pname, records in tagged_map.items():
                folder = os.path.join(config.PERSON_HIGHLIGHTS_FOLDER, pname)
                for record in records:
                    result = copy_to_output(record["path"], folder)
                    if result is None:
                        errors += 1
                    step += 1
                    self.root.after(0, lambda s=step: pb_var.set(s))

            def finish() -> None:
                prog.destroy()
                lines = [
                    f"✓ {len(approved)} approved  →  {config.ALBUM_APPROVED_FOLDER}",
                ]
                for pname, records in tagged_map.items():
                    folder = os.path.join(config.PERSON_HIGHLIGHTS_FOLDER, pname)
                    lines.append(f"  {pname}: {len(records)} photos  →  {folder}")
                if errors:
                    lines.append(f"\n⚠ {errors} file(s) failed to copy — see log")
                self.logger.info(
                    "Export complete: approved=%d  persons=%d  errors=%d",
                    len(approved), len(tagged_map), errors,
                )
                messagebox.showinfo(
                    "Export Complete", "\n".join(lines), parent=self.root
                )

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Launch the Phase 5 review application."""
    parser = argparse.ArgumentParser(
        description="Phase 5 -- Keyboard-driven photo review UI."
    )
    parser.add_argument(
        "--input",
        default=config.PHASE5_INPUT,
        help="Phase 4 selected-photos JSON to review (default: %(default)s)",
    )
    args = parser.parse_args()

    logger = setup_logging("phase5_review")
    logger.info("=== Phase 5 review started  input=%s ===", args.input)

    try:
        photos = _load_top800(args.input)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}")
        logger.error("Input file not found: %s", args.input)
        return

    logger.info("Loaded %d photos for review", len(photos))

    root = tk.Tk()
    root.title("Wedding Photo Review — Phase 5")
    root.geometry("1400x900")
    root.minsize(900, 600)

    # ── Resume check ─────────────────────────────────────────────────────────
    progress = None
    persons  = None

    if os.path.exists(config.REVIEW_PROGRESS_FILE):
        try:
            with open(config.REVIEW_PROGRESS_FILE, encoding="utf-8") as f:
                saved = json.load(f)
            reviewed = sum(
                1 for v in saved.get("reviews", {}).values()
                if v.get("status") != "unreviewed"
            )
            remaining = len(photos) - reviewed
            resume = messagebox.askyesno(
                "Resume Session",
                f"Resume previous review session?\n\n"
                f"{reviewed} reviewed  |  {remaining} remaining",
                parent=root,
            )
            if resume:
                progress = saved
                persons  = saved.get("persons") or [
                    {"letter": "G", "name": "gautam",
                     "color": config.PERSON_TAG_COLORS.get("gautam", "#4A90D9")}
                ]
                logger.info("Resuming session: %d reviewed", reviewed)
            else:
                os.remove(config.REVIEW_PROGRESS_FILE)
                logger.info("Starting fresh — previous progress discarded")
        except Exception as exc:
            logger.warning("Could not load progress file: %s", exc)

    # ── Person config (only if not resuming) ─────────────────────────────────
    if persons is None:
        dlg = StartupDialog(root, len(photos), logger)
        if dlg.result is None:
            root.destroy()
            logger.info("Review cancelled at startup dialog")
            return
        persons = dlg.result

    logger.info(
        "Persons: %s", [f"{p['letter']}={p['name']}" for p in persons]
    )

    # ── Launch ────────────────────────────────────────────────────────────────
    ReviewApp(root, photos, persons, progress, logger)
    root.mainloop()

    logger.info("=== Phase 5 review session ended ===")


if __name__ == "__main__":
    main()
