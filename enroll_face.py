"""
enroll_face.py — Register a known person's face embedding for use in phase 2.

Opens a tkinter GUI with 3 photo upload slots.  After all 3 are filled and
Enroll is clicked, DeepFace extracts a Facenet embedding from each photo,
the three are averaged and unit-normalised, then saved as
{person_name}_face.pkl in ENROLLED_FACES_FOLDER.

Usage:
    python enroll_face.py --person-name gautam
    python enroll_face.py --list
"""

import argparse
import glob
import os
import pickle
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from deepface import DeepFace
from PIL import Image, ImageTk

import config
from utils import setup_logging

# Deepface model used for enrollment — must match phase 2
_FACE_MODEL = "Facenet"

# Placeholder colour for empty thumbnail slots
_PLACEHOLDER_BG = "#cccccc"
_THUMB_SIZE = (80, 80)


def _make_placeholder() -> ImageTk.PhotoImage:
    """Create a grey PIL image of _THUMB_SIZE for use as a slot placeholder.

    Using a real image (not width/height text-unit arguments) guarantees
    the label is exactly _THUMB_SIZE pixels from the moment it is created.

    Returns:
        ImageTk.PhotoImage of a solid grey rectangle.
    """
    img = Image.new("RGB", _THUMB_SIZE, color=(204, 204, 204))
    return ImageTk.PhotoImage(img)


# ── Embedding helpers ─────────────────────────────────────────────────────────

def extract_embedding(image_path: str) -> np.ndarray:
    """Extract a Facenet face embedding from a single image.

    Uses DeepFace with enforce_detection=True so a missing face raises
    immediately rather than returning a bad embedding.

    Args:
        image_path: Absolute or relative path to a JPEG or PNG photo.

    Returns:
        1-D numpy array of shape (128,) — the raw Facenet embedding.

    Raises:
        ValueError: If DeepFace finds no face in the image.
        Exception:  Any other DeepFace or I/O error — propagated to caller.
    """
    result = DeepFace.represent(
        img_path=image_path,
        model_name=_FACE_MODEL,
        enforce_detection=True,
    )
    if not result:
        raise ValueError(f"No face detected in {image_path}")
    return np.array(result[0]["embedding"])


def average_embeddings(embeddings: list) -> np.ndarray:
    """Average a list of embeddings and normalise to unit length.

    Unit normalisation means phase 2 can use dot-product (cosine) similarity
    directly without an extra normalisation step.

    Args:
        embeddings: List of 1-D numpy arrays, all the same shape.

    Returns:
        Unit-normalised 1-D numpy array.
    """
    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    if norm == 0:
        return avg
    return avg / norm


def save_enrollment(person_name: str, embedding: np.ndarray,
                    source_paths: list) -> str:
    """Persist an enrollment record to disk as a pickle file.

    Overwrites any existing enrollment for the same person_name so
    re-enrollment always produces a clean record.

    Args:
        person_name: Identifier used as the filename stem.
        embedding:   Unit-normalised reference embedding to store.
        source_paths: List of the 3 source photo paths used.

    Returns:
        Absolute path to the saved .pkl file.
    """
    filename = f"{person_name}_face.pkl"
    dest = os.path.abspath(os.path.join(config.ENROLLED_FACES_FOLDER, filename))
    record = {
        "name": person_name,
        "embedding": embedding,
        "sources": source_paths,
    }
    with open(dest, "wb") as f:
        pickle.dump(record, f)
    return dest


def list_enrolled() -> list:
    """Return a sorted list of currently enrolled person names.

    Scans ENROLLED_FACES_FOLDER for files matching *_face.pkl and
    strips the suffix to recover the original person name.

    Returns:
        Sorted list of name strings.  Empty list if none found.
    """
    pattern = os.path.join(config.ENROLLED_FACES_FOLDER, "*_face.pkl")
    files = glob.glob(pattern)
    names = sorted(
        os.path.basename(f).replace("_face.pkl", "") for f in files
    )
    return names


# ── GUI helpers ───────────────────────────────────────────────────────────────

def load_photo_for_thumbnail(path: str,
                              size: tuple = _THUMB_SIZE) -> ImageTk.PhotoImage:
    """Load an image file and return a Tk-compatible thumbnail.

    Args:
        path: Path to the image file.
        size: (width, height) to constrain the thumbnail.

    Returns:
        ImageTk.PhotoImage suitable for a tkinter Label.

    Raises:
        Exception: Any PIL load failure — caller should handle.
    """
    img = Image.open(path)
    img.thumbnail(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_gui(person_name: str, logger) -> None:
    """Build and run the face enrollment GUI.

    Shows 3 photo slots inside a scrollable canvas.  Each slot has a
    thumbnail preview area and a Browse button.  The Enroll button is
    disabled until all 3 slots have a selected file.  On Enroll,
    embeddings are extracted, averaged, normalised, and saved.

    The window opens at 600x400, centred on screen, and is resizable.

    Args:
        person_name: Name of the person being enrolled, used in the window
                     title and as the output filename stem.
        logger:      Logger instance for this session.
    """
    root = tk.Tk()
    root.title(f"Enroll Face -- {person_name}")
    root.resizable(True, True)

    # ── Size and centre on screen ─────────────────────────────────────────────
    win_w, win_h = 600, 400
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - win_w) // 2
    y = (screen_h - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # ── Scrollable canvas (vertical + horizontal) ─────────────────────────────
    hscrollbar = ttk.Scrollbar(root, orient="horizontal")
    vscrollbar  = ttk.Scrollbar(root, orient="vertical")
    canvas = tk.Canvas(
        root,
        highlightthickness=0,
        xscrollcommand=hscrollbar.set,
        yscrollcommand=vscrollbar.set,
    )
    hscrollbar.config(command=canvas.xview)
    vscrollbar.config(command=canvas.yview)

    # Pack order: scrollbars first so canvas fills remaining space
    hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    vscrollbar.pack(side=tk.RIGHT,  fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    def on_frame_configure(event):
        """Update the canvas scroll region when inner content changes size."""
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner_frame.bind("<Configure>", on_frame_configure)

    def on_mousewheel(event):
        """Scroll the canvas with the mouse wheel (Windows)."""
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", on_mousewheel)

    # ── State ─────────────────────────────────────────────────────────────────
    slot_paths    = [tk.StringVar() for _ in range(3)]   # selected file paths
    # Pre-build placeholders; store in thumb_images to prevent GC
    thumb_images  = [_make_placeholder() for _ in range(3)]
    slot_frames   = []
    thumb_labels  = []
    enroll_btn    = None   # assigned after creation

    # ── Layout — all widgets parented to inner_frame ──────────────────────────
    header = ttk.Label(
        inner_frame,
        text=f"Select 3 clear, front-facing photos of  {person_name}",
        font=("Segoe UI", 10),
        padding=(12, 8),
    )
    header.pack()

    slots_frame = ttk.Frame(inner_frame, padding=(12, 4))
    slots_frame.pack()

    def check_enroll_ready(*_):
        """Enable the Enroll button only when all 3 slots are filled."""
        if all(p.get() for p in slot_paths):
            enroll_btn.config(state=tk.NORMAL)
        else:
            enroll_btn.config(state=tk.DISABLED)

    def make_browse_callback(idx: int):
        """Return a Browse callback bound to slot idx."""
        def browse():
            """Open a file dialog and load the selected photo into slot idx."""
            path = filedialog.askopenfilename(
                title=f"Select Photo {idx + 1}",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                photo = load_photo_for_thumbnail(path)
                thumb_images[idx] = photo           # hold reference
                thumb_labels[idx].config(
                    image=photo, background="white"
                )
                slot_paths[idx].set(path)
                slot_frames[idx].config(style="OK.TLabelframe")
                check_enroll_ready()
            except Exception as exc:
                logger.error("Could not load photo for slot %d: %s", idx + 1, exc)
                messagebox.showerror(
                    "Load Error",
                    f"Could not open photo:\n{exc}",
                )
        return browse

    # Build 3 slots side-by-side
    for i in range(3):
        frame = ttk.LabelFrame(
            slots_frame,
            text=f"  Photo {i + 1}  ",
            padding=(6, 6),
        )
        frame.grid(row=0, column=i, padx=8, pady=6)
        slot_frames.append(frame)

        # Thumbnail — initialised with a pixel-accurate grey placeholder image
        thumb = tk.Label(
            frame,
            image=thumb_images[i],   # real image = pixel-accurate size always
            relief="sunken",
        )
        thumb.pack(pady=(0, 4))
        thumb_labels.append(thumb)

        browse_btn = ttk.Button(
            frame,
            text="Browse",
            command=make_browse_callback(i),
        )
        browse_btn.pack()

    # ── Status label (shown during extraction) ────────────────────────────────
    status_label = ttk.Label(
        inner_frame,
        text="",
        font=("Segoe UI", 9),
        foreground="#555555",
    )
    status_label.pack(pady=(4, 0))

    # ── Enroll button ─────────────────────────────────────────────────────────
    enroll_btn = ttk.Button(
        inner_frame,
        text="Enroll",
        state=tk.DISABLED,
        command=lambda: on_enroll(),
        padding=(24, 6),
    )
    enroll_btn.pack(pady=(8, 16))

    # ── Enroll logic (threaded) ───────────────────────────────────────────────
    def on_enroll():
        """Kick off embedding extraction in a background thread.

        DeepFace can take 30-60 seconds on first run (model download +
        inference).  Running it in a daemon thread keeps the tkinter main
        loop responsive.  Results are communicated back via a shared dict
        and polled with root.after().
        """
        paths = [p.get() for p in slot_paths]
        _result = {}   # shared between worker thread and check_thread

        # ── Freeze UI while working ───────────────────────────────────────────
        enroll_btn.config(state=tk.DISABLED, text="Processing... (may take 60s)")
        status_label.config(text="Extracting embeddings -- please wait")
        root.update_idletasks()

        # ── Background worker ─────────────────────────────────────────────────
        def worker():
            """Extract embeddings for all 3 photos; store results in _result."""
            embeddings   = []
            failed_slots = []
            for idx, path in enumerate(paths):
                try:
                    emb = extract_embedding(path)
                    embeddings.append((idx, emb))
                except Exception as exc:
                    logger.error(
                        "No face detected in slot %d (%s): %s",
                        idx + 1, path, exc,
                    )
                    failed_slots.append(idx + 1)
            _result["embeddings"]   = embeddings
            _result["failed_slots"] = failed_slots
            _result["done"]         = True

        # ── Polling callback (main thread) ────────────────────────────────────
        def check_thread():
            """Poll every 100 ms; act when worker sets _result['done']."""
            if not _result.get("done"):
                root.after(100, check_thread)
                return

            # Worker is finished — safe to touch tkinter now
            failed_slots = _result["failed_slots"]

            if failed_slots:
                for idx in failed_slots:
                    slot_frames[idx - 1].config(style="Error.TLabelframe")
                enroll_btn.config(state=tk.NORMAL, text="Enroll")
                status_label.config(text="")
                messagebox.showerror(
                    "Face Detection Failed",
                    f"No face detected in Photo(s): {', '.join(map(str, failed_slots))}\n"
                    "Please replace those photos with a clear, front-facing image.",
                )
                return

            try:
                emb_arrays = [emb for _, emb in _result["embeddings"]]
                avg_emb    = average_embeddings(emb_arrays)
                saved_path = save_enrollment(person_name, avg_emb, paths)
                logger.info(
                    "Enrolled '%s' -- saved to %s  sources=%s",
                    person_name, saved_path, paths,
                )
                messagebox.showinfo(
                    "Enrollment Complete",
                    f"Enrolled: {person_name}\nSaved to: {os.path.basename(saved_path)}",
                )
                root.destroy()
            except Exception as exc:
                logger.error("Enrollment save failed: %s", exc)
                enroll_btn.config(state=tk.NORMAL, text="Enroll")
                status_label.config(text="")
                messagebox.showerror(
                    "Save Error",
                    f"Could not save enrollment:\n{exc}",
                )

        threading.Thread(target=worker, daemon=True).start()
        root.after(100, check_thread)

    # ── Custom LabelFrame styles for slot feedback ────────────────────────────
    style = ttk.Style(root)
    style.configure("OK.TLabelframe.Label",    foreground="green")
    style.configure("Error.TLabelframe.Label", foreground="red")

    root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Parse arguments and either list enrolled persons or open the GUI."""
    parser = argparse.ArgumentParser(
        description="Enroll a person's face embedding for phase 2 matching."
    )
    parser.add_argument(
        "--person-name",
        default=None,
        dest="person_name",
        help="Name of the person to enroll (e.g. gautam)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all currently enrolled persons and exit",
    )
    args = parser.parse_args()

    if args.list:
        names = list_enrolled()
        if names:
            print(f"Enrolled persons ({len(names)}):")
            for name in names:
                print(f"  - {name}")
        else:
            print(f"No enrolled persons found in {config.ENROLLED_FACES_FOLDER}")
        return

    if not args.person_name:
        parser.print_help()
        sys.exit(0)

    logger = setup_logging("enroll_face")
    logger.info("=== Enrollment started  person=%s ===", args.person_name)

    try:
        run_gui(args.person_name, logger)
    except Exception as exc:
        logger.error("GUI crashed unexpectedly: %s", exc)
        print(f"Error: {exc}")
        sys.exit(1)

    logger.info("=== Enrollment session ended ===")


if __name__ == "__main__":
    main()
