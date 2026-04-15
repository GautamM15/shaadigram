"""
download_models.py — Download and verify model weights for the wedding curator pipeline.

Downloads:
    aesthetic_model.pth — LAION improved-aesthetic-predictor MLP weights
                          (sa_0_4_vit_l_14_linear.pth, ~3.7 MB)

Verifies:
    CLIP ViT-B/32 via transformers (phase 2 CLIP embeddings)
    CLIP ViT-L/14 via transformers (phase 3 LAION aesthetic scoring)

Usage:
    python download_models.py
"""

import os
import sys

import config

# ── Download targets ──────────────────────────────────────────────────────────

_LAION_DEST     = config.LAION_MODEL_PATH   # "aesthetic_model.pth"
_LAION_URLS     = [config.LAION_MODEL_URL]  # sa_0_4_vit_b_32_linear.pth from LAION-AI


# ── Step 1 — Download LAION aesthetic model ───────────────────────────────────

def download_laion(dest: str = _LAION_DEST) -> bool:
    """Download the LAION aesthetic predictor weights (sa_0_4_vit_b_32_linear.pth).

    Source: https://github.com/LAION-AI/aesthetic-predictor
    Architecture: nn.Linear(512, 1) — single linear layer on CLIP ViT-B/32 embeddings.

    First checks for the aesthetic-predictor pip package (which bundles the weights).
    Falls back to direct GitHub download if the package is unavailable.
    Skips download if dest already exists and is larger than 1 MB.

    Args:
        dest: Destination file path in project root (default: aesthetic_model.pth).

    Returns:
        True on success, False on failure.
    """
    print("\n" + "=" * 56)
    print("  Step 1 — LAION aesthetic model (aesthetic_model.pth)")
    print("=" * 56)

    # Check pip package first — if installed it handles weights internally
    try:
        from aesthetic_predictor import predict_aesthetic  # noqa: F401
        print("  aesthetic-predictor pip package found — weights bundled, no download needed")
        return True
    except ImportError:
        print("  aesthetic-predictor pip package not installed — trying direct download")

    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"  Already present: {dest}  ({size_mb:.1f} MB) — skipping download")
        return True

    try:
        import requests as _req
    except ImportError:
        print("  ERROR: requests not installed — run: pip install requests")
        return False

    print(f"  Dest : {dest}")
    print(f"  Trying {len(_LAION_URLS)} source(s) ...")

    last_error = None
    for url in _LAION_URLS:
        print(f"\n  URL  : {url}")
        try:
            resp = _req.get(url, stream=True, timeout=60,
                            headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} — trying next source")
                last_error = f"HTTP {resp.status_code}"
                continue

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  Progress: {pct:5.1f}%  "
                                  f"({downloaded / 1_048_576:.1f}/{total / 1_048_576:.1f} MB)",
                                  end="", flush=True)
            print()

            size_mb = os.path.getsize(dest) / 1_048_576
            print(f"  Saved: {size_mb:.1f} MB")

            if size_mb < 0.5:
                print("  File too small — likely an LFS pointer, not the actual weights")
                os.remove(dest)
                last_error = "file too small (LFS pointer?)"
                continue

            # Verify it's a valid PyTorch checkpoint
            try:
                import torch
                state = torch.load(dest, map_location="cpu", weights_only=True)
                n_tensors = len(state) if isinstance(state, dict) else "?"
                print(f"  Verified: PyTorch checkpoint  ({n_tensors} tensors)")
            except ImportError:
                print("  NOTE: torch not installed — skipping checkpoint verification")
            except Exception as exc:
                print(f"  WARNING: torch.load failed: {exc}")
                os.remove(dest)
                last_error = str(exc)
                continue

            print("  LAION model: OK")
            return True

        except Exception as exc:
            last_error = str(exc)
            print(f"  Error: {exc}")
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass
            continue

    print(f"\n  All sources failed (last error: {last_error})")
    print("  Manual download:")
    print("    1. Go to: https://github.com/LAION-AI/aesthetic-predictor")
    print("    2. Download sa_0_4_vit_b_32_linear.pth")
    print(f"    3. Save as: {os.path.abspath(dest)}")
    print("  OR: pip install aesthetic-predictor")
    print("  Without this file, phase 3 will use BRISQUE (works fine).")
    return False


# ── Step 2 — Verify CLIP ViT-B/32 (phase 2) ──────────────────────────────────

def verify_clip_b32() -> bool:
    """Load CLIP ViT-B/32 via transformers and run a single forward pass.

    Downloads model weights from HuggingFace Hub on first run (~600 MB,
    cached at ~/.cache/huggingface/).

    Returns:
        True on success, False on failure.
    """
    print("\n" + "=" * 56)
    print("  Step 2 — CLIP ViT-B/32  (phase 2 embeddings)")
    print("=" * 56)

    try:
        import torch
        from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizerFast
    except ImportError as exc:
        print(f"  SKIP: import failed — {exc}")
        print("  Install with: pip install transformers torch torchvision")
        return False

    model_id = config.CLIP_MODEL   # "openai/clip-vit-base-patch32"
    print(f"  Model : {model_id}")
    print("  Loading (downloads ~600 MB to HuggingFace cache on first run) ...")

    try:
        img_proc  = CLIPImageProcessor.from_pretrained(model_id)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        model     = CLIPModel.from_pretrained(model_id)
        model.eval()

        from PIL import Image
        import numpy as np
        dummy  = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        # Image embedding (512-dim after visual_projection)
        v_in   = img_proc(images=[dummy], return_tensors="pt")
        with torch.no_grad():
            v_out  = model.vision_model(pixel_values=v_in["pixel_values"])
            img_f  = model.visual_projection(v_out.pooler_output)   # (1, 512)
        # Text embedding (512-dim after text_projection)
        t_in   = tokenizer(["wedding"], return_tensors="pt", padding=True)
        with torch.no_grad():
            t_out  = model.text_model(input_ids=t_in["input_ids"],
                                       attention_mask=t_in["attention_mask"])
            txt_f  = model.text_projection(t_out.pooler_output)   # (1, 512)
        print(f"  Image embedding: {tuple(img_f.shape)}  Text embedding: {tuple(txt_f.shape)}")
        print("  CLIP ViT-B/32: OK")
        return True

    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


# ── Step 3 — Verify CLIP ViT-L/14 (phase 3 LAION) ───────────────────────────

def verify_clip_l14() -> bool:
    """Load CLIP ViT-L/14 via transformers and run a single forward pass.

    Downloads model weights from HuggingFace Hub on first run (~890 MB,
    cached at ~/.cache/huggingface/).

    Returns:
        True on success, False on failure.
    """
    print("\n" + "=" * 56)
    print("  Step 3 — CLIP ViT-L/14  (phase 3 LAION aesthetic)")
    print("=" * 56)

    try:
        import torch
        from transformers import CLIPModel, CLIPImageProcessor
    except ImportError as exc:
        print(f"  SKIP: import failed — {exc}")
        return False

    model_id = "openai/clip-vit-large-patch14"
    print(f"  Model : {model_id}")
    print("  Loading (downloads ~890 MB to HuggingFace cache on first run) ...")

    try:
        img_proc = CLIPImageProcessor.from_pretrained(model_id)
        model    = CLIPModel.from_pretrained(model_id)
        model.eval()

        from PIL import Image
        import numpy as np
        dummy  = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        v_in   = img_proc(images=[dummy], return_tensors="pt")
        with torch.no_grad():
            v_out  = model.vision_model(pixel_values=v_in["pixel_values"])
            feats  = model.visual_projection(v_out.pooler_output)   # (1, 768)
        print(f"  Projected embedding shape: {tuple(feats.shape)}  (expected (1, 768) for LAION)")
        print("  CLIP ViT-L/14: OK")
        return True

    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Download and verify all model weights required by the pipeline."""
    print("\nWedding Curator — Model Download & Verification")
    print("=" * 56)

    results = {}
    results["laion"]       = download_laion()
    results["clip_b32"]    = verify_clip_b32()

    # ── Final report ──────────────────────────────────────────────────────────
    w = 56
    print("\n" + "=" * w)
    print("  Summary")
    print("=" * w)
    labels = {
        "laion":    "LAION aesthetic_model.pth (vit_b_32)",
        "clip_b32": "CLIP ViT-B/32 (phase 2 + phase 3)",
    }
    all_ok = True
    for key, label in labels.items():
        ok     = results[key]
        status = "OK" if ok else "FAILED"
        print(f"  {label:<34} : {status}")
        if not ok:
            all_ok = False

    print("=" * w)
    if all_ok:
        print("  All models ready. Pipeline is good to go.")
    else:
        print("  Some models failed — see details above.")
        print("  Pipeline will fall back to BRISQUE if LAION is missing.")
        print("  Phase 4 MMR requires clip_embeddings.npz from phase 2.")
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
