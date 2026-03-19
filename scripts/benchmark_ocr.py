#!/usr/bin/env python3
"""OCR benchmark: compare EasyOCR, Tesseract, and PaddleOCR on UI element crops.

Usage:
    # Run with all available backends
    python scripts/benchmark_ocr.py

    # Run with a specific backend only
    python scripts/benchmark_ocr.py --backend easyocr

    # Run on a specific image (full page — will detect + crop first)
    python scripts/benchmark_ocr.py --image data/image.png

    # Run on pre-cropped button images
    python scripts/benchmark_ocr.py --crops data/buy_now.png data/price.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gui_detector_api.services.ocr import OcrBackend, OcrService


BACKENDS = [OcrBackend.EASYOCR, OcrBackend.TESSERACT, OcrBackend.PADDLEOCR]

DEFAULT_CROPS = [
    ROOT / "data" / "buy_now.png",
    ROOT / "data" / "price.png",
]


def load_crops(paths: list[Path]) -> list[tuple[str, Image.Image]]:
    crops = []
    for path in paths:
        if path.exists():
            crops.append((path.name, Image.open(path).convert("RGB")))
        else:
            print(f"  [skip] {path} not found")
    return crops


def benchmark_backend(backend: OcrBackend, crops: list[tuple[str, Image.Image]]) -> list[dict]:
    results = []
    try:
        service = OcrService(backend=backend)
        # Warm-up: trigger model load
        t0 = time.perf_counter()
        service.extract_text(crops[0][1])
        load_time = time.perf_counter() - t0
        print(f"\n  [{backend.value}] Model loaded in {load_time:.2f}s")
    except RuntimeError as exc:
        print(f"\n  [{backend.value}] SKIPPED: {exc}")
        return results

    for name, crop in crops:
        t0 = time.perf_counter()
        result = service.extract_text(crop)
        elapsed = time.perf_counter() - t0
        results.append({
            "backend": backend.value,
            "crop": name,
            "text": result.text,
            "confidence": result.confidence,
            "raw_texts": result.raw_texts,
            "time_ms": elapsed * 1000,
        })
        print(f"  {name:30s} | conf={result.confidence:.2f} | {elapsed*1000:.0f}ms | \"{result.text}\"")

    return results


def main():
    parser = argparse.ArgumentParser(description="OCR benchmark on UI element crops")
    parser.add_argument("--backend", choices=["easyocr", "tesseract", "paddleocr"], help="Test only this backend")
    parser.add_argument("--crops", nargs="+", type=Path, help="Paths to pre-cropped images")
    args = parser.parse_args()

    crop_paths = [Path(p) for p in args.crops] if args.crops else DEFAULT_CROPS
    crops = load_crops(crop_paths)

    if not crops:
        print("No crop images found. Place button crops in data/ or pass --crops.")
        sys.exit(1)

    print(f"Benchmarking {len(crops)} crop(s):")
    for name, crop in crops:
        print(f"  {name}: {crop.size[0]}x{crop.size[1]}")

    backends = [OcrBackend(args.backend)] if args.backend else BACKENDS
    all_results = []

    for backend in backends:
        results = benchmark_backend(backend, crops)
        all_results.extend(results)

    # Summary table
    if all_results:
        print("\n" + "=" * 80)
        print(f"{'Backend':15s} | {'Crop':30s} | {'Conf':>5s} | {'Time':>7s} | Text")
        print("-" * 80)
        for r in all_results:
            print(f"{r['backend']:15s} | {r['crop']:30s} | {r['confidence']:.2f}  | {r['time_ms']:6.0f}ms | \"{r['text']}\"")
        print("=" * 80)


if __name__ == "__main__":
    main()
