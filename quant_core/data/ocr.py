"""Last-resort document extraction: text PDFs first, OCR only when forced.

Dependency policy: nothing here is required by the Streamlit runtime. The
extraction chain degrades explicitly:

1. ``pdfplumber`` for born-digital PDFs (embedded text and tables).
2. ``pytesseract`` + OpenCV preprocessing only for image-only pages
   (requires the system ``tesseract-ocr`` binary — free, installed via apt
   in GitHub Actions or locally).

Install extras with ``pip install -r requirements-ocr.txt``. Every OCR output
must pass :func:`validate_ocr_frame` before it may enter the cache — OCR
never writes unvalidated data.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _require_pdfplumber():
    try:
        import pdfplumber  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("pdfplumber is required: pip install -r requirements-ocr.txt") from exc
    return pdfplumber


def extract_pdf_tables(path: str | Path, *, max_pages: int | None = None) -> list[pd.DataFrame]:
    """Extract tables from a born-digital PDF via pdfplumber."""
    pdfplumber = _require_pdfplumber()
    tables: list[pd.DataFrame] = []
    with pdfplumber.open(str(path)) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page in pages:
            for raw in page.extract_tables() or []:
                if not raw or len(raw) < 2:
                    continue
                frame = pd.DataFrame(raw[1:], columns=[str(c).strip() for c in raw[0]])
                if not frame.empty:
                    tables.append(frame)
    return tables


def pdf_page_text(path: str | Path, page_number: int = 0) -> str:
    """Embedded text of one PDF page (empty string for image-only pages)."""
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(path)) as pdf:
        if page_number >= len(pdf.pages):
            return ""
        return pdf.pages[page_number].extract_text() or ""


def _configure_tesseract(pytesseract_module) -> None:
    """Point pytesseract to a local Tesseract binary when it is not on PATH.

    The zero-cost Windows installer usually places the executable under the
    Program Files Tesseract-OCR directory without adding it to PATH. Keep the
    behavior explicit and overridable via ``TESSERACT_CMD``.
    """
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return


def ocr_image(path: str | Path, *, lang: str = "eng", psm: int = 6) -> str:
    """OCR a raster image file using the validated Tesseract backend."""
    try:  # pragma: no cover - exercised only with optional deps installed.
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pytesseract/Pillow are required: pip install -r requirements-ocr.txt") from exc
    _configure_tesseract(pytesseract)
    return pytesseract.image_to_string(Image.open(str(path)), lang=lang, config=f"--psm {int(psm)}")


def ocr_pdf_page(path: str | Path, page_number: int = 0, *, dpi: int = 300, lang: str = "spa+eng") -> str:
    """OCR an image-only PDF page (requires pytesseract + opencv + tesseract)."""
    try:  # pragma: no cover - exercised only with optional deps installed.
        import cv2  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pytesseract/opencv are required: pip install -r requirements-ocr.txt") from exc
    _configure_tesseract(pytesseract)
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(str(path)) as pdf:  # pragma: no cover
        if page_number >= len(pdf.pages):
            return ""
        image = pdf.pages[page_number].to_image(resolution=dpi).original
        arr = np.array(image.convert("L"))
        # Binarize + light denoise before OCR.
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(binary, lang=lang)


def validate_ocr_frame(
    frame: pd.DataFrame,
    *,
    numeric_columns: dict[str, tuple[float, float]] | None = None,
    date_column: str | None = None,
    require_monotonic_dates: bool = True,
    max_null_fraction: float = 0.20,
) -> tuple[bool, list[str]]:
    """Schema/plausibility gate every OCR-extracted table must pass.

    Checks: non-empty, bounded null fraction, numeric columns inside
    plausible ranges, and (optionally) monotonically non-decreasing dates.
    Returns (passed, list_of_violations).
    """
    violations: list[str] = []
    if frame is None or frame.empty:
        return False, ["empty_frame"]
    null_fraction = float(frame.isna().mean().mean())
    if null_fraction > max_null_fraction:
        violations.append(f"null_fraction:{null_fraction:.2f}")
    for col, (lo, hi) in (numeric_columns or {}).items():
        if col not in frame.columns:
            violations.append(f"missing_column:{col}")
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        if vals.empty:
            violations.append(f"non_numeric:{col}")
            continue
        if float(vals.min()) < lo or float(vals.max()) > hi:
            violations.append(f"out_of_range:{col}")
    if date_column:
        if date_column not in frame.columns:
            violations.append(f"missing_column:{date_column}")
        else:
            dates = pd.to_datetime(frame[date_column], errors="coerce")
            if dates.isna().any():
                violations.append(f"unparseable_dates:{date_column}")
            elif require_monotonic_dates and not dates.is_monotonic_increasing:
                violations.append(f"non_monotonic_dates:{date_column}")
    return (not violations), violations
