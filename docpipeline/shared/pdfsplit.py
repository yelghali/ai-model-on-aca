"""PDF page rasterisation and figure extraction (PyMuPDF).

The important structural point for the workflow:

    pages  != figures

A 300-page technical document may contain ~60 embedded diagrams. Some pages have
none, some have three. That is exactly WHY YOLO can be batched here while the
per-page text APIs cannot: figures are collected across the whole document and
then chunked, instead of being stuck at one image per page.

Every extracted figure carries (page_index, figure_index) so results can be
mapped back explicitly after batching - never by positional luck.
"""
from __future__ import annotations

import io

import fitz  # PyMuPDF


def page_count(pdf_bytes: bytes) -> int:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def render_page(pdf_bytes: bytes, page_index: int, dpi: int = 150) -> bytes:
    """Rasterise ONE page to PNG. Page images feed the text/vision APIs."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def extract_figures(pdf_bytes: bytes, page_index: int, min_pixels: int = 80 * 80) -> list[dict]:
    """Every figure on a page: embedded rasters AND vector drawings.

    Both are needed, and missing the second one is a classic trap. `get_images()`
    only sees images EMBEDDED in the PDF. A large share of technical diagrams -
    anything produced by TikZ, matplotlib PDF export, Illustrator, Visio - is
    stored as vector paths and is completely invisible to it.

    Measured on real papers with rasters only:
        bert.pdf    16 pages -> 34 figures
        resnet.pdf  12 pages ->  0 figures   <- a paper full of diagrams

    So vector regions are clustered and rendered too. Returns
    [{figure, width, height, kind, png}] with `kind` = raster | vector.
    """
    figures = _extract_raster(pdf_bytes, page_index, min_pixels)
    figures += _extract_vector(pdf_bytes, page_index, start_index=len(figures))
    return figures


def _extract_raster(pdf_bytes: bytes, page_index: int, min_pixels: int) -> list[dict]:
    """Embedded raster images.

    min_pixels filters out logos, rules and bullet glyphs that would otherwise
    inflate the YOLO batch with images containing nothing.
    """
    out: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        for fig_index, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            raw, w, h = base["image"], base.get("width", 0), base.get("height", 0)
            if w * h < min_pixels:
                continue
            # Normalise to PNG: extract_image returns whatever was embedded
            # (JPEG, JPX, CCITT...), and the model API expects a common format.
            try:
                pix = fitz.Pixmap(raw)
                if pix.n - pix.alpha >= 4:          # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png = pix.tobytes("png")
            except Exception:
                continue
            out.append({"figure": len(out), "width": w, "height": h,
                        "kind": "raster", "png": png})
    return out


def _extract_vector(pdf_bytes: bytes, page_index: int, start_index: int = 0,
                    dpi: int = 150, min_side: float = 60.0,
                    max_paths: int = 4000) -> list[dict]:
    """Cluster vector paths into figure regions and rasterise each region.

    A diagram is hundreds of separate path objects. Individually they are
    meaningless; clustered by proximity they form the figure. Clusters are then
    rendered with `clip=`, which is what turns a vector drawing into something
    an image model can consume.

    Guard rails: table rules and underlines are also vector paths, so clusters
    thinner than `min_side` in either axis are dropped, and anything covering
    almost the whole page is treated as a background frame, not a figure.
    """
    out: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        try:
            drawings = page.get_drawings()
        except Exception:
            return out

        rects = []
        for d in drawings[:max_paths]:
            r = d.get("rect")
            if r is None or r.is_empty or r.is_infinite:
                continue
            if r.width < 2 and r.height < 2:        # dots, tick marks
                continue
            rects.append(fitz.Rect(r))
        if not rects:
            return out

        merged = _merge_rects(rects, gap=12.0)
        page_area = page.rect.get_area() or 1.0

        for r in merged:
            if r.width < min_side or r.height < min_side:
                continue                             # rules, underlines, axes
            if r.get_area() / page_area > 0.92:
                continue                             # page border / frame
            try:
                clip = r & page.rect                 # never clip outside the page
                pix = page.get_pixmap(dpi=dpi, clip=clip)
                if pix.width < 16 or pix.height < 16:
                    continue
                out.append({
                    "figure": start_index + len(out),
                    "width": pix.width, "height": pix.height,
                    "kind": "vector", "png": pix.tobytes("png"),
                })
            except Exception:
                continue
    return out


def _merge_rects(rects: list, gap: float, max_passes: int = 6) -> list:
    """Union rectangles that touch or nearly touch, repeatedly until stable.

    Bounded passes on purpose: a page can contain thousands of paths and this is
    O(n^2) per pass. Slightly over-merging is far cheaper than scanning forever,
    and an over-merged figure is still a usable crop.
    """
    current = list(rects)
    for _ in range(max_passes):
        merged: list = []
        used = [False] * len(current)
        for i, r in enumerate(current):
            if used[i]:
                continue
            acc = fitz.Rect(r)
            used[i] = True
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                grown = fitz.Rect(acc.x0 - gap, acc.y0 - gap, acc.x1 + gap, acc.y1 + gap)
                if grown.intersects(current[j]):
                    acc |= current[j]
                    used[j] = True
            merged.append(acc)
        if len(merged) == len(current):
            return merged                            # stable
        current = merged
    return current
