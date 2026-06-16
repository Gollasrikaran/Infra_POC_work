"""
PDF page classifier — figures out which pages are vector vs raster
and which ones are cross-section drawings (19-series and 23-series).
"""

import fitz
import re


VECTOR_MIN_DRAWINGS = 100


class PageInfo:
    """Holds analysis results for one page."""

    def __init__(self, page_number):
        self.page_number = page_number
        self.vector_count = 0
        self.image_count = 0
        self.text_block_count = 0
        self.page_width = 0.0
        self.page_height = 0.0
        self.classification = "UNKNOWN"
        self.workflow = "unknown"
        self.is_cross_section = False
        self.drawing_number = None
        self.h_scale = None
        self.v_scale = None


class DocInfo:
    """Holds analysis results for the whole PDF."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.total_pages = 0
        self.pages = []

    @property
    def vector_pages(self):
        return [p for p in self.pages if p.classification == "VECTOR"]

    @property
    def hybrid_pages(self):
        return [p for p in self.pages if p.classification == "HYBRID"]

    @property
    def raster_pages(self):
        return [p for p in self.pages if p.classification == "RASTER"]

    @property
    def cross_section_pages(self):
        return [p for p in self.pages if p.is_cross_section]

    @property
    def processable_pages(self):
        return [
            p for p in self.pages
            if p.is_cross_section and p.workflow in ("vector", "hybrid")
        ]


class PDFClassifier:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def analyze(self):
        doc = fitz.open(self.pdf_path)
        result = DocInfo(self.pdf_path)
        result.total_pages = len(doc)

        for i in range(len(doc)):
            page = doc[i]
            info = self._analyze_page(page, i)
            result.pages.append(info)

        doc.close()
        return result

    def _analyze_page(self, page, page_index):
        rect = page.rect
        drawings = page.get_drawings()
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")

        info = PageInfo(page_index + 1)
        info.vector_count = len(drawings)
        info.image_count = len(images)
        info.text_block_count = len(text_blocks)
        info.page_width = round(rect.width, 2)
        info.page_height = round(rect.height, 2)

        has_vectors = info.vector_count > VECTOR_MIN_DRAWINGS
        has_images = info.image_count > 0

        if has_vectors and not has_images:
            info.classification = "VECTOR"
        elif has_vectors and has_images:
            info.classification = "HYBRID"
        elif has_images:
            info.classification = "RASTER"

        wf_map = {"VECTOR": "vector", "HYBRID": "hybrid", "RASTER": "cv_fallback"}
        info.workflow = wf_map.get(info.classification, "unknown")

        self._check_cross_section(page, rect, info)
        return info

    def _check_cross_section(self, page, rect, info):
        corner = fitz.Rect(
            rect.width * 0.65, rect.height * 0.75,
            rect.width, rect.height
        )
        corner_text = page.get_text("text", clip=corner).strip()

        # look for 19-XXXX or 23-XXXX drawing numbers
        for line in corner_text.split("\n"):
            cleaned = re.sub(r"(?i)DRAWING|NO\.?|DRG|[:\s]", "", line).strip()
            m = re.match(r"((?:19|23)-\d+)", cleaned)
            if m:
                info.drawing_number = m.group(1)
                break

        if not info.drawing_number:
            return

        # accept "cross section" or "cross sections" or "staging cross sections"
        if not re.search(r"(?i)cross[-\s]*section", corner_text):
            return

        info.is_cross_section = True

        scale_area = fitz.Rect(0, rect.height * 0.70, rect.width, rect.height)
        scale_text = page.get_text("text", clip=scale_area)

        h_match = re.search(r'(?i)HORIZONTAL[:\s]*1"\s*=\s*(\d+\'?)', scale_text)
        v_match = re.search(r'(?i)VERTICAL[:\s]*1"\s*=\s*(\d+\'?)', scale_text)

        info.h_scale = f'1" = {h_match.group(1)}' if h_match else '1" = 10\' (default)'
        info.v_scale = f'1" = {v_match.group(1)}' if v_match else '1" = 10\' (default)'

    def to_dict_list(self, doc_info):
        return [
            {
                "page": p.page_number,
                "vectors": p.vector_count,
                "images": p.image_count,
                "type": p.classification,
                "workflow": p.workflow,
                "cross_section": p.is_cross_section,
                "drawing_no": p.drawing_number,
                "h_scale": p.h_scale,
                "v_scale": p.v_scale,
            }
            for p in doc_info.pages
        ]