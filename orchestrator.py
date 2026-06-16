"""
Pipeline orchestrator — cross-section workflow.

Flow:
  1. Classify pages (find 19/23-series)
  2. For each page: split into cross-section regions
  3. For each region: extract geometry → identify profiles → transform → normalize → compute area
  4. Sort all stations, compute volumes via Average End Area
  5. Generate reports
"""

import fitz

from pdf_classifier import PDFClassifier
from cross_section_splitter import CrossSectionSplitter
from geometry_extractor import GeometryExtractor
from profile_identifier import ProfileIdentifier
from coordinate_transformer import CoordinateTransformer
from profile_normalizer import ProfileNormalizer
from earthwork_calculator import EarthworkCalculator
from validation import Validator, ReportGenerator


class StationResult:
    """Everything produced for one cross-section station."""

    def __init__(self, station_label, station_ft, page_number):
        self.station_label = station_label
        self.station_ft = station_ft
        self.page_number = page_number
        self.drawing_number = ""

        self.region = None
        self.profiles = None
        self.scale = None
        self.existing_transformed = None
        self.proposed_transformed = None
        self.normalized = None
        self.station_area = None

        self.success = False
        self.error = None
        self.stage_reached = "init"


class PipelineResult:
    """All results across all pages + aggregate volumes."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.document_analysis = None
        self.station_results = []
        self.earthwork = None       # EarthworkResult with volumes

        self.plot_path = None
        self.json_report_path = None
        self.csv_report_path = None

    @property
    def successful_stations(self):
        return [s for s in self.station_results if s.success]

    @property
    def failed_stations(self):
        return [s for s in self.station_results if not s.success]

    @property
    def total_cut_volume_cy(self):
        return self.earthwork.total_cut_volume_cy if self.earthwork else 0.0

    @property
    def total_fill_volume_cy(self):
        return self.earthwork.total_fill_volume_cy if self.earthwork else 0.0


class VectorPipeline:

    def __init__(self, bezier_steps=8, offset_interval=1.0,
                 interpolation_method="linear", output_dir="reports"):
        self.splitter = CrossSectionSplitter()
        self.extractor = GeometryExtractor(bezier_steps=bezier_steps)
        self.identifier = ProfileIdentifier()
        self.transformer = CoordinateTransformer()
        self.normalizer = ProfileNormalizer(interval=offset_interval, method=interpolation_method)
        self.calculator = EarthworkCalculator(offset_interval=offset_interval)
        self.validator = Validator()
        self.reporter = ReportGenerator(output_dir=output_dir)

    def run(self, pdf_path, page_numbers=None, progress_callback=None):
        result = PipelineResult(pdf_path)

        # step 1: classify pages
        if progress_callback:
            progress_callback(0, 1, "Analyzing PDF structure...")

        classifier = PDFClassifier(pdf_path)
        doc_analysis = classifier.analyze()
        result.document_analysis = doc_analysis

        # filter to requested pages
        if page_numbers:
            targets = [p for p in doc_analysis.pages if p.page_number in page_numbers]
        else:
            targets = doc_analysis.processable_pages

        if not targets:
            return result

        # step 2-3: process each page → split → extract per station
        doc = fitz.open(pdf_path)
        total_pages = len(targets)
        all_station_areas = []

        for idx, page_info in enumerate(targets):
            if progress_callback:
                progress_callback(idx, total_pages, f"Processing page {page_info.page_number}...")

            page = doc[page_info.page_number - 1]
            dwg = page_info.drawing_number or ""

            # split page into cross-section regions
            regions = self.splitter.split(page, page_info.page_number, dwg)

            if not regions:
                sr = StationResult("N/A", 0, page_info.page_number)
                sr.drawing_number = dwg
                sr.error = "No cross-section regions found on this page."
                sr.stage_reached = "splitting"
                result.station_results.append(sr)
                continue

            # process each region
            for region in regions:
                sr = self._process_region(page, region)
                result.station_results.append(sr)

                if sr.success and sr.station_area:
                    all_station_areas.append(sr.station_area)

        doc.close()

        # step 4: compute volumes across all stations
        if progress_callback:
            progress_callback(total_pages, total_pages, "Computing volumes...")

        if len(all_station_areas) >= 2:
            result.earthwork = self.calculator.compute_volumes(all_station_areas)

            # generate aggregate reports
            ok_stations = result.successful_stations
            if ok_stations:
                # find any normalized profile for the plot (use first one)
                first_norm = next((s.normalized for s in ok_stations if s.normalized), None)
                if first_norm and result.earthwork:
                    result.plot_path = self.reporter.plot_cross_sections(
                        ok_stations, result.earthwork, "All Stations"
                    )
                    validation = self.validator.validate_cross_sections(
                        result.earthwork, ok_stations
                    )
                    result.json_report_path = self.reporter.save_json(
                        result.earthwork, validation, "aggregate"
                    )
                    result.csv_report_path = self.reporter.save_csv(
                        result.earthwork, "aggregate"
                    )
        elif len(all_station_areas) == 1:
            result.earthwork = self.calculator.compute_volumes(all_station_areas)

        if progress_callback:
            progress_callback(total_pages, total_pages, "Done!")

        return result

    def _process_region(self, page, region):
        """Process one cross-section region → StationResult."""
        sr = StationResult(region.station_label, region.station_ft, region.page_number)
        sr.drawing_number = region.drawing_number
        sr.region = region

        try:
            pw = page.rect.width
            region_h = region.y_bottom - region.y_top

            # extract geometry within this region's Y bounds
            sr.stage_reached = "extraction"
            paths = self.extractor.extract_region(page, region.y_top, region.y_bottom)
            if not paths:
                sr.error = f"No geometry found in region y=[{region.y_top:.0f}, {region.y_bottom:.0f}]"
                return sr

            # identify existing/proposed profiles
            sr.stage_reached = "identification"
            profiles = self.identifier.identify(paths, pw, region_h)
            sr.profiles = profiles

            if not profiles.existing_ground:
                sr.error = "Could not identify Existing Ground profile."
                return sr
            if not profiles.proposed_grade:
                sr.error = "Could not identify Proposed Grade profile."
                return sr

            # build coordinate scale from text labels in this region
            sr.stage_reached = "transformation"
            scale = self.transformer.build_scale(page, region)
            sr.scale = scale

            ex_t = self.transformer.transform(profiles.existing_ground_points, scale, "existing")
            pr_t = self.transformer.transform(profiles.proposed_grade_points, scale, "proposed")
            sr.existing_transformed = ex_t
            sr.proposed_transformed = pr_t

            if len(ex_t.offsets) < 2:
                sr.error = "Existing ground has too few points after transformation."
                return sr
            if len(pr_t.offsets) < 2:
                sr.error = "Proposed grade has too few points after transformation."
                return sr

            # normalize to common offset grid
            sr.stage_reached = "normalization"
            normalized = self.normalizer.normalize(ex_t, pr_t)
            sr.normalized = normalized

            # compute cut/fill area at this station
            sr.stage_reached = "calculation"
            station_area = self.calculator.compute_area(
                normalized, region.station_label, region.station_ft, region.page_number
            )
            sr.station_area = station_area

            sr.success = True
            sr.stage_reached = "complete"

        except Exception as e:
            sr.error = f"[{sr.stage_reached}] {type(e).__name__}: {e}"

        return sr