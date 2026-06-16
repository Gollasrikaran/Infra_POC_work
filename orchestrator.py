"""
Pipeline orchestrator — chains every module into one .run() call.
"""

import fitz

from pdf_classifier import PDFClassifier
from geometry_extractor import GeometryExtractor
from profile_identifier import ProfileIdentifier
from coordinate_transformer import CoordinateTransformer
from profile_normalizer import ProfileNormalizer
from earthwork_calculator import EarthworkCalculator
from validation import Validator, ReportGenerator


class PageResult:
    """Holds everything produced for one page."""

    def __init__(self, page_number):
        self.page_number = page_number
        self.page_label = ""
        self.classification = ""
        self.workflow = ""

        self.profiles = None
        self.scale = None
        self.existing_transformed = None
        self.proposed_transformed = None
        self.normalized = None
        self.earthwork = None
        self.validation = None

        self.plot_path = None
        self.json_report_path = None
        self.csv_report_path = None

        self.success = False
        self.error = None
        self.stage_reached = "init"


class PipelineResult:
    """Holds results for all pages + aggregate totals."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.document_analysis = None
        self.page_results = []
        self.total_cut_volume_cy = 0.0
        self.total_fill_volume_cy = 0.0

    @property
    def successful_pages(self):
        return [p for p in self.page_results if p.success]

    @property
    def failed_pages(self):
        return [p for p in self.page_results if not p.success]


class VectorPipeline:

    def __init__(self, bezier_steps=8, station_interval=1.0,
                 interpolation_method="linear", output_dir="reports"):
        self.extractor = GeometryExtractor(bezier_steps=bezier_steps)
        self.identifier = ProfileIdentifier()
        self.transformer = CoordinateTransformer()
        self.normalizer = ProfileNormalizer(interval=station_interval, method=interpolation_method)
        self.calculator = EarthworkCalculator()
        self.validator = Validator()
        self.reporter = ReportGenerator(output_dir=output_dir)

    def run(self, pdf_path, page_numbers=None, h_scale_override=None,
            v_scale_override=None, progress_callback=None):
        result = PipelineResult(pdf_path)

        if progress_callback:
            progress_callback(0, 1, "Analyzing PDF structure...")

        classifier = PDFClassifier(pdf_path)
        doc_analysis = classifier.analyze()
        result.document_analysis = doc_analysis

        if page_numbers:
            targets = [p for p in doc_analysis.pages if p.page_number in page_numbers]
        else:
            targets = doc_analysis.processable_pages

        if not targets:
            return result

        doc = fitz.open(pdf_path)
        total = len(targets)

        for idx, page_info in enumerate(targets):
            if progress_callback:
                progress_callback(idx + 1, total, f"Processing page {page_info.page_number}...")

            pr = self._process_page(doc, page_info, h_scale_override, v_scale_override)
            result.page_results.append(pr)

        doc.close()

        for pr in result.successful_pages:
            if pr.earthwork:
                result.total_cut_volume_cy += pr.earthwork.total_cut_volume_cy
                result.total_fill_volume_cy += pr.earthwork.total_fill_volume_cy

        return result

    def _process_page(self, doc, page_info, h_override, v_override):
        page_num = page_info.page_number
        label = f"Page {page_num}"
        if page_info.drawing_number:
            label += f" (Dwg {page_info.drawing_number})"

        pr = PageResult(page_num)
        pr.page_label = label
        pr.classification = page_info.classification
        pr.workflow = page_info.workflow

        try:
            page = doc[page_num - 1]
            pw, ph = page.rect.width, page.rect.height

            pr.stage_reached = "extraction"
            paths = self.extractor.extract(page)
            if not paths:
                pr.error = "No vector geometry found."
                return pr

            pr.stage_reached = "identification"
            profiles = self.identifier.identify(paths, pw, ph)
            pr.profiles = profiles

            if not profiles.existing_ground:
                pr.error = "Could not identify Existing Ground profile."
                return pr
            if not profiles.proposed_grade:
                pr.error = "Could not identify Proposed Grade profile."
                return pr

            pr.stage_reached = "transformation"
            scale = self.transformer.detect_scale(page, h_override, v_override)
            pr.scale = scale

            ex_t = self.transformer.transform(profiles.existing_ground_points, scale, "existing_ground")
            pr_t = self.transformer.transform(profiles.proposed_grade_points, scale, "proposed_grade")
            pr.existing_transformed = ex_t
            pr.proposed_transformed = pr_t

            if len(ex_t.stations) < 2:
                pr.error = "Existing ground has too few points after transformation."
                return pr
            if len(pr_t.stations) < 2:
                pr.error = "Proposed grade has too few points after transformation."
                return pr

            pr.stage_reached = "normalization"
            normalized = self.normalizer.normalize(ex_t, pr_t)
            pr.normalized = normalized

            pr.stage_reached = "calculation"
            earthwork = self.calculator.calculate(normalized)
            pr.earthwork = earthwork

            pr.stage_reached = "validation"
            validation = self.validator.validate(normalized, earthwork)
            pr.validation = validation

            pr.stage_reached = "reporting"
            pr.plot_path = self.reporter.plot_profiles(normalized, earthwork, label)
            pr.json_report_path = self.reporter.save_json(earthwork, validation, label)
            pr.csv_report_path = self.reporter.save_csv(earthwork, label)

            pr.success = True
            pr.stage_reached = "complete"

        except Exception as e:
            pr.error = f"[{pr.stage_reached}] {type(e).__name__}: {e}"

        return pr
