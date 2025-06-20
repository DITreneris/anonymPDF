import fitz  # PyMuPDF
from app.core.memory_utils import memory_optimized
from typing import List, Tuple, Dict
from collections import defaultdict


@memory_optimized(processing_mode="normal")
def redact_pdf(input_path: str, output_path: str, sensitive_words: list):
    """Redacts specified sensitive words from a PDF file using PyMuPDF.

    Args:
        input_path: Path to the original PDF file.
        output_path: Path to save the redacted PDF file.
        sensitive_words: A list of words to redact from the PDF.

    Returns:
        True if redaction was successful, False otherwise.
    """
    try:
        doc = fitz.open(input_path)
        for page in doc:
            for word in sensitive_words:
                areas = page.search_for(word)
                for area in areas:
                    page.add_redact_annot(area, text="[REDACTED]", fill=(0, 0, 0), text_color=(1, 1, 1))
            page.apply_redactions()
        doc.save(
            output_path, garbage=4, deflate=True, clean=True
        )  # Save with options for smaller size and clean
        return True
    except Exception as e:
        print(f"Error during PDF redaction: {e}")  # Added basic error logging
        return False

    def _consolidate_detections(self, initial_detections: Dict[str, List[Tuple]], full_text: str) -> Dict[str, List[Tuple[str, str]]]:
        final_detections = defaultdict(list)

        for category, detections in initial_detections.items():
            for text, start, end, source, confidence in detections:
                if self.contextual_validator:
                    context_result = self.contextual_validator.validate_with_context(
                        text, category, full_text, start, end
                    )
                    # If the adaptive pattern confirms a match, it can override the category
                    if context_result.is_valid and context_result.validated_category:
                        validated_category = context_result.validated_category
                        final_detections[validated_category].append(
                            (text, f"{context_result.validation_source}_CONF_{context_result.confidence:.2f}")
                        )
                        # Skip adding it to the original category to avoid duplicates
                        continue

                # If no validator or validation is uncertain, add the original detection
                final_detections[category].append((text, f"{source}_CONF_{confidence:.2f}"))

        return final_detections

        
