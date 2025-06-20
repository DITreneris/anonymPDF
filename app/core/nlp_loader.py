import spacy
from spacy.language import Language
from app.core.logging import StructuredLogger

nlp_loader_logger = StructuredLogger("anonympdf.nlp_loader")

class NLPLoader:
    """Handles loading of spaCy NLP models."""
    _models: dict[str, Language] = {}

    def get_model(self, model_name: str) -> Language:
        """
        Loads a spaCy model if not already loaded.
        
        Args:
            model_name: The name of the spaCy model.
            
        Returns:
            The loaded spaCy Language object.
        """
        if model_name not in self._models:
            try:
                self._models[model_name] = spacy.load(model_name)
                nlp_loader_logger.info(f"SpaCy model loaded: {model_name}")
            except OSError:
                nlp_loader_logger.error(f"Could not load model {model_name}. Please download it.")
                raise
        return self._models[model_name]

    def get_english_model(self) -> Language:
        """Convenience method for the English model."""
        return self.get_model("en_core_web_sm")

    def get_lithuanian_model(self) -> Language:
        """Convenience method for the Lithuanian model."""
        return self.get_model("lt_core_news_sm") 