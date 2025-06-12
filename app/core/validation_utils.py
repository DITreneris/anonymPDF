"""
Validation utilities for PII detection accuracy improvements.

This module contains validation functions and exclusion lists to reduce
false positives and improve the precision of PII detection.
"""

import re
import logging
import yaml
from pathlib import Path
from typing import List, Set, Dict, Any

# Use standard logging instead of structlog for broader compatibility
validation_logger = logging.getLogger(__name__)

# Geographic terms that should not be classified as personal names
GEOGRAPHIC_EXCLUSIONS = {
    # Countries and territories
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia", "Bulgaria", 
    "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", 
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", 
    "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", 
    "Montenegro", "Netherlands", "Norway", "Poland", "Portugal", "Romania", 
    "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", 
    "Ukraine", "Vatican", "Gibraltar", "Isle", "Jersey", "Guernsey",
    
    # States and regions  
    "United", "Kingdom", "Great", "Britain", "Northern", "Ireland", "Scotland", 
    "Wales", "England", "California", "Texas", "Florida", "York", "Illinois", "United Kingdom",
    
    # Continents and geographic regions
    "Europe", "Asia", "Africa", "America", "Australia", "Antarctica", 
    "European", "Union", "Baltic", "Scandinavia", "Mediterranean",
    
    # Lithuanian regions and international
    "Europoje", "Europos", "Sąjungos", "Baltijos", "Šiaurės", "Rytų", "Vakarų",
    "Pietų", "Vidurio", "Respublikos", "Respublika", "Lietuvos", "Latvijos", 
    "Estijos", "Lenkijos", "Rusijos", "Baltarusijos", "Ukrainos", "Vilnius", "Kaunas",
}

# Document terminology that should not be classified as personal names
DOCUMENT_TERMS = {
    # Generic document terms
    "Document", "Certificate", "Policy", "Agreement", "Contract", "Report", 
    "Statement", "Invoice", "Receipt", "Form", "Application", "Declaration",
    "Registration", "License", "Permit", "Warranty", "Terms", "Conditions",
    "Summary", "Details", "Information", "Data", "Record", "File", "Number",
    "Code", "Reference", "Identification", "Identity", "Personal", "Business",
    "Company", "Organization", "Institution", "Department", "Office", "Agency",
    "Ministry", "Government", "Authority", "Service", "Center", "Unit", "Branch", "Card",
    
    # Lithuanian document terms
    "Dokumentas", "Pažymėjimas", "Polisas", "Sutartis", "Ataskaita", "Pareiškimas",
    "Registracija", "Licencija", "Leidimas", "Garantija", "Sąlygos", "Informacija",
    "Duomenys", "Įrašas", "Failas", "Numeris", "Kodas", "Nuoroda", "Tapatybė",
    "Asmens", "Verslo", "Įmonės", "Organizacijos", "Institucijos", "Skyrius",
    "Biuras", "Agentūra", "Ministerija", "Valdžia", "Tarnyba", "Centras",
    "Padalinys", "Filialas", "Privalomojo", "Draudimo", "Civile", "Atsakomybe",
    "Transporto", "Priemone", "Numeris", "When", "Where", "What", "Which", "Who",
    "Pavadinimas", "Liudijimas", "Arba", "Metų", "Mėnesio",
    
    # Insurance specific terms
    "Insurance", "Insurer", "Insured", "Coverage", "Premium", "Claim", "Liability",
    "Motor", "Vehicle", "Auto", "Property", "Health", "Life", "Travel", "Standard",
    "Third", "Party", "Comprehensive", "Collision", "Deductible", "Excess",
    "Draudimas", "Draudėjas", "Draudimo", "Draustojas", "Draudžiamasis", "Polisas",
    "Išmoka", "Žala", "Atsakomybė", "Automobilio", "Turto", "Sveikatos", "Gyvybės",
    "Kelionių", "Standartinis", "Trečiojo", "Asmens", "Visapusiškas", "Išskaita",
}

# Lithuanian common words that might match SWIFT/BIC patterns
LITHUANIAN_COMMON_WORDS = {
    "PRIVALOMOJO", "DRAUDIMO", "SUTARTIS", "LIETUVOS", "RESPUBLIKA", "BENDROVĖ",
    "UŽDAROJI", "AKCINĖ", "VIEŠOJI", "MAŽOJI", "INDIVIDUALI", "VEIKLA", "ĮMONĖ",
    "VALDYBA", "DIREKTOR", "ATSTOVAS", "PARTNERIS", "DARBUOTOJAS", "KLIENTAS",
    "PASLAUGOS", "PRODUKTAS", "SISTEMA", "TECHNOLOG", "ELEKTRONIKA", "KOMPIUTER",
    "PROGRAMOS", "DUOMENYS", "SAUGUMAS", "PRIVATUMAS", "KONFIDENCIAL", "SLAPTAS",
    "OFICIALUS", "DOKUMENTAS", "PAŽYMĖJIMAS", "SERTIFIKAT", "REGISTRAC", "LICENCIJ",
    "LEIDIMAS", "TEISĖS", "PAREIGOS", "ATSAKOMYB", "GARANTIJA", "SĄLYGOS", "NUOSTAT",
}

# Common ALL CAPS phrases that are not PII but might be caught by name patterns
COMMON_ALL_CAPS_NON_NAMES = {
    "LENGVASIS AUTOMOBILIS",
    "KROVININIS AUTOMOBILIS",
    "TECHNINĖ APŽIŪRA",
    "GALIOJA IKI",
    "ASMUO KONTAKTAI",
    "VARIKLIO GALIA",
    "SĖDIMŲ VIETŲ",
    "STOVIMŲ VIETŲ",
    "PRIEKINIAI ŽIBINTAI",
    "GALINIAI ŽIBINTAI",
    "DUOMENYS NEIDENTIFIKUOTI",
    "ŽALIOJI KORTELĖ",
    "TRANSPORTO PRIEMONĖS REGISTRACIJOS NUMERIS",
    "TECHNINĖS APŽIŪROS REZULTATŲ ATASKAITA",
    "VALSTYBINIS NUMERIS",
    "DRAUDIMO POLISAS",
    "AUTOMOBILIO REGISTRACIJOS NUMERIS",
    "ARBA AUTOBUSAS", # From KROVININIS AUTOMOBILIS ARBA AUTOBUSAS
    "MOTOCIKLAS DVIRATIS",
    "VARIKLIU PRIEKABA",
    "KITI AUTOBUSAS",
    "PRIEKABA DVIRATIS",
    "VARIKLIU MOTOCIKLAS",
    # Add more common phrases as identified
}

# Brand names and product identifiers that should be preserved
BRAND_NAMES = {}

def load_brand_names() -> Dict[str, Any]:
    """Load brand names from configuration file."""
    try:
        brand_config_path = Path("config/brand_names.yaml")
        if brand_config_path.exists():
            with open(brand_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        else:
            validation_logger.warning("Brand names config file not found, using empty brand list")
            return {}
    except Exception as e:
        validation_logger.error(f"Error loading brand names config: {e}")
        return {}

# Load brand names on module import
BRAND_NAMES = load_brand_names()

def get_all_brand_names() -> Set[str]:
    """Get all brand names as a flat set for efficient lookup."""
    all_brands = set()
    
    if not BRAND_NAMES:
        return all_brands
    
    try:
        # Automotive brands
        automotive = BRAND_NAMES.get('automotive_brands', {})
        all_brands.update(automotive.get('manufacturers', []))
        all_brands.update(automotive.get('models', []))
        
        # Technology brands
        technology = BRAND_NAMES.get('technology_brands', {})
        all_brands.update(technology.get('companies', []))
        all_brands.update(technology.get('products', []))
        
        # Financial institutions
        financial = BRAND_NAMES.get('financial_institutions', {})
        all_brands.update(financial.get('lithuanian_banks', []))
        all_brands.update(financial.get('international_banks', []))
        
        # Insurance companies
        insurance = BRAND_NAMES.get('insurance_companies', {})
        all_brands.update(insurance.get('lithuanian_insurers', []))
        all_brands.update(insurance.get('international_insurers', []))
        
        # Business terms
        business = BRAND_NAMES.get('business_terms', {})
        all_brands.update(business.get('general_business', []))
        all_brands.update(business.get('lithuanian_business', []))
        
        # Product categories
        products = BRAND_NAMES.get('product_categories', {})
        all_brands.update(products.get('automotive_services', []))
        all_brands.update(products.get('common_products', []))
        
    except Exception as e:
        validation_logger.error(f"Error processing brand names: {e}")
    
    return all_brands

def is_brand_name(text: str) -> bool:
    """
    Check if text is a known brand name or product identifier.
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if text is a recognized brand name
    """
    if not text or len(text.strip()) < 2:
        return False
    
    # Get validation rules
    validation_rules = BRAND_NAMES.get('validation_rules', {})
    case_sensitive = validation_rules.get('case_sensitive', False)
    
    # Normalize text for comparison
    text_normalized = text.strip()
    if not case_sensitive:
        text_normalized = text_normalized.lower()
    
    # Get all brand names
    all_brands = get_all_brand_names()
    
    # Create normalized brand set based on case sensitivity setting
    if case_sensitive:
        brand_set = all_brands
    else:
        brand_set = {brand.lower() for brand in all_brands}
    
    # Check exact match
    if text_normalized in brand_set:
        validation_logger.debug(
            "Brand name detected - preserving",
            text=text,
            reason="exact_brand_match"
        )
        return True
    
    # Check if any word in the text is a brand name
    words = text_normalized.split()
    for word in words:
        if word in brand_set and len(word) >= validation_rules.get('minimum_length', 2):
            validation_logger.debug(
                "Brand name word detected - preserving",
                text=text,
                brand_word=word,
                reason="brand_word_match"
            )
            return True
    
    return False

def validate_organization_with_brand_context(text: str, context: str = "") -> bool:
    """
    Enhanced organization validation that considers brand names.
    
    Args:
        text: The detected text to validate
        context: Surrounding text context
        
    Returns:
        bool: True if should be redacted (not a brand), False if should be preserved (is a brand)
    """
    # First check if it's a known brand name
    if is_brand_name(text):
        validation_logger.info(
            "Organization preserved as brand name",
            text=text,
            reason="brand_name_preservation"
        )
        return False  # Don't redact brand names
    
    # Apply existing organization validation
    return validate_organization_name(text)

# Validation functions
def validate_person_name(text: str, context: str = "") -> bool:
    """
    Validate if detected text is likely a real person name.
    
    Args:
        text: The detected text to validate
        context: Surrounding text context for additional validation
        
    Returns:
        bool: True if likely a person name, False otherwise
    """
    text_stripped = text.strip()

    # Basic length checks
    if not text_stripped or len(text_stripped) < 2 or len(text_stripped) > 100: # Min 2 chars, Max 100 chars
        validation_logger.info(
            "Name validation failed: length criteria",
            text=text,
            length=len(text_stripped),
            reason="invalid_length"
        )
        return False

    # Check for presence of digits - names usually don't have digits
    if any(char.isdigit() for char in text_stripped):
        validation_logger.info(
            "Name validation failed: contains digits",
            text=text,
            reason="contains_digits"
        )
        return False

    # Normalize for case-insensitive checks against exclusion lists
    # For exclusion lists, we often want to match regardless of original casing,
    # but for name structure (e.g. title case), original or specific casing matters.
    text_normalized_for_exclusions = text_stripped.title() # Use Title Case for matching exclusions like "Vilnius"
    
    # Check against geographic exclusions (both full text and individual words)
    if text_normalized_for_exclusions in GEOGRAPHIC_EXCLUSIONS:
        validation_logger.info(
            "Name validation failed: geographic term (full match)",
            text=text,
            normalized_for_exclusions=text_normalized_for_exclusions,
            reason="geographic_exclusion_full"
        )
        return False
    
    words_for_exclusion_check = text_normalized_for_exclusions.split()
    for word in words_for_exclusion_check:
        if word in GEOGRAPHIC_EXCLUSIONS: # Assumes GEOGRAPHIC_EXCLUSIONS are TitleCased or proper nouns
            validation_logger.info(
                "Name validation failed: geographic term (word match)",
                text=text,
                matched_word=word,
                reason="geographic_exclusion_word"
            )
            return False

    # Check against document terms (both full text and individual words)
    # For document terms, many are common nouns, so checking lower/title might be needed.
    # Assuming DOCUMENT_TERMS contains terms in formats they might appear or normalized (e.g. Title or UPPER)
    text_lower = text_stripped.lower() # For matching common document terms that might be lowercase
    
    # Check full text against DOCUMENT_TERMS (try lower and title case of text)
    if text_lower in DOCUMENT_TERMS or text_normalized_for_exclusions in DOCUMENT_TERMS:
        validation_logger.info(
            "Name validation failed: document term (full match)",
            text=text,
            reason="document_term_full"
        )
        return False

    # Check individual words against DOCUMENT_TERMS
    # Consider the case of words in DOCUMENT_TERMS. If they are stored as TitleCase:
    normalized_words = words_for_exclusion_check # These are already .title().split()
    # If DOCUMENT_TERMS might also contain lowercase common words, then:
    # lower_case_words = text_lower.split()

    for word in normalized_words: # Matching title-cased words from text against DOCUMENT_TERMS
        if word in DOCUMENT_TERMS:
            validation_logger.info(
                "Name validation failed: document term (word match on title-cased word)",
                text=text,
                matched_word=word,
                reason="document_term_word"
            )
            return False
    
    # Additionally, check if lowercase words from text match DOCUMENT_TERMS, if terms are stored as lowercase
    # This depends on how DOCUMENT_TERMS are populated. For now, rely on title case matching above.
    # Example if DOCUMENT_TERMS are mostly lowercase:
    # for word_lower in text_lower.split():
    #    if word_lower in DOCUMENT_TERMS:
    #        validation_logger.info(f"Name validation failed for '{text}': document term (word_lower_match: {word_lower})")
    #        return False


    # Basic name structure: should not be all lowercase if multiple words
    if len(words_for_exclusion_check) > 1 and text_stripped.islower():
        validation_logger.info(
            "Name validation failed: multiple words, all lowercase",
            text=text,
            reason="all_lowercase_multi_word"
        )
        return False

    # Check for too many special characters (allowing spaces, hyphens, apostrophes)
    allowed_chars = set("'- ") # Space, hyphen, apostrophe
    num_special = 0
    for char in text_stripped:
        if not char.isalnum() and char not in allowed_chars:
            num_special +=1
    
    # Allow a small number of other special characters (e.g. dots in Dr. or middle names)
    # but if it's excessive, it's likely not a name.
    # Heuristic: if more than 2 such special chars, or if name is short and has them.
    if num_special > 2 or (num_special > 0 and len(text_stripped) < 5):
         validation_logger.info(
            "Name validation failed: excessive or invalid special characters",
            text=text,
            num_special=num_special,
            reason="excessive_special_chars"
        )
         return False
    
    # If it looks like a sentence fragment or common phrase (e.g. ends with common punctuation not typical for names)
    # This is a simple check, more sophisticated NLP would be needed for robust sentence detection.
    if text_stripped.endswith((",", ".", ";", ":", "?", "!")) and len(text_stripped) > 5 : # Check for common sentence-ending punctuation
         # Further check if this is not a common abbreviation like "Dr." or "Jr."
        parts = text_stripped.split()
        if not any(part.endswith('.') and len(part) <=3 for part in parts): # Allow "Dr.", "Mr." etc.
            validation_logger.info(
                "Name validation failed: looks like sentence fragment due to ending punctuation",
                text=text,
                reason="ends_with_punctuation"
            )
            return False


    # TODO: Add more sophisticated checks if needed, e.g.,
    # - List of common non-name words (beyond document/geo terms)
    # - Check capitalization patterns (e.g. "John Doe" is good, "john doe" is suspect, "JOHN DOE" is suspect unless short)
    # - Contextual analysis (e.g., preceded by "Name:", "Vardas:")

    validation_logger.debug(f"Name validation tentatively passed for '{text}'")
    return True


def validate_swift_bic(text: str) -> bool:
    """
    Validate if detected text is a valid SWIFT/BIC code.
    
    Args:
        text: The detected text to validate
        
    Returns:
        bool: True if likely a valid SWIFT/BIC code, False otherwise
    """
    text_upper = text.strip().upper()

    if text_upper in LITHUANIAN_COMMON_WORDS:
        validation_logger.info("SWIFT/BIC validation failed: Lithuanian common word", text=text, reason="lithuanian_word")
        return False

    # Length validation - SWIFT codes are 8 or 11 characters
    if len(text_upper) not in [8, 11]:
        validation_logger.info("SWIFT/BIC validation failed: invalid length", text=text, length=len(text_upper), reason="invalid_length")
        return False

    bank_code = text_upper[0:4]
    country_code = text_upper[4:6]
    location_code = text_upper[6:8]

    if not bank_code.isalpha():
        validation_logger.info("SWIFT/BIC validation failed: invalid bank code", text=text, bank_code=bank_code, reason="invalid_bank_code")
        return False

    if not country_code.isalpha():
        validation_logger.info("SWIFT/BIC validation failed: invalid country code", text=text, country_code=country_code, reason="invalid_country_code")
        return False

    # Location code validation (positions 7-8)
    if not location_code.isalnum(): # Must be 2 alphanumeric characters
        validation_logger.info("SWIFT/BIC validation failed: location code not alphanumeric", text=text, location_code=location_code, reason="invalid_location_code_format")
        return False
    
    # Specific rules for location code based on ISO 9362
    # If first char of location code is '0', second must be a digit.
    if location_code[0] == '0' and not location_code[1].isdigit():
        validation_logger.info("SWIFT/BIC validation failed: invalid location code starting with 0", text=text, location_code=location_code, reason="invalid_location_code_zero")
        return False
    # If first char of location code is '1', second must be a digit.
    if location_code[0] == '1' and not location_code[1].isdigit():
        validation_logger.info("SWIFT/BIC validation failed: invalid location code starting with 1", text=text, location_code=location_code, reason="invalid_location_code_one")
        return False

    # If 11 characters, last 3 should be branch code (alphanumeric)
    if len(text_upper) == 11:
        branch_code = text_upper[8:11]
        if not branch_code.isalnum():
            validation_logger.info("SWIFT/BIC validation failed: invalid branch code", text=text, branch_code=branch_code, reason="invalid_branch_code")
            return False
            
    return True


def validate_organization_name(text: str) -> bool:
    """
    Validate if detected text is likely a real organization name.
    
    Args:
        text: The detected text to validate
        
    Returns:
        bool: True if likely an organization name, False otherwise
    """
    # Convert to title case for comparison
    text_normalized = text.strip().title()
    
    # Check if the ENTIRE name is a geographic exclusion
    if text_normalized in GEOGRAPHIC_EXCLUSIONS:
        validation_logger.info(
            "Organization validation failed: name is a geographic term",
            text=text,
            reason="full_geographic_exclusion"
        )
        return False
    
    # The original check for individual words from GEOGRAPHIC_EXCLUSIONS has been removed
    # as it was too strict (e.g., "UAB Lietuvos Technologijos" was failing).

    # Very short organization names are likely false positives
    if len(text.strip()) <= 2:
        validation_logger.info(
            "Organization validation failed: too short",
            text=text,
            reason="too_short"
        )
        return False
    
    return True


def deduplicate_detections(detections: Dict[str, List[tuple]]) -> Dict[str, List[tuple]]:
    """
    Remove duplicate detections and prioritize more specific patterns.
    
    Args:
        detections: Dictionary of detection categories and their matches
        
    Returns:
        Dict: Deduplicated detections with priority given to specific patterns
    """
    # Track all detected text positions to avoid duplicates
    detected_positions = {}  # text -> (category, specificity_score)
    deduplicated = {}
    
    # Define specificity scores (higher = more specific/reliable)
    specificity_scores = {
        "lithuanian_personal_codes": 100,
        "lithuanian_vat_codes": 95,
        "emails": 90,
        "lithuanian_phones_compact": 85,
        "lithuanian_phones_generic": 83,
        "mobile_phones_prefixed": 80,
        "phones_international": 75,
        "phones": 70,
        "financial_enhanced": 65,
        "credit_cards": 60,
        "dates_yyyy_mm_dd": 55,
        "healthcare_medical": 50,
        "identity_documents": 45,
        "automotive": 40,
        "legal_entities": 35,
        "addresses_prefixed": 30,
        "locations": 25,
        "names": 26,
        "organizations": 20,
        "eleven_digit_numerics": 10,
        "ssns": 5,
    }
    
    # Initialize deduplicated structure
    for category in detections:
        deduplicated[category] = []
    
    # Process detections by specificity (highest first)
    sorted_categories = sorted(
        detections.keys(), 
        key=lambda x: specificity_scores.get(x, 0), 
        reverse=True
    )
    
    for category in sorted_categories:
        category_score = specificity_scores.get(category, 0)
        
        for text, detection_type in detections[category]:
            text_normalized = text.strip().lower()
            
            # Check if this text was already detected with higher specificity
            if text_normalized in detected_positions:
                existing_category, existing_score = detected_positions[text_normalized]
                if existing_score > category_score:
                    validation_logger.info(
                        "Duplicate detection removed",
                        text=text,
                        current_category=category,
                        existing_category=existing_category,
                        reason="lower_specificity"
                    )
                    continue
                else:
                    # Remove the less specific detection
                    for i, (existing_text, existing_type) in enumerate(deduplicated[existing_category]):
                        if existing_text.strip().lower() == text_normalized:
                            deduplicated[existing_category].pop(i)
                            validation_logger.info(
                                "Replacing lower specificity detection",
                                text=text,
                                old_category=existing_category,
                                new_category=category
                            )
                            break
            
            # Add this detection
            detected_positions[text_normalized] = (category, category_score)
            deduplicated[category].append((text, detection_type))
    
    # Log deduplication statistics
    original_count = sum(len(matches) for matches in detections.values())
    deduplicated_count = sum(len(matches) for matches in deduplicated.values())
    
    validation_logger.info(
        "Deduplication completed",
        original_detections=original_count,
        deduplicated_detections=deduplicated_count,
        removed_duplicates=original_count - deduplicated_count
    )
    
    return deduplicated


def validate_detection_context(text: str, context: str, detection_type: str) -> bool:
    """
    Validate detection based on surrounding context.
    
    Args:
        text: The detected text
        context: Surrounding text context
        detection_type: Type of detection being validated
        
    Returns:
        bool: True if detection is valid in context, False otherwise
    """
    context_lower = context.lower()
    text_lower = text.lower()
    
    # Context-specific validations
    if detection_type == "PERSON":
        # If surrounded by document structure terms, likely not a person name
        structure_indicators = [
            "section", "chapter", "article", "paragraph", "clause", "item",
            "skyrius", "straipsnis", "dalis", "punktas", "papunktis"
        ]
        
        if any(indicator in context_lower for indicator in structure_indicators):
            validation_logger.info(
                "Context validation failed: document structure",
                text=text,
                detection_type=detection_type,
                reason="document_structure"
            )
            return False
    
    elif detection_type == "ORG":
        # Very common words that spaCy might mistake for organizations
        common_false_orgs = [
            "when", "where", "what", "which", "who", "how", "why",
            "kada", "kur", "kas", "kuris", "kaip", "kodėl"
        ]
        
        if text_lower in common_false_orgs:
            validation_logger.info(
                "Context validation failed: common word",
                text=text,
                detection_type=detection_type,
                reason="common_word"
            )
            return False
    
    return True 