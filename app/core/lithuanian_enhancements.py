"""
Enhanced Lithuanian language support for Priority 2 improvements.

This module provides comprehensive Lithuanian language patterns,
expanded exclusion lists, and language-specific validation rules.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from app.core.logging import StructuredLogger
lithuanian_logger = StructuredLogger("anonympdf.lithuanian")


@dataclass
class LithuanianPattern:
    """Lithuanian-specific pattern with metadata."""
    pattern: str
    category: str
    confidence_modifier: float
    description: str
    examples: List[str]


class LithuanianLanguageEnhancer:
    """Enhanced Lithuanian language processing for PII detection."""
    
    def __init__(self):
        # Expanded Lithuanian exclusion lists
        self.lithuanian_geographic_terms = {
            # Major cities and their variations
            'Vilnius', 'Vilniaus', 'Vilniuje', 'Vilnių',
            'Kaunas', 'Kauno', 'Kaune', 'Kauną',
            'Klaipėda', 'Klaipėdos', 'Klaipėdoje', 'Klaipėdą',
            'Šiauliai', 'Šiaulių', 'Šiauliuose', 'Šiaulius',
            'Panevėžys', 'Panevėžio', 'Panevėžyje', 'Panevėžį',
            'Alytus', 'Alytaus', 'Alytuje', 'Alytų',
            'Marijampolė', 'Marijampolės', 'Marijampolėje', 'Marijampolę',
            'Mažeikiai', 'Mažeikių', 'Mažeikiuose', 'Mažeikius',
            'Jonava', 'Jonavos', 'Jonavoje', 'Jonavą',
            'Utena', 'Utenos', 'Utenoje', 'Uteną',
            
            # Administrative divisions
            'Vilniaus apskritis', 'Kauno apskritis', 'Klaipėdos apskritis',
            'Šiaulių apskritis', 'Panevėžio apskritis', 'Alytaus apskritis',
            'Marijampolės apskritis', 'Tauragės apskritis', 'Telšių apskritis',
            'Utenos apskritis',
            
            # Regions and areas
            'Aukštaitija', 'Žemaitija', 'Suvalkija', 'Dzūkija',
            'Baltijos jūra', 'Kuršių nerija', 'Nemunas', 'Neris',
            
            # International
            'Lietuva', 'Lietuvos', 'Lietuvoje', 'Lietuvą',
            'Europa', 'Europos', 'Europoje', 'Europą',
            'Baltijos šalys', 'Baltijos', 'Skandinavija',
            'Lenkija', 'Lenkijos', 'Latvija', 'Latvijos',
            'Estija', 'Estijos', 'Rusija', 'Rusijos',
            'Baltarusija', 'Baltarusijos', 'Ukraina', 'Ukrainos',
        }
        
        self.lithuanian_document_terms = {
            # Document types
            'Dokumentas', 'Dokumentai', 'Dokumento', 'Dokumentų',
            'Pažymėjimas', 'Pažymėjimai', 'Pažymėjimo', 'Pažymėjimų',
            'Liudijimas', 'Liudijimai', 'Liudijimo', 'Liudijimų',
            'Sertifikatas', 'Sertifikatai', 'Sertifikato', 'Sertifikatų',
            'Polisas', 'Polisai', 'Poliso', 'Polisų',
            'Sutartis', 'Sutartys', 'Sutarties', 'Sutarčių',
            'Ataskaita', 'Ataskaitos', 'Ataskaitų',
            'Pareiškimas', 'Pareiškimai', 'Pareiškimo', 'Pareiškimų',
            
            # Automotive and technical terms
            'Automobilis', 'Automobiliai', 'Automobilio', 'Automobilių',
            'Numeris', 'Numeriai', 'Numerio', 'Numerių',
            'Valstybinis', 'Valstybiniai', 'Valstybinio', 'Valstybinių',
            'Variklis', 'Varikliai', 'Variklio', 'Variklių',
            'Transportas', 'Transportai', 'Transporto', 'Transportų',
            
            # Insurance terms
            'Draudimas', 'Draudimai', 'Draudimo', 'Draudimų',
            'Draudėjas', 'Draudėjai', 'Draudėjo', 'Draudėjų',
            'Draustojas', 'Draustojai', 'Draustojo', 'Draustojų',
            'Draudžiamasis', 'Draudžiamieji', 'Draudžiamojo', 'Draudžiamųjų',
            'Privalomasis', 'Privalomieji', 'Privalomojo', 'Privalomųjų',
            'Civilė atsakomybė', 'Civilės atsakomybės',
            'Transporto priemonė', 'Transporto priemonės', 'Transporto priemonių',
            'Automobilis', 'Automobiliai', 'Automobilio', 'Automobilių',
            
            # Business terms
            'Įmonė', 'Įmonės', 'Įmonių',
            'Bendrovė', 'Bendrovės', 'Bendrovių',
            'Uždaroji akcinė bendrovė', 'UAB',
            'Akcinė bendrovė', 'AB',
            'Viešoji įstaiga', 'VšĮ',
            'Individuali veikla', 'IV',
            'Mažoji bendrija', 'MB',
            'Kooperatyvas', 'Kooperatyvai',
            
            # Legal terms
            'Įstatymas', 'Įstatymai', 'Įstatymo', 'Įstatymų',
            'Kodeksas', 'Kodeksai', 'Kodekso', 'Kodeksų',
            'Straipsnis', 'Straipsniai', 'Straipsnio', 'Straipsnių',
            'Punktas', 'Punktai', 'Punkto', 'Punktų',
            'Papunktis', 'Papunkčiai', 'Papunkčio', 'Papunkčių',
            'Skyrius', 'Skyriai', 'Skyriaus', 'Skyrių',
            'Dalis', 'Dalys', 'Dalies', 'Dalių',
            
            # Common question words and connectors
            'Kas', 'Kur', 'Kada', 'Kaip', 'Kodėl', 'Kiek', 'Kuris', 'Kuri', 'Kurie',
            'Ar', 'Bet', 'Ir', 'Arba', 'Taip', 'Ne', 'Taigi', 'Todėl',
            'Jei', 'Jeigu', 'Kai', 'Kol', 'Nes', 'Nors', 'Tačiau',
            
            # Time and date terms
            'Metai', 'Mėnuo', 'Diena', 'Valanda', 'Minutė', 'Sekundė',
            'Sausis', 'Vasaris', 'Kovas', 'Balandis', 'Gegužė', 'Birželis',
            'Liepa', 'Rugpjūtis', 'Rugsėjis', 'Spalis', 'Lapkritis', 'Gruodis',
            'Pirmadienis', 'Antradienis', 'Trečiadienis', 'Ketvirtadienis',
            'Penktadienis', 'Šeštadienis', 'Sekmadienis',
        }
        
        self.lithuanian_common_words = {
            # Very common words that might be misidentified
            'Asmuo', 'Asmenys', 'Asmens', 'Asmenų',
            'Vardas', 'Vardai', 'Vardo', 'Vardų',
            'Pavardė', 'Pavardės', 'Pavardžių',
            'Numeris', 'Numeriai', 'Numerio', 'Numerių',
            'Kodas', 'Kodai', 'Kodo', 'Kodų',
            'Duomenys', 'Duomenų',
            'Informacija', 'Informacijos',
            'Tekstas', 'Tekstai', 'Teksto', 'Tekstų',
            'Žodis', 'Žodžiai', 'Žodžio', 'Žodžių',
            'Sakinys', 'Sakiniai', 'Sakinio', 'Sakinių',
            'Puslapis', 'Puslapiai', 'Puslapio', 'Puslapių',
            'Eilutė', 'Eilutės', 'Eilučių',
            'Stulpelis', 'Stulpeliai', 'Stulpelio', 'Stulpelių',
            'Lentelė', 'Lentelės', 'Lentelių',
            'Paveikslas', 'Paveikslai', 'Paveikslo', 'Paveikslų',
            'Schema', 'Schemos', 'Schemų',
            'Diagrama', 'Diagramos', 'Diagramų',
        }
        
        # Enhanced Lithuanian patterns
        self.enhanced_lithuanian_patterns = {
            'lithuanian_name_with_title': LithuanianPattern(
                pattern=r'(?i)(?:Ponas|Ponia|Dr\.?|Prof\.?)\s+((?:[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+)(?:\s+[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+)?)',
                category='names',
                confidence_modifier=0.4,
                description='Lithuanian name with title (captures name only).',
                examples=['Ponas Jonas Petraitis', 'Ponia Žaneta Stankevičienė']
            ),
            'lithuanian_name_simple': LithuanianPattern(
                pattern=r'\b(([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]{3,})\s+([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]{3,}(?:ienė|aitė|ytė|utė|ūtė|as|is|ys|us|ius|ė|a)))\b(?!\s*\d)',
                category='names',
                confidence_modifier=0.3,
                description='A simple Lithuanian name (First Last), excluding patterns followed by numbers.',
                examples=['Linas Vaitkus', 'Rūta Vaitkienė']
            ),
            'lithuanian_address_prefixed': LithuanianPattern(
                pattern=r'(?i)(?:Adresas|Gyv\.\s*vieta|Adresas korespondencijai):\s*([A-ZĄČĘĖĮŠŲŪŽ][^,\n]+?g\.\s*\d+[^,]*)',
                category='addresses_prefixed',
                confidence_modifier=0.25,
                description='Lithuanian address (street and number only), non-greedy. Overwrites base pattern.',
                examples=['Adresas: Vilniaus g. 1,']
            ),
            'lithuanian_city_generic': LithuanianPattern(
                pattern=r'\b(Vilni(us|aus|uje|ų)|Kaun(as|o|e|ą)|Klaipėd(a|os|oje|ą)|Šiauli(ai|ų|uose|us)|Panevėž(ys|io|yje|į))\b',
                category='locations',
                confidence_modifier=0.9,
                description='Common Lithuanian city names with grammatical cases.',
                examples=['Vilniaus', 'Kaune', 'Klaipėdos', 'Šiaulius', 'Panevėžyje']
            ),
            'lithuanian_standalone_city': LithuanianPattern(
                pattern=r'\b(Vilnius|Kaunas|Klaipėda|Šiauliai|Panevėžys)\b',
                category='locations',
                confidence_modifier=0.95, # VERY HIGH confidence for standalone nominative case
                description='Standalone Lithuanian city names in nominative case.',
                examples=['Vilnius', 'Kaunas']
            ),
            'lithuanian_address_flexible': LithuanianPattern(
                pattern=r'([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+s?\s+(?:g\.|pr\.|al\.)\s*\d+(?:-\d+)?)',
                category='addresses_street',
                confidence_modifier=0.15,
                description='Flexible Lithuanian address pattern (street, number)',
                examples=['Vilniaus g. 1', 'Gedimino pr. 25-10A']
            ),
            'lithuanian_address_full': LithuanianPattern(
                pattern=r'([A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+s?\s+(?:g\.|pr\.|al\.)\s*\d+(?:-\d+)?,\s*LT-\d{5},?\s*[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+)',
                category='addresses_full',
                confidence_modifier=0.25,
                description='Full Lithuanian address with postal code and city',
                examples=['Paupio g. 50-136, LT-11341 Vilnius', 'Gedimino pr. 25, LT-01103, Vilnius']
            ),
            'lithuanian_company_full': LithuanianPattern(
                pattern=r'(UAB|AB|VšĮ|MB|IV)\s+"([^"]+)"',
                category='organizations',
                confidence_modifier=0.2,
                description='Lithuanian company with legal form',
                examples=['UAB "Lietuvos Technologijos"', 'AB "Swedbank"']
            ),
            'lithuanian_personal_code_labeled': LithuanianPattern(
                pattern=r'(?i:\(?\s*(?:asmens\s+kodas|a\.k\.?|AK)\s*:?\s*)\s*(\d{11})\s*\)?',
                category='lithuanian_personal_codes',
                confidence_modifier=0.3,
                description='Lithuanian personal code with label (case-insensitive)',
                examples=['Asmens kodas: 38901234567', '(a.k. 49012345678)', 'AK: 39012345678']
            ),
            'lithuanian_phone_formatted': LithuanianPattern(
                pattern=r'(?:Tel\.|Telefonas|Mob\.):\s*(\+370\s+\d{1,2}\s+\d{3}\s+\d{4,5})',
                category='lithuanian_phones_generic',
                confidence_modifier=0.2,
                description='Formatted Lithuanian phone number',
                examples=['Tel.: +370 6 123 4567', 'Mob.: +370 60 123 456']
            ),
            'lithuanian_email_labeled': LithuanianPattern(
                pattern=r'(?:El\.\s*paštas|Elektroninis\s+paštas):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                category='emails',
                confidence_modifier=0.2,
                description='Lithuanian email with label',
                examples=['El. paštas: jonas@example.lt', 'Elektroninis paštas: info@company.com']
            ),
            'lithuanian_bank_account': LithuanianPattern(
                pattern=r'(?:Banko\s+sąskaita|Sąskaitos\s+numeris):\s*(LT\d{2}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{4})',
                category='financial_enhanced',
                confidence_modifier=0.25,
                description='Lithuanian bank account with label',
                examples=['Banko sąskaita: LT12 3456 7890 1234 5678']
            ),
            'lithuanian_vat_labeled': LithuanianPattern(
                pattern=r'(?:PVM\s+kodas|PVM\s+Nr\.):\s*(LT\d{9,12})',
                category='lithuanian_vat_codes',
                confidence_modifier=0.25,
                description='Lithuanian VAT code with label',
                examples=['PVM kodas: LT100001738313', 'PVM Nr.: LT123456789']
            ),
            'lithuanian_date_full': LithuanianPattern(
                pattern=r'(\d{4}\s+m\.\s+(?:sausio|vasario|kovo|balandžio|gegužės|birželio|liepos|rugpjūčio|rugsėjo|spalio|lapkričio|gruodžio)\s+\d{1,2}\s+d\.)',
                category='dates_yyyy_mm_dd',
                confidence_modifier=0.15,
                description='Full Lithuanian date format',
                examples=['2024 m. sausio 15 d.', '2023 m. gruodžio 31 d.']
            )
        }
        
        # Explicitly remove the old, flawed pattern to avoid ambiguity.
        if 'lithuanian_address_street_only' in self.enhanced_lithuanian_patterns:
            del self.enhanced_lithuanian_patterns['lithuanian_address_street_only']

        # Compile patterns for performance
        self.compiled_patterns = {}
        for name, pattern_info in self.enhanced_lithuanian_patterns.items():
            self.compiled_patterns[name] = {
                'pattern': re.compile(pattern_info.pattern, re.IGNORECASE),
                'info': pattern_info
            }
    
    def is_lithuanian_geographic_term(self, text: str) -> bool:
        """Check if text is a Lithuanian geographic term."""
        text_normalized = text.strip().title()
        return text_normalized in self.lithuanian_geographic_terms
    
    def is_lithuanian_document_term(self, text: str) -> bool:
        """Check if text is a Lithuanian document term."""
        text_normalized = text.strip().title()
        return text_normalized in self.lithuanian_document_terms
    
    def is_lithuanian_common_word(self, text: str) -> bool:
        """Check if text is a common Lithuanian word."""
        text_normalized = text.strip().title()
        return text_normalized in self.lithuanian_common_words
    
    def validate_lithuanian_name(self, text: str, context: str = "") -> Tuple[bool, float]:
        """
        Validate if text is likely a Lithuanian person name.
        
        Args:
            text: Text to validate
            context: Surrounding context
            
        Returns:
            Tuple of (is_valid, confidence_modifier)
        """
        confidence_modifier = 0.0
        
        # Check for Lithuanian name patterns
        lithuanian_name_patterns = [
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+as$',  # Male names ending in -as
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+is$',  # Male names ending in -is
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+ė$',   # Female names ending in -ė
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+a$',   # Female names ending in -a
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+ienė$', # Married female surnames
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+aitė$', # Unmarried female surnames
            r'[A-ZĄČĘĖĮŠŲŪŽ][a-ząčęėįšųūž]+ytė$',  # Unmarried female surnames
        ]
        
        words = text.split()
        for word in words:
            for pattern in lithuanian_name_patterns:
                if re.match(pattern, word):
                    confidence_modifier += 0.1
                    lithuanian_logger.debug(
                        "Lithuanian name pattern matched",
                        word=word,
                        pattern=pattern
                    )
        
        # Check against exclusion lists
        if self.is_lithuanian_geographic_term(text):
            return False, -0.5
        
        if self.is_lithuanian_document_term(text):
            return False, -0.5
        
        if self.is_lithuanian_common_word(text):
            return False, -0.3
        
        # Check for title indicators in context
        title_patterns = [
            r'(?:Ponas|Ponia|Daktaras|Dr\.|Profesorius|Prof\.)\s+' + re.escape(text),
            re.escape(text) + r'\s+(?:gimęs|gimusi|born)',
            r'(?:Vardas|Name):\s*' + re.escape(text),
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                confidence_modifier += 0.2
                lithuanian_logger.debug(
                    "Lithuanian name context indicator found",
                    text=text,
                    pattern=pattern
                )
        
        return True, confidence_modifier
    
    def find_enhanced_lithuanian_patterns(self, text: str) -> List[Dict]:
        """Find all enhanced Lithuanian patterns in the text."""
        detections = []
        for name, compiled_pattern in self.compiled_patterns.items():
            pattern = compiled_pattern['pattern']
            info = compiled_pattern['info']
            
            for match in pattern.finditer(text):
                # If the pattern has a capturing group, use the group's content.
                # Otherwise, use the full match.
                matched_text = match.group(1) if match.groups() else match.group(0)
                
                # Filter out document/common terms for name patterns
                if info.category == 'names':
                    # Check if any part of the matched text is a document term
                    words = matched_text.split()
                    if any(self.is_lithuanian_document_term(word) or 
                           self.is_lithuanian_common_word(word) for word in words):
                        continue
                
                detection = {
                    'text': matched_text.strip(),
                    'category': info.category,
                    'start': match.start(1) if match.groups() else match.start(0),
                    'end': match.end(1) if match.groups() else match.end(0),
                    'confidence': info.confidence_modifier,
                    'pattern_name': name
                }
                detections.append(detection)
                
        return detections
    
    def get_lithuanian_exclusions(self) -> Dict[str, Set[str]]:
        """Get all Lithuanian exclusion lists."""
        return {
            'geographic_terms': self.lithuanian_geographic_terms,
            'document_terms': self.lithuanian_document_terms,
            'common_words': self.lithuanian_common_words
        }
    
    def validate_lithuanian_swift_bic(self, text: str) -> bool:
        """
        Validate SWIFT/BIC code specifically for Lithuanian context.
        
        Args:
            text: Text to validate
            
        Returns:
            True if likely a valid SWIFT/BIC code
        """
        # Use helper methods for exclusion, as they handle correct casing (title case)
        if self.is_lithuanian_common_word(text):
            lithuanian_logger.debug(f"SWIFT/BIC rejected (common word): {text}")
            return False
        
        if self.is_lithuanian_document_term(text):
            lithuanian_logger.debug(f"SWIFT/BIC rejected (document term): {text}")
            return False
        
        text_upper = text.strip().upper()
        
        # Standard SWIFT/BIC structure validation first
        if len(text_upper) not in [8, 11]:
            lithuanian_logger.debug(f"SWIFT/BIC rejected (invalid length): {text_upper}")
            return False
        
        if not text_upper[:4].isalpha(): # Bank code
            lithuanian_logger.debug(f"SWIFT/BIC rejected (invalid bank code format): {text_upper}")
            return False
        
        if not text_upper[4:6].isalpha() or text_upper[4:6] != 'LT': # Country code must be LT
            lithuanian_logger.debug(f"SWIFT/BIC rejected (invalid country code or not LT): {text_upper}")
            return False
        
        if not text_upper[6:8].isalnum(): # Location code
            lithuanian_logger.debug(f"SWIFT/BIC rejected (invalid location code format): {text_upper}")
            return False
        
        if len(text_upper) == 11 and not text_upper[8:11].isalnum(): # Optional branch code
            lithuanian_logger.debug(f"SWIFT/BIC rejected (invalid branch code format): {text_upper}")
            return False

        # Check against known Lithuanian bank codes (first 4 characters)
        # This list should contain only 4-character uppercase bank codes.
        lithuanian_bank_codes = {
            'CBVI',  # Citadele Bank (CBVILT2X)
            'AGBL',  # AB SEB bankas (AGBLLT2X)
            'HABA',  # Swedbank (HABALT22) - Note: Swedbank Lithuania is HABALT22, bank code HABA
            'REVO',  # Revolut (REVOLT21)
            'INDU',  # Šiaulių bankas (INDULT2X)
            'LCBA',  # Luminor Bank (formerly DNB, Nordea - e.g. AGBLLT2X before merge, now RIKOLT22, NDEALT2X. Luminor is now RIKOLT22. LCBA is not obvious for Luminor. Using RIKO for Luminor for now.)
                     # Let's use actual bank codes. For Luminor, it's RIKO (RIKOLT22).
            'RIKO',  # Luminor Bank AS Lithuanian branch (RIKOLT22)
            'LPSA'   # Lietuvos paštas (LPSALT21) - Example of another type
        }
        
        # If it's a structurally valid SWIFT/BIC with 'LT' and its bank code is known, it's very likely.
        if text_upper[:4] in lithuanian_bank_codes:
            lithuanian_logger.debug(f"SWIFT/BIC validated (known LT bank code): {text_upper}")
            return True
        
        # If it's a structurally valid SWIFT/BIC with 'LT' but unknown bank code,
        # it's still potentially valid as a Lithuanian SWIFT/BIC.
        # The main goal of this function is to weed out false positives like "DRAUDIMO".
        # The structural and 'LT' check already does a good job for that.
        lithuanian_logger.debug(f"SWIFT/BIC provisionally validated (structurally valid LT SWIFT/BIC): {text_upper}")
        return True


class LithuanianContextAnalyzer:
    """Specialized context analyzer for Lithuanian documents."""
    
    def __init__(self):
        self.language_enhancer = LithuanianLanguageEnhancer()
        
        # Lithuanian-specific document structure patterns
        self.lithuanian_document_sections = {
            'insurance_header': [
                r'PRIVALOMOJO.*DRAUDIMO.*PAŽYMĖJIMAS',
                r'DRAUDIMO.*POLISAS',
                r'CIVILINĖS.*ATSAKOMYBĖS.*DRAUDIMAS',
            ],
            'personal_info_section': [
                r'DRAUDĖJO.*DUOMENYS',
                r'ASMENS.*DUOMENYS',
                r'TRANSPORTO.*PRIEMONĖS.*DUOMENYS',
            ],
            'company_info_section': [
                r'ĮMONĖS.*DUOMENYS',
                r'DRAUSTOJAS',
                r'DRAUDIMO.*BENDROVĖ',
            ],
            'legal_section': [
                r'DRAUDIMO.*SĄLYGOS',
                r'TEISĖS.*IR.*PAREIGOS',
                r'ATSAKOMYBĖS.*RIBOJIMAS',
            ]
        }
        
        # Compile patterns
        self.compiled_lithuanian_sections = {}
        for section_type, patterns in self.lithuanian_document_sections.items():
            self.compiled_lithuanian_sections[section_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def identify_lithuanian_section(self, text: str, position: int, window_size: int = 150) -> Optional[str]:
        """
        Identify Lithuanian document section.
        
        Args:
            text: Full document text
            position: Character position to analyze
            window_size: Size of context window
            
        Returns:
            Lithuanian section type or None
        """
        # Extract context around the position
        start = max(0, position - window_size)
        end = min(len(text), position + window_size)
        context = text[start:end]
        
        # Check each Lithuanian section type
        for section_type, patterns in self.compiled_lithuanian_sections.items():
            for pattern in patterns:
                if pattern.search(context):
                    lithuanian_logger.debug(
                        "Lithuanian document section identified",
                        section=section_type,
                        position=position,
                        pattern=pattern.pattern
                    )
                    return section_type
        
        return None
    
    def calculate_lithuanian_confidence(self, detection: str, category: str, 
                                     context: str, section: Optional[str] = None) -> float:
        """
        Calculate confidence for Lithuanian-specific context.
        
        Args:
            detection: Detected text
            category: PII category
            context: Surrounding context
            section: Lithuanian document section
            
        Returns:
            Confidence modifier (-1.0 to 1.0)
        """
        confidence_modifier = 0.0
        
        # Section-specific adjustments
        if section:
            section_adjustments = {
                'insurance_header': -0.2,  # Headers often contain non-PII
                'personal_info_section': 0.3,  # Personal info sections likely contain PII
                'company_info_section': 0.2,  # Company sections may contain PII
                'legal_section': -0.3,  # Legal sections rarely contain actual PII
            }
            
            if section in section_adjustments:
                confidence_modifier += section_adjustments[section]
        
        # Category-specific Lithuanian validation
        if category == 'names':
            is_valid, name_modifier = self.language_enhancer.validate_lithuanian_name(detection, context)
            if not is_valid:
                confidence_modifier -= 0.5
            else:
                confidence_modifier += name_modifier
        
        elif category == 'swift_bic':
            if not self.language_enhancer.validate_lithuanian_swift_bic(detection):
                confidence_modifier -= 0.4
        
        return confidence_modifier 