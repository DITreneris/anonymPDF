import spacy

# 1. SpaCy versija
print("spaCy version:", spacy.__version__)

# 2. Modelio krovimas
nlp = spacy.load("lt_core_news_sm")
print("Pipeline components:", nlp.pipe_names)

# 3. Paprastas testas NER
test_text = "UAB „Vilniaus vandenys“ direktorė Audronė Žukauskienė džiaugiasi."
doc = nlp(test_text)
print("Entities found:", [(ent.text, ent.label_) for ent in doc.ents])
