import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_entities(text: str):
    """
    Simple NER proxy.

    Returns:
        set of entities
    """
    return set(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text))
