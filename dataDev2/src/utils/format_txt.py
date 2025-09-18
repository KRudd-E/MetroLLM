import re

def format_txt(text: str) -> str:
    
    text = text.strip()
    
    # Remove common unwanted phrases and symbols
    text = re.sub(r'(Copyright\s*©?|visit us @|Visit us @|©|®)', '', text)
    text = text.replace('""', '"')

    # Collapse all whitespace to single spaces early
    text = ' '.join(text.split())

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove large blocks of periods (both with and without spaces)
    text = re.sub(r'\.{3,}', ' ', text)               # ... style
    text = re.sub(r'(?:\s*\.\s*){3,}', ' ', text)     # . . . . style

    # Remove large blocks of underscores
    text = re.sub(r'_{3,}', ' ', text)

    # Remove badly encoded or stray symbols (but keep common punctuation)
    text = re.sub(r'[^\w\s,.!?;:\'\"-]', ' ', text)

    # Remove long numeric sequences
    text = re.sub(r'\b\d{6,}\b', '', text)

    # Remove URLs and email addresses
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)

    # Fix spaced-out letters: e.g., "I s t h i s" -> "Is this"
    def fix_spaced_words(match):
        return match.group(0).replace(' ', '')

    # Pattern: sequences of single letters with spaces between them
    text = re.sub(r'(?<=\b)(?:[A-Za-z]\s){2,}[A-Za-z](?=\b)', fix_spaced_words, text)

    # Remove excessive whitespace again
    text = re.sub(r'\s+', ' ', text)
    
    # Remove residual full stops
    text = text.replace('..', '.')
    text = text.replace('. .', '.')
    text = text.replace('. . ', '. ')
    text = text.replace(' . ', '. ')
    
    text = text.replace('- ', '')

    return text.strip()
