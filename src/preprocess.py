import re
import html

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('\\', ' ')
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.lower().split()
    return ' '.join([t for t in tokens if len(t) > 2])
