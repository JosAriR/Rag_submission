"""Student exercise: implement retrieval over the fictional clinic policies."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
DOCS = Path(__file__).parent / "knowledge_base"

def load_documents(directory=DOCS):
    """Return list of (filename, text) for *.txt files."""
    doc_list = []
    files = Path(directory)
    for f in sorted(directory.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        doc_list.append((f.name, text))
    return doc_list

def retrieve(question: str, k: int = 2, documents=None):
    """Return top-k (filename, text, score) matches using TF-IDF."""

    files = []
    texts = []
    if documents is None:
        documents = load_documents()

    if not documents or k <= 0:
        return []


    for i in documents:
        files.append(i[0])

    for t in documents:
        texts.append(t[1])

    vectorize = TfidfVectorizer()

    doc_vectors = vectorize.fit_transform(texts)
    question_vector = vectorize.transform([question])
    scores = cosine_similarity(question_vector, doc_vectors)[0]
    result = []
    for x, y in enumerate(scores):
        if y > 0:
            result.append((files[x], texts[x], float(y)))

    result.sort()
    return result[:k]


def build_context(results):
    context = []
    for file, text, _ in results:
        context.append(f"[Source: {file}]\n{text}")

    return "\n\n".join(context)