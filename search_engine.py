"""
search_engine.py - Egyptian Public Figures Search Engine
TF-IDF and BM25 ranking with built-in stopwords (no NLTK data download needed)
"""
print('code is running from search_enginge.py')
import json
import math
import pickle
import re
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','will','would','shall','should','may','might','must','can',
    'could','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','just','don','now','also','known',
    'one','two','three','many','well','including','became','become','becomes',
    'his','its','their','there','where','when','who','whom','which','whose',
    'born','died','served','made','had','has','have','been','was','were',
    'wrote','played','worked','won','considered','regarded','known',
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    # Simple suffix stripping (poor man's lemmatizer)
    result = []
    for t in tokens:
        if t.endswith('ing') and len(t) > 5: t = t[:-3]
        elif t.endswith('tion') and len(t) > 6: t = t[:-4]
        elif t.endswith('ies') and len(t) > 4: t = t[:-3] + 'y'
        elif t.endswith('es') and len(t) > 4: t = t[:-2]
        elif t.endswith('ed') and len(t) > 4: t = t[:-2]
        elif t.endswith('ers') and len(t) > 5: t = t[:-2]
        elif t.endswith('er') and len(t) > 4: t = t[:-2]
        elif t.endswith('ly') and len(t) > 4: t = t[:-2]
        result.append(t)
    return result


class TFIDFIndex:
    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.doc_tfidf: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []
        self.vocab: set = set()
        self.N: int = 0

    def fit(self, documents: List[str]) -> "TFIDFIndex":
        self.N = len(documents)
        tokenized = [tokenize(d) for d in documents]
        df: Dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1
        self.idf = {t: math.log((self.N + 1) / (df[t] + 1)) + 1 for t in df}
        self.vocab = set(self.idf.keys())
        self.doc_tfidf = []
        for tokens in tokenized:
            freq = Counter(tokens)
            max_f = max(freq.values(), default=1)
            tfidf = {}
            for term, f in freq.items():
                tf = f / max_f
                tfidf[term] = tf * self.idf.get(term, 0)
            self.doc_tfidf.append(tfidf)
        self.doc_norms = [
            math.sqrt(sum(v**2 for v in d.values())) or 1.0
            for d in self.doc_tfidf
        ]
        return self

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        q_freq = Counter(q_tokens)
        max_f = max(q_freq.values(), default=1)
        dot = 0.0; q_norm = 0.0
        for t, f in q_freq.items():
            q_tf = f / max_f
            q_w = q_tf * self.idf.get(t, 0)
            dot += q_w * self.doc_tfidf[doc_idx].get(t, 0)
            q_norm += q_w**2
        q_norm = math.sqrt(q_norm) or 1.0
        return dot / (q_norm * self.doc_norms[doc_idx])

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores = [(i, s) for i, s in scores if s > 0]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0

    def fit(self, documents: List[str]) -> "BM25Index":
        self.N = len(documents)
        tokenized = [tokenize(d) for d in documents]
        self.doc_lens = [len(t) for t in tokenized]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1
        df = defaultdict(int)
        self.doc_freqs = []
        for tokens in tokenized:
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for t in freq: df[t] += 1
        self.idf = {}
        for t, n in df.items():
            self.idf[t] = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
        return self

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        freq = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for t in q_tokens:
            if t not in self.idf: continue
            f = freq.get(t, 0)
            idf = self.idf[t]
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (num / den)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores = [(i, s) for i, s in scores if s > 0]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


class EgyptianFiguresSearchEngine:
    def __init__(self):
        self.corpus: List[Dict] = []
        self.tfidf = TFIDFIndex()
        self.bm25 = BM25Index()
        self._is_fitted = False

    def load_corpus(self, corpus_path: str) -> "EgyptianFiguresSearchEngine":
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        print(f"Loaded {len(self.corpus)} documents.")
        return self

    def fit(self) -> "EgyptianFiguresSearchEngine":
        if not self.corpus:
            raise ValueError("Corpus is empty.")
        texts = [doc["full_text"] for doc in self.corpus]
        self.tfidf.fit(texts)
        self.bm25.fit(texts)
        self._is_fitted = True
        print(f"Fitted on {len(texts)} docs. Vocab: {len(self.tfidf.vocab)} terms.")
        return self

    def search(self, query: str, method: str = "bm25", top_k: int = 10, alpha: float = 0.6) -> List[Dict]:
        if not self._is_fitted:
            raise RuntimeError("Engine not fitted.")
        N = len(self.corpus)
        if method == "tfidf":
            raw = self.tfidf.search(query, top_k=N)
            scored = [(i, s, 0.0) for i, s in raw]
        elif method == "bm25":
            raw = self.bm25.search(query, top_k=N)
            scored = [(i, 0.0, s) for i, s in raw]
        elif method == "hybrid":
            tfidf_raw = dict(self.tfidf.search(query, top_k=N))
            bm25_raw  = dict(self.bm25.search(query, top_k=N))
            all_ids = set(tfidf_raw) | set(bm25_raw)
            def normalize(d):
                if not d: return d
                mn, mx = min(d.values()), max(d.values())
                span = mx - mn or 1
                return {k: (v - mn) / span for k, v in d.items()}
            tfidf_n = normalize(tfidf_raw)
            bm25_n  = normalize(bm25_raw)
            scored = [(i, tfidf_n.get(i, 0.0), bm25_n.get(i, 0.0)) for i in all_ids]
        else:
            raise ValueError(f"Unknown method: {method}")

        results = []
        for idx, tfidf_s, bm25_s in scored:
            if method == "tfidf":   final = tfidf_s
            elif method == "bm25":  final = bm25_s
            else:                   final = alpha * bm25_s + (1 - alpha) * tfidf_s
            if final > 0:
                doc = self.corpus[idx].copy()
                doc["score"] = round(final, 4)
                doc["tfidf_score"] = round(tfidf_s, 4)
                doc["bm25_score"]  = round(bm25_s, 4)
                doc["method"] = method
                results.append(doc)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_snippet(self, text: str, query: str, window: int = 200) -> str:
        terms = tokenize(query)
        lower = text.lower()
        best_pos = 0
        for t in terms:
            pos = lower.find(t)
            if pos > 0:
                best_pos = max(0, pos - 60)
                break
        snippet = text[best_pos: best_pos + window]
        if best_pos > 0: snippet = "..." + snippet
        if best_pos + window < len(text): snippet += "..."
        return snippet

    def save(self, model_dir: str = "models") -> None:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "search_engine.pkl"), "wb") as f:
            pickle.dump(self, f)
        print(f"Saved to {model_dir}/search_engine.pkl")

    @classmethod
    def load(cls, model_dir: str = "models") -> "EgyptianFiguresSearchEngine":
        with open(os.path.join(model_dir, "search_engine.pkl"), "rb") as f:
            engine = pickle.load(f)
        print(f"Loaded from {model_dir}/search_engine.pkl")
        return engine

    def stats(self) -> Dict:
        return {
            "num_documents": len(self.corpus),
            "vocab_size": len(self.tfidf.vocab),
            "avg_doc_length": round(self.bm25.avgdl, 1),
            "fields": list({d["field"] for d in self.corpus}),
            "eras": list({d["era"] for d in self.corpus}),
        }
