from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class MathRAGAgent:
    def __init__(self,
                 model_name: str = 'thenlper/gte-small',
                 index_file: str = 'vector_index.faiss',
                 metadata_file: str = 'chunks.pkl'):
        self.embed_model = SentenceTransformer(model_name)
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.chunks = []

    def create_embeddings(self, chunks: list[str]):
        # Encode all chunks
        embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        # Persist
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(chunks, f)
        self.chunks = chunks

    def load_embeddings(self):
        # Load existing index & metadata
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, 'rb') as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        # Embed query, search top_k contexts
        query_emb = self.embed_model.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in indices[0]]
