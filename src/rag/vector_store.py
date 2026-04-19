import os
import chromadb
from chromadb.utils import embedding_functions

# Keep db local in the project directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "knowledge_base")

class KnowledgeVectorStore:
    def __init__(self):
        # Initialize chroma client pointing to local directory
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="retention_strategies",
            embedding_function=self.embedding_fn
        )
        
    def load_documents(self):
        """Read markdown files from knowledge_base and load them into chroma if empty."""
        # Simple check to avoid re-adding documents if they already exist
        if self.collection.count() > 0:
            return
            
        docs = []
        ids = []
        
        for filename in os.listdir(KB_PATH):
            if filename.endswith(".md"):
                file_path = os.path.join(KB_PATH, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Simple chunking: split by markdown headers
                chunks = content.split("\n## ")
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    docs.append(f"## {chunk}" if i > 0 else chunk)
                    ids.append(f"{filename}_chunk_{i}")
                    
        if docs:
            self.collection.add(
                documents=docs,
                ids=ids
            )
            print(f"Loaded {len(docs)} strategy chunks into ChromaDB.")

    def search(self, query: str, n_results: int = 2) -> list:
        """Search the vector database for relevant strategies."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        return results['documents'][0]
