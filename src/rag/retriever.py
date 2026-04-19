from src.rag.vector_store import KnowledgeVectorStore

class StrategyRetriever:
    def __init__(self):
        # We handle importing and loading DB gracefully
        try:
            self.store = KnowledgeVectorStore()
            self.store.load_documents()
            self.enabled = True
        except Exception as e:
            print(f"RAG system disabled. Error initializing ChromaDB: {e}")
            self.enabled = False

    def get_context(self, query: str) -> str:
        """
        Takes a query (like the analyzer's output) and returns 
        relevant strategies from the knowledge base as a single string.
        """
        if not self.enabled:
            return "General gaming knowledge."
            
        try:
            # Query chroma
            docs = self.store.search(query, n_results=2)
            
            if not docs:
                return "No specific database strategies found."
                
            # Combine retrieved docs into a context string
            return "\n\n".join(docs)
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return "General gaming knowledge."
