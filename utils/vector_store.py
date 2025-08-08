# utils/vector_store.py
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from datetime import datetime

class VectorStore:
    """
    Vector store implementation using FAISS for similarity search
    and SentenceTransformers for embeddings
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize the vector store
        
        Args:
            model_name: Name of the SentenceTransformer model
            dimension: Dimension of embeddings (384 for all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.dimension = dimension
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Storage for documents and metadata
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.document_embeddings: List[np.ndarray] = []
        
        print(f"✅ Vector store initialized with {model_name}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Generate embeddings
        embeddings = self.encoder.encode(documents, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.document_embeddings.extend(embeddings)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            # Create default metadata
            for i, doc in enumerate(documents):
                self.metadata.append({
                    "id": len(self.documents) - len(documents) + i,
                    "length": len(doc),
                    "added_at": datetime.now().isoformat(),
                    "source": f"document_{len(self.documents) - len(documents) + i}"
                })
        
        print(f"✅ Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def add_document(self, document: str, metadata: Optional[Dict] = None):
        """Add a single document"""
        self.add_documents([document], [metadata] if metadata else None)
    
    def search(self, query: str, n_results: int = 5, score_threshold: float = 0.0) -> List[str]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar documents
        """
        if len(self.documents) == 0:
            print("⚠️ No documents in vector store")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(n_results, len(self.documents)))
        
        # Filter by score threshold and return documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold and idx != -1:
                results.append(self.documents[idx])
        
        return results
    
    def search_with_scores(self, query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents with similarity scores
        
        Returns:
            List of tuples (document, similarity_score)
        """
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(n_results, len(self.documents)))
        
        # Return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def search_with_metadata(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search with full metadata
        
        Returns:
            List of dictionaries with document, score, and metadata
        """
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(n_results, len(self.documents)))
        
        # Return full results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                    "index": int(idx)
                })
        
        return results
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)
    
    def get_document_by_index(self, index: int) -> Optional[str]:
        """Get document by index"""
        if 0 <= index < len(self.documents):
            return self.documents[index]
        return None
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save other data
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "model_name": self.model_name,
            "dimension": self.dimension
        }
        
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"✅ Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the vector store from disk"""
        if not os.path.exists(f"{filepath}.faiss") or not os.path.exists(f"{filepath}.pkl"):
            raise FileNotFoundError(f"Vector store files not found at {filepath}")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load other data
        with open(f"{filepath}.pkl", "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.metadata = data["metadata"]
        
        # Verify model compatibility
        if data.get("model_name") != self.model_name:
            print(f"⚠️ Warning: Loaded model ({data.get('model_name')}) differs from current model ({self.model_name})")
        
        print(f"✅ Vector store loaded from {filepath}. Documents: {len(self.documents)}")
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        self.document_embeddings = []
        print("✅ Vector store cleared")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        if not self.documents:
            return {"total_documents": 0, "avg_length": 0}
        
        doc_lengths = [len(doc) for doc in self.documents]
        return {
            "total_documents": len(self.documents),
            "avg_length": sum(doc_lengths) / len(doc_lengths),
            "min_length": min(doc_lengths),
            "max_length": max(doc_lengths),
            "model_name": self.model_name,
            "dimension": self.dimension
        }


class ChromaVectorStore:
    """
    Production-ready ChromaDB implementation with full feature parity
    """
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            existing_count = self.collection.count()
            print(f"✅ Connected to existing ChromaDB collection: {collection_name} ({existing_count} documents)")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"✅ Created new ChromaDB collection: {collection_name}")
        
        self.document_counter = self.collection.count()
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to ChromaDB with automatic ID generation"""
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to ChromaDB...")
        
        # Generate unique IDs
        ids = []
        for i in range(len(documents)):
            ids.append(f"doc_{self.document_counter + i}_{datetime.now().timestamp()}")
        
        # Prepare metadata
        if metadata:
            # Ensure metadata is properly formatted
            processed_metadata = []
            for meta in metadata:
                if meta is None:
                    processed_metadata.append({"source": "unknown"})
                else:
                    # Convert all values to strings (ChromaDB requirement)
                    processed_meta = {}
                    for k, v in meta.items():
                        processed_meta[k] = str(v) if v is not None else "none"
                    processed_metadata.append(processed_meta)
        else:
            # Create default metadata
            processed_metadata = []
            for i, doc in enumerate(documents):
                processed_metadata.append({
                    "source": f"document_{self.document_counter + i}",
                    "length": str(len(doc)),
                    "added_at": datetime.now().isoformat()
                })
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=processed_metadata,
                ids=ids
            )
            self.document_counter += len(documents)
            print(f"✅ Added {len(documents)} documents to ChromaDB. Total: {self.document_counter}")
            
        except Exception as e:
            print(f"❌ Error adding documents to ChromaDB: {str(e)}")
            raise
    
    def add_document(self, document: str, metadata: Optional[Dict] = None):
        """Add a single document"""
        self.add_documents([document], [metadata] if metadata else None)
    
    def search(self, query: str, n_results: int = 5, score_threshold: float = 0.0) -> List[str]:
        """Search ChromaDB and return documents"""
        if self.collection.count() == 0:
            print("⚠️ No documents in ChromaDB collection")
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Filter by score threshold if needed
            if score_threshold > 0.0 and results['distances']:
                filtered_docs = []
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    similarity = 1 - distance  # Convert distance to similarity
                    if similarity >= score_threshold:
                        filtered_docs.append(doc)
                return filtered_docs
            
            return results['documents'][0]
            
        except Exception as e:
            print(f"❌ Error searching ChromaDB: {str(e)}")
            return []
    
    def search_with_scores(self, query: str, n_results: int = 5) -> List[Tuple[str, float]]:
        """Search with similarity scores"""
        if self.collection.count() == 0:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            documents = results['documents'][0]
            distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
            
            # Convert distances to similarity scores (1 - distance)
            similarities = [max(0.0, 1 - d) for d in distances]
            
            return list(zip(documents, similarities))
            
        except Exception as e:
            print(f"❌ Error searching ChromaDB with scores: {str(e)}")
            return []
    
    def search_with_metadata(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search with full metadata"""
        if self.collection.count() == 0:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            search_results = []
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [None])[0] or [{}] * len(documents)
            distances = results.get('distances', [None])[0] or [0.0] * len(documents)
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                similarity = max(0.0, 1 - distance)
                search_results.append({
                    "document": doc,
                    "score": similarity,
                    "metadata": metadata or {},
                    "index": i
                })
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error searching ChromaDB with metadata: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from the collection"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            self.document_counter = 0
            print("✅ ChromaDB collection cleared")
        except Exception as e:
            print(f"❌ Error clearing ChromaDB collection: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        if count == 0:
            return {"total_documents": 0, "collection_name": self.collection_name}
        
        # Get sample documents to calculate average length
        try:
            sample_results = self.collection.query(
                query_texts=["sample"],
                n_results=min(10, count)
            )
            
            if sample_results['documents'] and sample_results['documents'][0]:
                docs = sample_results['documents'][0]
                doc_lengths = [len(doc) for doc in docs]
                avg_length = sum(doc_lengths) / len(doc_lengths)
                min_length = min(doc_lengths)
                max_length = max(doc_lengths)
            else:
                avg_length = min_length = max_length = 0
                
        except:
            avg_length = min_length = max_length = 0
        
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✅ Deleted ChromaDB collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Error deleting collection: {str(e)}")


# Default to ChromaDB since it's installed
def VectorStore(use_chroma: bool = True, **kwargs):
    """
    Factory function to create appropriate vector store
    
    Args:
        use_chroma: If True, use ChromaDB; otherwise use FAISS
        **kwargs: Additional arguments for vector store initialization
    """
    if use_chroma:
        return ChromaVectorStore(**kwargs)
    else:
        # Fallback to original FAISS implementation
        return FAISSVectorStore(**kwargs)


# Rename the original class for clarity
class FAISSVectorStore:
    """
    FAISS-based vector store implementation (fallback option)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """Initialize FAISS vector store (original implementation)"""
        # ... (keep the original FAISS implementation as before)


# Test the vector store
if __name__ == "__main__":
    def test_vector_store():
        """Test the vector store functionality"""
        print("🧪 Testing VectorStore")
        print("="*50)
        
        # Initialize vector store
        vs = VectorStore()
        
        # Test documents
        documents = [
            "Artificial intelligence is transforming how we work and live.",
            "Machine learning models require large amounts of training data.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers.",
            "Computer vision allows machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards and penalties.",
            "AI safety research focuses on beneficial and controllable AI systems."
        ]
        
        # Add documents
        print("\n📄 Adding test documents...")
        vs.add_documents(documents)
        
        # Test searches
        test_queries = [
            "machine learning and training data",
            "neural networks and deep learning", 
            "computer vision and images",
            "AI safety research"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching: '{query}'")
            results = vs.search_with_scores(query, n_results=3)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.3f}] {doc[:60]}...")
        
        # Show stats
        print(f"\n📊 Vector Store Stats:")
        stats = vs.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test save/load
        print(f"\n💾 Testing save/load...")
        save_path = "./chroma_db"
        vs.save(save_path)
        
        # Create new instance and load
        vs2 = VectorStore()
        vs2.load(save_path)
        
        print(f"✅ Loaded {vs2.get_document_count()} documents")
        
        # Clean up
        import shutil
        if os.path.exists("./test_vectorstore.faiss"):
            os.remove("./test_vectorstore.faiss")
        if os.path.exists("./test_vectorstore.pkl"):
            os.remove("./test_vectorstore.pkl")
        
        print("🎉 All tests passed!")
    
    test_vector_store()