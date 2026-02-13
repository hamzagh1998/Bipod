import faiss
import numpy as np
import httpx
import os
import json
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("bipod.services.vector")

class VectorService:
    """Service for long-term semantic memory using FAISS (Local & Weightless)."""
    
    def __init__(self):
        self.index_path = os.path.join(settings.VECTOR_DIR, "bipod.index")
        self.metadata_path = os.path.join(settings.VECTOR_DIR, "metadata.json")
        self.dim = 768 # Expected dimension for nomic-embed-text/small models
        self.max_chunk_chars = 6000 # ~1500 tokens, safe limit for context windows
        self.chunk_overlap = 500
        
        # Load or create index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()

    def _create_empty_index(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []
        logger.info("Created new empty FAISS index")

    def _chunk_text(self, text: str) -> List[str]:
        """Splits long text into overlapping chunks for better semantic coverage."""
        if len(text) <= self.max_chunk_chars:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start += (self.max_chunk_chars - self.chunk_overlap)
        return chunks

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            # Prevent context overflow: truncate query/text if it's still way too large
            safe_text = text[:12000] # Hard cap as a last resort
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": settings.EMBEDDING_MODEL, "prompt": safe_text}
                )
                if response.status_code == 200:
                    return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
        return None

    async def add_memory(self, text: str, user_id: int, message_id: int, conversation_id: str):
        """Adds a text snippet to the vector database, chunking if necessary."""
        if not text or len(text.strip()) < 10:
            return

        chunks = self._chunk_text(text)
        logger.info(f"Indexing memory (splits: {len(chunks)})")

        for i, chunk in enumerate(chunks):
            embedding = await self._get_embedding(chunk)
            if embedding:
                # Update dimension if first run mismatch
                if len(embedding) != self.dim:
                    self.dim = len(embedding)
                    self._create_empty_index()
                
                vec = np.array([embedding]).astype('float32')
                self.index.add(vec)
                
                self.metadata.append({
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "text": chunk,
                    "message_id": message_id,
                    "chunk_index": i,
                    "embedding": embedding # Cache for fast rebuilds
                })
        
        # Save state after all chunks are added
        if chunks:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.info(f"Memory indexed successfully for user {user_id}")

    async def search_memories(
        self, 
        query: str, 
        user_id: int, 
        n_results: int = 3,
        conversation_id: Optional[str] = None,
        exclude_conversation_id: Optional[str] = None
    ) -> List[str]:
        """Searches for semantically similar memories for a given user.
        
        Args:
            query: The text to search for similar memories.
            user_id: Only return memories belonging to this user.
            n_results: Maximum number of results to return.
            conversation_id: If set, only return memories from this conversation.
            exclude_conversation_id: If set, exclude memories from this conversation
                                     (useful to avoid echoing back the current conversation).
        """
        if self.index.ntotal == 0:
            return []

        embedding = await self._get_embedding(query)
        if not embedding:
            return []

        vec = np.array([embedding]).astype('float32')
        distances, indices = self.index.search(vec, min(n_results * 5, self.index.ntotal))
        
        results = []
        for idx, i in enumerate(indices[0]):
            if i == -1: continue
            if i >= len(self.metadata): continue  # Safety check
            meta = self.metadata[i]
            
            # Must belong to this user
            if meta["user_id"] != user_id:
                continue
            
            # Filter by conversation if specified
            if conversation_id and meta.get("conversation_id") != conversation_id:
                continue
            
            # Exclude current conversation if specified
            if exclude_conversation_id and meta.get("conversation_id") == exclude_conversation_id:
                continue
            
            # Check distance threshold — ignore very distant (irrelevant) memories
            distance = distances[0][idx]
            if distance > 100.0:  # Skip memories that are too far away semantically
                continue
                
            results.append(meta["text"])
            if len(results) >= n_results:
                break
                    
        return results

    async def delete_conversation_memories(self, conversation_id: str):
        """Deletes all memories associated with a specific conversation."""
        try:
            old_count = len(self.metadata)
            # Filter metadata
            new_metadata = [m for m in self.metadata if m.get("conversation_id") != conversation_id]
            
            if len(new_metadata) == old_count:
                logger.info(f"No memories found for conversation {conversation_id} in vector DB.")
                return

            logger.info(f"Deleting {old_count - len(new_metadata)} memories for conversation {conversation_id}")
            
            # Rebuild index
            self.metadata = new_metadata
            self._create_empty_index() # Clear current index

            # Re-index remaining memories
            if self.metadata:
                embeddings = [m["embedding"] for m in self.metadata if "embedding" in m]
                
                if embeddings:
                    vecs = np.array(embeddings).astype('float32')
                    self.index.add(vecs)
            
            # Save updated state
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            
            logger.info(f"Vector DB cleanup complete for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}")


vector_service = VectorService()
