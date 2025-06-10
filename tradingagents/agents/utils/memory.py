import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Tuple, Dict, Any, Optional


class FinancialSituationMemory:
    """
    Memory system for financial situations using HuggingFace sentence transformers
    instead of OpenAI embeddings.
    """

    def __init__(self, name: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the financial situation memory.

        Args:
            name (str): Name of the collection
            embedding_model (str): HuggingFace sentence transformer model name
        """
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

        # Initialize HuggingFace sentence transformer model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            # Set to evaluation mode and move to appropriate device
            self.embedding_model.eval()
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.cuda()
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading embedding model {embedding_model}: {e}")
            # Fallback to a smaller model
            fallback_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
            print(f"Falling back to: {fallback_model}")
            self.embedding_model = SentenceTransformer(fallback_model)
            self.embedding_model.eval()
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.cuda()

    def get_embedding(self, text: str) -> List[float]:
        """
        Get HuggingFace sentence transformer embedding for a text.

        Args:
            text (str): Input text to embed

        Returns:
            List[float]: Embedding vector
        """
        try:
            # Generate embedding using sentence transformer
            with torch.no_grad():
                embedding = self.embedding_model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                # Convert to CPU and then to list
                if torch.is_tensor(embedding):
                    embedding = embedding.cpu().numpy()

                return embedding.tolist()

        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_model.get_sentence_embedding_dimension()

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch for efficiency.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            with torch.no_grad():
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=32  # Process in batches to manage memory
                )

                # Convert to CPU and then to list
                if torch.is_tensor(embeddings):
                    embeddings = embeddings.cpu().numpy()

                return embeddings.tolist()

        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            # Return zero vectors as fallback
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [[0.0] * dim for _ in texts]

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """
        Add financial situations and their corresponding advice using HuggingFace embeddings.

        Args:
            situations_and_advice (List[Tuple[str, str]]): List of tuples (situation, recommendation)
        """
        if not situations_and_advice:
            print("No situations to add")
            return

        situations = []
        advice = []
        ids = []

        offset = self.situation_collection.count()

        # Prepare data
        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))

        # Generate embeddings in batch for efficiency
        print(f"Generating embeddings for {len(situations)} situations...")
        embeddings = self.get_batch_embeddings(situations)

        try:
            # Add to ChromaDB collection
            self.situation_collection.add(
                documents=situations,
                metadatas=[{"recommendation": rec} for rec in advice],
                embeddings=embeddings,
                ids=ids,
            )
            print(f"Successfully added {len(situations)} situations to memory")

        except Exception as e:
            print(f"Error adding situations to collection: {e}")

    def get_memories(self, current_situation: str, n_matches: int = 1) -> List[Dict[str, Any]]:
        """
        Find matching recommendations using HuggingFace embeddings.

        Args:
            current_situation (str): Current financial situation to match
            n_matches (int): Number of matches to return

        Returns:
            List[Dict[str, Any]]: List of matched situations with recommendations and similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.get_embedding(current_situation)

            # Query the collection
            results = self.situation_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_matches, self.situation_collection.count()),
                include=["metadatas", "documents", "distances"],
            )

            # Process results
            matched_results = []

            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    # Convert distance to similarity score (1 - distance for cosine similarity)
                    similarity_score = 1 - results["distances"][0][i]

                    matched_results.append({
                        "matched_situation": results["documents"][0][i],
                        "recommendation": results["metadatas"][0][i]["recommendation"],
                        "similarity_score": max(0.0, similarity_score),  # Ensure non-negative
                    })

            return matched_results

        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def clear_memory(self):
        """Clear all stored memories."""
        try:
            # Delete and recreate the collection
            collection_name = self.situation_collection.name
            self.chroma_client.delete_collection(collection_name)
            self.situation_collection = self.chroma_client.create_collection(name=collection_name)
            print(f"Cleared memory collection: {collection_name}")
        except Exception as e:
            print(f"Error clearing memory: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory collection."""
        try:
            count = self.situation_collection.count()
            return {
                "collection_name": self.situation_collection.name,
                "total_memories": count,
                "embedding_model": self.embedding_model._modules['0'].auto_model.name_or_path if hasattr(
                    self.embedding_model, '_modules') else "unknown",
                "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    def search_similar_situations(self, query: str, threshold: float = 0.7, max_results: int = 10) -> List[
        Dict[str, Any]]:
        """
        Search for situations above a similarity threshold.

        Args:
            query (str): Query situation
            threshold (float): Minimum similarity score (0-1)
            max_results (int): Maximum number of results

        Returns:
            List[Dict[str, Any]]: Filtered results above threshold
        """
        memories = self.get_memories(query, n_matches=max_results)
        return [memory for memory in memories if memory["similarity_score"] >= threshold]


def create_default_financial_memory(name: str = "default_financial_memory") -> FinancialSituationMemory:
    """
    Create a financial memory with some default financial situations.

    Args:
        name (str): Name for the memory collection

    Returns:
        FinancialSituationMemory: Initialized memory with default situations
    """
    memory = FinancialSituationMemory(name)

    # Default financial situations and advice
    default_situations = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration. Monitor commodity exposure as hedge against inflation.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows. Consider quality tech stocks trading at discounts.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt. Focus on dollar-earning multinational companies.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates like financials. Reduce duration risk.",
        ),
        (
            "Economic recession indicators with declining corporate earnings",
            "Increase cash position and focus on quality defensive stocks. Consider government bonds for safety. Avoid highly leveraged companies and cyclical sectors.",
        ),
        (
            "Bull market with low volatility and strong momentum",
            "Consider taking some profits and rebalancing. Look for quality growth stocks. Be cautious of overvaluation. Maintain disciplined position sizing.",
        ),
        (
            "Central bank dovish policy with quantitative easing announcements",
            "Consider increasing risk assets allocation. Look at growth stocks and real estate. Monitor inflation expectations. Consider commodities as inflation hedge.",
        ),
        (
            "Geopolitical tensions affecting energy and commodity markets",
            "Diversify across regions and consider commodity exposure. Monitor supply chain disruptions. Focus on companies with strong balance sheets and pricing power.",
        ),
    ]

    # Add default situations to memory
    memory.add_situations(default_situations)

    return memory


if __name__ == "__main__":
    # Example usage with HuggingFace embeddings
    print("Testing FinancialSituationMemory with HuggingFace embeddings...")

    # Create memory instance
    memory = FinancialSituationMemory("test_memory")

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    memory.add_situations(example_data)

    # Print memory stats
    stats = memory.get_memory_stats()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        print(f"\nQuerying situation: {current_situation.strip()}")
        recommendations = memory.get_memories(current_situation, n_matches=2)

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\nMatch {i}:")
                print(f"Similarity Score: {rec['similarity_score']:.3f}")
                print(f"Matched Situation: {rec['matched_situation']}")
                print(f"Recommendation: {rec['recommendation']}")
        else:
            print("No recommendations found")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")

    # Test search with threshold
    print(f"\n" + "=" * 50)
    print("Testing similarity search with threshold...")
    similar_situations = memory.search_similar_situations(
        "Tech stocks declining due to interest rate concerns",
        threshold=0.3,
        max_results=5
    )

    if similar_situations:
        print(f"Found {len(similar_situations)} situations above threshold 0.3:")
        for i, situation in enumerate(similar_situations, 1):
            print(f"\n{i}. Score: {situation['similarity_score']:.3f}")
            print(f"   Situation: {situation['matched_situation'][:100]}...")
    else:
        print("No situations found above threshold")