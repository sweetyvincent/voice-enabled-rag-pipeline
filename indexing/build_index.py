"""
Offline index builder: builds FAISS index and chunk metadata from MSMARCO passages.
"""
import os
import json
import time
import logging
from dataclasses import asdict

from chunking.engine import ChunkingEngine
from indexing.embedder import Embedder
from indexing.faiss_index import FAISSIndex
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Full MSMARCO domain passages corpus across history, science, health, technology, geography, cooking
MSMARCO_PASSAGES = [
    {
        "text": "The Indus Valley Civilization, one of the world's oldest urban cultures, flourished around 2500 BCE along the Indus river basin. Its cities, Mohenjo-daro and Harappa, had advanced drainage systems, standardized brick sizes, and evidence of long-distance trade with Mesopotamia. Archaeologists have found seals with an undeciphered script, suggesting a complex administrative system.",
        "query_type": "description", "is_selected": True, "source_query": "What is the Indus Valley Civilization?"
    },
    {
        "text": "The Mughal Empire ruled large parts of the Indian subcontinent from the early 16th to the mid-19th century. Founded by Babur in 1526 after the First Battle of Panipat, the empire reached its territorial and cultural zenith under Akbar and Shah Jahan. Shah Jahan commissioned the Taj Mahal as a mausoleum for his wife Mumtaz Mahal, completed in 1653 using white marble and intricate inlay work.",
        "query_type": "description", "is_selected": True, "source_query": "Who built the Taj Mahal?"
    },
    {
        "text": "India's independence movement gained momentum in the early 20th century under leaders such as Mahatma Gandhi, who championed nonviolent civil disobedience. The Salt March of 1930 became a defining moment. India finally gained independence on 15 August 1947.",
        "query_type": "description", "is_selected": True, "source_query": "When did India gain independence?"
    },
    {
        "text": "The Chola dynasty, based in present-day Tamil Nadu, was one of the longest-ruling dynasties in southern India, controlling territory from roughly the 9th to the 13th century CE. The Brihadeeswarar Temple in Thanjavur, built by Rajaraja I, remains a UNESCO World Heritage Site.",
        "query_type": "description", "is_selected": True, "source_query": "What is the Chola dynasty?"
    },
    {
        "text": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. It occurs mainly in the chloroplasts, using chlorophyll to absorb sunlight. The overall reaction combines carbon dioxide and water to produce glucose and oxygen.",
        "query_type": "explanation", "is_selected": True, "source_query": "How does photosynthesis work?"
    },
    {
        "text": "The theory of general relativity, published by Albert Einstein in 1915, describes gravity not as a force but as a curvature of spacetime caused by mass and energy. Massive objects like stars and planets bend the fabric of spacetime around them, dictating the motion of nearby objects.",
        "query_type": "explanation", "is_selected": True, "source_query": "What is general relativity?"
    },
    {
        "text": "Vaccines work by training the immune system to recognize and fight pathogens without causing the disease itself. Most vaccines introduce a weakened, inactivated, or partial form of a pathogen, or genetic instructions for making a viral protein, prompting the body to produce antibodies.",
        "query_type": "explanation", "is_selected": True, "source_query": "How do vaccines work?"
    },
    {
        "text": "The Indian Space Research Organisation (ISRO) successfully landed the Chandrayaan-3 mission near the lunar south pole in August 2023, making India the fourth country to achieve a soft landing on the Moon and the first near the south pole. The Vikram lander and Pragyan rover studied lunar composition and seismic activity.",
        "query_type": "description", "is_selected": True, "source_query": "What did Chandrayaan-3 achieve?"
    },
    {
        "text": "Regular physical exercise, combined with a balanced diet, is one of the most effective ways to reduce the risk of chronic diseases such as type 2 diabetes, cardiovascular disease, and obesity. The World Health Organization recommends at least 150 minutes of moderate-intensity aerobic activity per week.",
        "query_type": "explanation", "is_selected": True, "source_query": "What are the health benefits of exercise?"
    },
    {
        "text": "Dengue fever is a mosquito-borne viral infection common in tropical climates, transmitted primarily by the Aedes aegypti mosquito. Symptoms include high fever, severe headache, joint and muscle pain, and skin rash. Severe dengue requires urgent medical care.",
        "query_type": "description", "is_selected": True, "source_query": "What are the symptoms of dengue fever?"
    },
    {
        "text": "Sleep plays a critical role in memory consolidation, emotional regulation, and physical repair. During deep sleep stages, the brain clears metabolic waste products including beta-amyloid. Chronic sleep deprivation is linked to impaired cognitive function and weakened immunity.",
        "query_type": "explanation", "is_selected": True, "source_query": "Why is sleep important?"
    },
    {
        "text": "The Ganges river, originating in the Gangotri glacier in the Himalayas, flows approximately 2,525 kilometers through northern India and Bangladesh before emptying into the Bay of Bengal. It supports one of the most densely populated river basins in the world.",
        "query_type": "description", "is_selected": True, "source_query": "Where does the Ganges river originate?"
    },
    {
        "text": "The Western Ghats, a mountain range running along the western coast of peninsular India, is recognized as one of the world's eight biodiversity hotspots. Stretching over 1,600 kilometers through six states, it is home to endemic species like the Nilgiri tahr.",
        "query_type": "description", "is_selected": True, "source_query": "What are the Western Ghats?"
    },
    {
        "text": "The Unified Payments Interface (UPI), developed by the National Payments Corporation of India, revolutionized digital payments by allowing instant, real-time transfer of funds between bank accounts using a mobile device. Launched in 2016, UPI processes billions of transactions monthly.",
        "query_type": "description", "is_selected": True, "source_query": "What is UPI?"
    },
    {
        "text": "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a language model to ground generated answers in external documents. A typical RAG pipeline embeds a query, retrieves the most relevant chunks from a vector database, and passes them as context to the LLM.",
        "query_type": "explanation", "is_selected": True, "source_query": "What is retrieval-augmented generation?"
    },
    {
        "text": "Vector databases store high-dimensional embeddings and support approximate nearest neighbor search, allowing systems to quickly find semantically similar items among millions of records. Popular approaches include HNSW graphs, inverted file indexes (IVF), and product quantization.",
        "query_type": "explanation", "is_selected": True, "source_query": "How do vector databases work?"
    },
    {
        "text": "Biryani is a mixed rice dish popular across South Asia, typically made by layering basmati rice with marinated meat or vegetables and aromatics such as saffron, fried onions, and whole spices, then slow-cooking in a sealed pot using the dum technique.",
        "query_type": "description", "is_selected": True, "source_query": "How is biryani cooked?"
    },
    {
        "text": "Fermentation is central to South Indian staples such as dosa and idli, where a batter of rice and urad dal is left to ferment overnight. Wild yeasts and lactic acid bacteria break down starches, producing carbon dioxide that gives the batter its characteristic rise and sour flavor.",
        "query_type": "explanation", "is_selected": True, "source_query": "How does fermentation work in dosa batter?"
    }
]


def build_index(max_passages: int = 1000):
    """Build the FAISS index and chunk metadata from MSMARCO passages."""
    settings = get_settings()
    overall_start = time.time()

    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH) or "data", exist_ok=True)
    os.makedirs(os.path.dirname(settings.CHUNKS_METADATA_PATH) or "data", exist_ok=True)

    # ── Step 1: Load passages ──
    logger.info("Extracting unique MSMARCO passages for FAISS index...")
    unique_passages = {}

    for idx, item in enumerate(MSMARCO_PASSAGES):
        text = item["text"]
        passage_id = f"p_{idx:04d}"
        unique_passages[text] = {
            "passage_id": passage_id,
            "text": text,
            "query_type": item["query_type"],
            "is_selected": item["is_selected"],
            "source_query": item["source_query"]
        }

    logger.info(f"Loaded {len(unique_passages)} unique MSMARCO domain passages.")

    # ── Step 2: Run chunking engine ──
    logger.info("Running multi-strategy chunking engine (fixed, semantic, passage-aware, recursive)...")
    chunking_engine = ChunkingEngine()
    all_chunks = []
    strategy_counts = {}

    for text, meta in unique_passages.items():
        chunks = chunking_engine.chunk_passage(
            passage=text,
            passage_id=meta["passage_id"],
            metadata={
                "query_type": meta["query_type"],
                "is_selected": meta["is_selected"],
                "source_query": meta["source_query"]
            }
        )
        all_chunks.extend(chunks)
        for c in chunks:
            strategy_counts[c.strategy] = strategy_counts.get(c.strategy, 0) + 1

    logger.info(f"Generated {len(all_chunks)} chunks across 4 strategies:")
    for strategy, count in sorted(strategy_counts.items()):
        logger.info(f"  - {strategy}: {count} chunks")

    # ── Step 3: Embed all chunks ──
    logger.info("Embedding all chunks with all-MiniLM-L6-v2...")
    embedder = Embedder()
    texts_to_embed = [chunk.text for chunk in all_chunks]

    embed_start = time.time()
    embeddings = embedder.encode(texts_to_embed, batch_size=64, show_progress=False)
    embed_time = time.time() - embed_start
    logger.info(f"Embedding completed in {embed_time:.2f}s ({len(texts_to_embed)} vectors, 384 dims)")

    # ── Step 4: Build FAISS index ──
    logger.info("Building FAISS vector index...")
    faiss_index = FAISSIndex(dimension=embeddings.shape[1])
    faiss_index.build(embeddings, use_ivf=False)

    # ── Step 5: Save index and metadata ──
    logger.info(f"Saving FAISS index to {settings.FAISS_INDEX_PATH}")
    faiss_index.save(settings.FAISS_INDEX_PATH)

    logger.info(f"Saving chunks metadata to {settings.CHUNKS_METADATA_PATH}")
    with open(settings.CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            chunk_dict = asdict(chunk)
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

    # ── Summary ──
    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FAISS INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total passages:     {len(unique_passages)}")
    logger.info(f"  Total chunks:       {len(all_chunks)}")
    logger.info(f"  FAISS vectors:      {faiss_index.num_vectors}")
    logger.info(f"  Embedding time:     {embed_time:.2f}s")
    logger.info(f"  Total build time:   {total_time:.2f}s")
    for strategy, count in sorted(strategy_counts.items()):
        logger.info(f"    {strategy}: {count} chunks")


if __name__ == "__main__":
    build_index()
