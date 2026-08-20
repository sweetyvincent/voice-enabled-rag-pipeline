import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Reranker:
    """
    Reranker for retrieved chunks.
    """
    def rerank(self, results: List[Dict[str, Any]], final_k: int = 5) -> List[Dict[str, Any]]:
        """
        Apply diversity enforcement and strategy bonuses, and limit output.
        """
        if not results:
            return []

        # Start with the raw results sorted by score
        sorted_results = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)
        
        passage_counts = {}
        
        for res in sorted_results:
            passage_id = res.get('passage_id')
            if not passage_id and res.get('doc_metadata') and isinstance(res.get('doc_metadata'), dict):
                passage_id = res.get('doc_metadata').get('passage_id')
            
            score = res.get('score', 0.0)
            
            # Apply diversity enforcement
            if passage_id:
                count = passage_counts.get(passage_id, 0)
                if count > 0:
                    score = score * (0.8 ** count)
                passage_counts[passage_id] = count + 1
                
            # Apply strategy bonus
            strategy = res.get('strategy', '')
            if strategy == 'semantic':
                score += 0.02
            elif strategy == 'passage_aware':
                score += 0.01
                
            res['score'] = score
            
        # Re-sort by adjusted score
        adjusted_results = sorted(sorted_results, key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Enforce max 3 chunks from any single strategy in final results
        strategy_counts = {}
        final_results = []
        
        for res in adjusted_results:
            strategy = res.get('strategy', 'unknown')
            if strategy_counts.get(strategy, 0) < 3:
                final_results.append(res)
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if len(final_results) >= final_k:
                break
                
        # Normalize final scores to [0, 1] range
        if final_results:
            scores = [r.get('score', 0.0) for r in final_results]
            min_score = min(scores)
            max_score = max(scores)
            
            for res in final_results:
                if max_score > min_score:
                    res['score'] = (res['score'] - min_score) / (max_score - min_score)
                else:
                    res['score'] = 1.0 if max_score > 0 else 0.0
                    
        return final_results
