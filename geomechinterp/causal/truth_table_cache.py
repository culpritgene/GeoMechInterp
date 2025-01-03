from typing import Dict, Tuple
import numpy as np
from geomechinterp.causal.hasse import generate_truth_tables


class TruthTableCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TruthTableCache, cls).__new__(cls)
            cls._instance._cache = {}
            # Pre-generate truth tables for 1-3 inputs
            for n in range(4):
                cls._instance._cache[n] = generate_truth_tables(
                    n, exclude_non_causal=False
                )
        return cls._instance

    def get_table(self, key: Tuple) -> np.ndarray:
        """Get truth table from cache or create if not exists"""
        return self._cache.get(key)

    def store_table(self, key: Tuple, table: np.ndarray) -> None:
        """Store truth table in cache"""
        self._cache[key] = table

    def clear(self):
        """Clear the cache"""
        self._cache.clear()


# Global singleton instance
truth_table_cache = TruthTableCache()
