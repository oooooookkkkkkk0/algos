import heapq
import numpy as np


def top_k_search(embeddings, query, k):
    """
    embeddings: np.ndarray shape (N, d)
    query: np.ndarray shape (d,)
    k: int
    Returns: list [(distance, index)] sorted ascending
    """

    heap = []
    for index, vector in enumerate(embeddings):
        distance = np.linalg.norm(vector - query)
        item = (-distance, index)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif distance < -heap[0][0]:
            heapq.heapreplace(heap, item)

    result = [(-distance, index) for distance, index in heap]
    result.sort()
    return result

assert len(
    top_k_search(
        np.random.randn(1000, 8),
        np.random.randn(8),
        5
    )
) == 5