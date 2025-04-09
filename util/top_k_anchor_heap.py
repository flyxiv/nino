import heapq

class TopKAnchorHeap:
    """Efficient Data Structure for saving only the top-k elements
    """

    def __init__(self, k):
        self.k = k
        self.min_heap = []
        
    def add(self, score, pred_idx):
        if len(self.min_heap) < self.k:
            heapq.heappush(self.min_heap, (score, pred_idx))
        elif score > self.min_heap[0][0]:
            heapq.heappushpop(self.min_heap, (score, pred_idx))
    
    def get_top_k(self):
        return sorted(self.min_heap, reverse=True)