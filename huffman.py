import heapq

class HeapNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, node):
        return self.freq < node.freq
    
    def __eq__(self, node):
        if node == None:
            return False
        if not isinstance(node, HeapNode):
            return False
        return self.freq == node.freq

class HuffmanCoding:
    def __init__(self, arr):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.make_frequency_dict(arr)
        self.make_heap()
        self.merge_nodes()
        self.make_codes()
    
    def make_frequency_dict(self, arr):
        self.freqs = {}
        for i in arr:
            if not i in self.freqs:
                self.freqs[i] = 0
            self.freqs[i] += 1
    
    def make_heap(self):
        for k in self.freqs:
            node = HeapNode(k, self.freqs[k])
            heapq.heappush(self.heap, node)
    
    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)
    
    def make_codes_helper(self, root: HeapNode, current_code):
        if root == None:
            return
        if root.symbol != None:
            self.codes[root.symbol] = current_code
            self.reverse_mapping[current_code] = root.symbol
            return
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")
    
    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)
    
    def encode_arr(self, arr):
        encoded_text = ""
        for i in arr:
            encoded_text += self.codes[i]
        return encoded_text
    
    def decode_arr(self, encoded_text, length):
        tmp = ""
        arr = []
        for b in encoded_text:
            tmp += b
            if tmp in self.reverse_mapping:
                i = self.reverse_mapping[tmp]
                arr.append(i)
                tmp = ""
        return arr