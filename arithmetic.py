import numpy as np

def compute_frequency(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    sum_count = sum(counts)
    counts = [i / sum_count for i in counts]
    frequencies = dict(zip(unique_elements, counts))
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
    return sorted_frequencies


def create_range_dictionary(probabilities):
    range_dict = {}
    start = 0.0

    for key, probability in probabilities.items():
        end = start + probability
        range_dict[key] = {'start': start, 'end': end}
        start = end

    return range_dict

class Arithmetic:
    
    def __init__(self, arr: np.ndarray):  
        self.rngs = create_range_dictionary(compute_frequency(arr))

    
    def encode(self, arr):
        low, high, rng = 0.0, 1.0, 1.0
        for i in arr:
            high = low + rng * self.rngs[i]["end"]
            low = low + rng * self.rngs[i]["start"]
            rng = high - low
        return low, high, rng
    
    def convert_to_decimal(self, bits):
        if bits == "":
            return -1
        deci = 0.0
        for i, k in enumerate(bits):
            deci += int(k) * 2 ** (-(i + 1))
        return deci
    
    def assing2str(self, code, i, c):
        if i != None:
            c = str(c)
            new_code = list(code)
            new_code[i] = c
            return "".join(new_code)
        else:
            c = str(c)
            new_code = list(code)
            new_code.append(c)
            return "".join(new_code)
    
    def generate_codeword(self, low, high):
        code = ""
        while(self.convert_to_decimal(code) < low):
            code = self.assing2str(code, None, 1)
            if self.convert_to_decimal(code) > high:
                code = self.assing2str(code, -1, 0)
        return code
    
    def decode_codeword(self, bits):
        return self.convert_to_decimal(bits)
    
    def encode_arr(self, arr):
        low, high, _ = self.encode(arr)
        return self.generate_codeword(low, high)
    
    def decode_arr(self, codeword, length):
        arr = []
        value = self.convert_to_decimal(codeword)
        for i in range(length):
            k = self.find_symbol(value)
            low = self.rngs[k]["start"]
            high = self.rngs[k]["end"]
            rng = high - low
            value = (value - low) / rng
            arr.append(k)
        return arr


    
    def find_symbol(self, x):
        for k in self.rngs:
            if self.rngs[k]["start"] <= x < self.rngs[k]["end"]:
                return k