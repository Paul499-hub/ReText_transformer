import random
# For simple two number addition (100K limit)
class MathEncoder:
    def __init__(self):
        chars=list("0123456789+=")
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.n_vocab = len(self.stoi)

    def encode(self, x:str ) -> list[int]:
        return [self.stoi[ch] for ch in x]

    def decode(self, x:list[int]):
        return ''.join([self.itos[num] for num in x])
    
    def get_math_eq(self): # 042+100=142 (always 3 digits)
        r1 = random.randint(0,999)
        r2 = random.randint(0,999)
        sample = f"{r1:03d}+{r2:03d}={(r1+r2):04d}"
        return sample
    
    def generate_training_samples_from_eq(self, math_sample:str)->list[dict]:
        """
        Returns 1 left-shifted Q->A pair.
        Example output:
        [
            {
                'q': {'start': 0, 'end': 10}, 
                'a': {'start': 1, 'end': 11}
            }
        ]
        """
        pointer = math_sample.index("=") + 3
        q = {'start':0,'end':pointer}
        pointer += 1
        a = {'start':1,'end':pointer}
        return q, a