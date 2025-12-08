import torch

class RandomBatcher:
    def __init__(self, data_tensor: torch.Tensor, context_len: int, batch_size: int ):
        """
        `data_tensor` is raw text encoded into integers, torch tensor
        Example data_tensor:
            train_bucket.shape: torch.Size([1000]) ---> shape[context_len * bucket_size] (not precisely but close to this value)
        """
        self.ptr = 0 # This pointer marks a START of context_length input passed to model for every batch. So 10 = 10*context_length tokens were passed
        self.bs = batch_size
        self.data_tensor = data_tensor
        self.context_len = context_len
        self._max = data_tensor.size(0) - context_len        # Get max valid index
        self.starting_points = torch.randperm(self._max + 1, device=self.data_tensor.device) # All valid offsets shuffled once per epoch

    def random_batch_torch(self) -> torch.Tensor:
        """
        ## Get training data batch
        
        ### Returns torch.Tensor of shape: 
        - [batch_size, context_length] = [B, T]
        """
        if self.bs <= 0 or self.data_tensor.size(0) < self.context_len:
            print(f'------ X ------- Invalid batch. Conditions not met : self.bs <= 0 or self.data_tensor.size(0) < self.context_len')
            return None  # invalid batch
        # If we reached the end of starting_points -> return None
        if self.ptr + self.bs > len(self.starting_points):
            return None
        # Get a batch of starting_points shape: [self.bs]
        starts = self.starting_points[self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        # Get indexes of context_len from starting_points (create a new dimension for each starting point and add arange to it.)
        idx = starts.unsqueeze(1) + torch.arange(self.context_len, device=self.data_tensor.device)
        # Convert indexes into data
        out = self.data_tensor[idx] 
        assert out.size(0) == self.bs and out.size(1) == self.context_len
        return out
   
def example_usage():
    context_length = 3
    batch_size = 2
    rbat = RandomBatcher( torch.tensor([10,20,30,40,50,60,70,80,90,100]), context_length, batch_size)
    for _ in range(5):
        out = rbat.random_batch_torch()
        print( out )
        if out is not None:
            print( out.shape )
        print('---')
    # --------- CONSOLE OUTPUT ---------------- 
    # tensor([[ 80,  90, 100],
    #         [ 70,  80,  90]])
    # torch.Size([2, 3])
    # ---
    # tensor([[20, 30, 40],
    #         [10, 20, 30]])
    # torch.Size([2, 3])
    # ---
    # tensor([[30, 40, 50],
    #         [60, 70, 80]])
    # torch.Size([2, 3])
    # ---
    # tensor([[40, 50, 60],
    #         [50, 60, 70]])
    # torch.Size([2, 3])
    # ---
    # None
    # ---

# example_usage()