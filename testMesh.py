import torch  
  
# 假设你有两个张量 tensor1 和 tensor2  
tensor1 = torch.tensor([[1, 2, 3]])  
tensor2 = torch.tensor([[0.5,0.6,0.7]])

tensor1 = torch.mul(tensor1, tensor2)
print(tensor1)