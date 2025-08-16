import torch
import torch_directml

print(f'torch.backends.directml.is_available(): {torch.backends.directml.is_available()}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
