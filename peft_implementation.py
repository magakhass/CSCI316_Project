# Manual PyTorch Implementation of LoRA (Low-Rank Adaptation)
# Based on paper: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
# Paper reference: https://arxiv.org/abs/2106.09685
# Code reference: https://medium.com/@aseer-ansari/parameter-efficient-fine-tuning-lora-in-pytorch-3749f45c64af
# For default hyperparameters we use the same ones as the HuggingFace alternative
 
import torch
import torch.nn as nn
 
 
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
 
        self.original_layer = original_layer
        self.r = r
        # Scaling factor to control the magnitude of LoRA updates
        self.scaling = lora_alpha / r
 
        in_features = original_layer.in_features
        out_features = original_layer.out_features
 
        # Low-rank matrices A and B
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
 
        # Initializing LoRA weights
        # lora_A: Gaussian initialization as per the original LoRA paper (Hu et al. 2021)
        # This gives small random values to start learning from
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        # lora_B: Zero initialization ensures LoRA updates start at zero,
        # so the model begins identical to the pretrained baseline
        nn.init.zeros_(self.lora_B.weight)
 
        # Freezing the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
 
    def forward(self, x):
        # Original frozen output + scaled LoRA update
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return original_output + lora_output
 
 
def apply_lora(model, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"]):
    # Replacing target linear layers with LoRALayer
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, LoRALayer(module, r, lora_alpha, lora_dropout))
 
    return model
 
 
def get_trainable_parameters(model):
    # Printing trainable vs total parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
 