#!/usr/bin/env python3

with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()

# 修复 ContrastiveLearner - 添加动态输入维度处理
old_contrastive = '''class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_dim: int = 64, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        anchors = self.projection_head(anchors)
        positives = self.projection_head(positives)
        negatives = self.projection_head(negatives)'''

new_contrastive = '''class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_dim: int = 64, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.target_dim = encoder_dim
        self.projection_head = None

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        actual_dim = anchors.size(-1)
        
        if self.projection_head is None:
            self.projection_head = nn.Sequential(
                nn.Linear(actual_dim, self.target_dim),
                nn.ReLU(),
                nn.Linear(self.target_dim, self.target_dim)
            ).to(anchors.device).to(anchors.dtype)
        
        anchors = self.projection_head(anchors)
        positives = self.projection_head(positives)
        negatives = self.projection_head(negatives)'''

content = content.replace(old_contrastive, new_contrastive)

# 修复 _extract_layer_features - 确保输出正确
old_extract = '''    def _extract_layer_features(self, layer_grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        layer_features = []
        
        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]
            
            pos_ratio, neg_ratio, _ = self._analyze_gradient_direction(grad)
            stability, grad_diff, momentum = self._analyze_gradient_stability(layer_name, grad)
            
            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]

            base_features = self.gradient_encoder(grad.unsqueeze(0).to(self.dtype))
            
            layer_features.append(base_features)
        
        layer_features = torch.cat(layer_features, dim=0)
        
        return layer_features'''

new_extract = '''    def _extract_layer_features(self, layer_grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        layer_features = []
        
        for layer_name in sorted(layer_grads.keys()):
            grad = layer_grads[layer_name]
            
            pos_ratio, neg_ratio, _ = self._analyze_gradient_direction(grad)
            stability, grad_diff, momentum = self._analyze_gradient_stability(layer_name, grad)
            
            if grad.size(0) < self.encoder_hidden_dim:
                grad = F.pad(grad, (0, self.encoder_hidden_dim - grad.size(0)))
            elif grad.size(0) > self.encoder_hidden_dim:
                grad = grad[:self.encoder_hidden_dim]

            base_features = self.gradient_encoder(grad.unsqueeze(0).to(self.dtype))
            
            layer_features.append(base_features)
        
        layer_features = torch.cat(layer_features, dim=0)
        
        return layer_features'''

content = content.replace(old_extract, new_extract)

# 写入文件
with open('code_generation_DGMM.py', 'w') as f:
    f.write(content)

print("✅ 修复完成！ContrastiveLearner现在会自动适应输入维度")
