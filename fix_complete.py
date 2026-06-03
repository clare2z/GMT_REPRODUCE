#!/usr/bin/env python3

with open('code_generation_DGMM.py', 'r') as f:
    content = f.read()

# 修复 ContrastiveLearner - 每次forward都检查维度
old_forward = '''    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
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

new_forward = '''    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        actual_dim = anchors.size(-1)

        if self.projection_head is None or self.projection_head[0].in_features != actual_dim:
            self.projection_head = nn.Sequential(
                nn.Linear(actual_dim, self.target_dim),
                nn.ReLU(),
                nn.Linear(self.target_dim, self.target_dim)
            ).to(anchors.device).to(anchors.dtype)

        anchors = self.projection_head(anchors)
        positives = self.projection_head(positives)
        negatives = self.projection_head(negatives)'''

content = content.replace(old_forward, new_forward)

# 写入文件
with open('code_generation_DGMM.py', 'w') as f:
    f.write(content)

print("✅ 修复完成！现在会检查并重新创建投影头")
