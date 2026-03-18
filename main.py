import torch, torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
 
ANSWER_VOCAB = ['yes','no','red','blue','green','1','2','3','cat','dog','car','house','left','right']
 
class VQAModel(nn.Module):
    def __init__(self, n_answers=len(ANSWER_VOCAB)):
        super().__init__()
        # Visual encoder
        resnet = models.resnet50(pretrained=False)
        self.visual_enc = nn.Sequential(*list(resnet.children())[:-2],
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(2048, 768))
        # Language encoder
        self.text_enc = BertModel.from_pretrained('bert-base-uncased')
        # Fusion
        self.attention = nn.MultiheadAttention(768, 8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, n_answers))
 
    def forward(self, img, input_ids, attention_mask):
        vis = self.visual_enc(img).unsqueeze(1)  # (B,1,768)
        text_out = self.text_enc(input_ids, attention_mask=attention_mask)
        txt = text_out.last_hidden_state  # (B,L,768)
        attended, _ = self.attention(vis, txt, txt)
        fused = torch.cat([attended.squeeze(1), text_out.pooler_output], dim=-1)
        return self.classifier(fused)
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = VQAModel()
questions = ["What color is the car?", "Is there a cat in the image?", "How many dogs are there?"]
for q in questions:
    enc = tokenizer(q, return_tensors='pt', padding='max_length', max_length=32, truncation=True)
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(img, enc['input_ids'], enc['attention_mask'])
    answer = ANSWER_VOCAB[out.argmax(-1).item()]
    print(f"Q: {q:40s} → A: {answer}")
