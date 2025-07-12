import torch
import pickle
from models.PoSGRU import PoSGRU

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = PoSGRU(
    vocab_size=vocab.lenWords(),
    embed_dim=100,
    hidden_dim=256,
    num_layers=2,
    output_dim=vocab.lenLabels(),
    residual=True,
    embed_init=None
)

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

sentence = input("Enter a sentence: ").strip().lower()
tokens = sentence.split()

x = vocab.numeralizeSentence(tokens)
x_tensor = torch.LongTensor(x).unsqueeze(0)

with torch.no_grad():
    logits = model(x_tensor)
    pred = torch.argmax(logits, dim=-1).squeeze(0)

tags = vocab.denumeralizeLabels(pred.tolist())

print("\nPredicted POS Tags:")
print("-" * 30)
for token, tag in zip(tokens, tags):
    print(f"{token:15}   {tag}")