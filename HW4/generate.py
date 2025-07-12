import torch
torch.use_deterministic_algorithms(True)
import sys
from models.TransformerLM import *
from data.TinyStories import *
#from spacy.tokenizer import Tokenizer
#torch.serialization.add_safe_globals([Vocabulary, Tokenizer])
from torch.distributions import Categorical
torch.backends.cudnn.deterministic = True

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)
MAX_LENGTH = 500

def main():
   CHKPT_PATH = "./chkpts/XDYf78_TinyStories"
   chkpt = torch.load(CHKPT_PATH, weights_only=False, map_location='cpu')
   config = chkpt['config']

   print(CHKPT_PATH +" // "+str(chkpt['epoch']))
   # load vocab
   vocab = chkpt['vocab']

   # load model
   model = TransformerLM(len(vocab), config["d_model"], config["n_heads"], config["n_layers"])
   model.load_state_dict(chkpt['model_state_dict'])
   model.to(device)
   
   while True:
    # ask for prompt
    prompt = input("\n\nPrompt:\n")

    # numeralize prompt
    num_prompt = vocab.text2idx(prompt)
    l = len(num_prompt)

    
    for sampler in [argmaxDecode, sampleDecode, nucleusDecode]:
      torch.manual_seed(0)
      random.seed(0)
      torch.cuda.manual_seed(0)
      torch.cuda.manual_seed_all(0)
      
      src = torch.zeros(1,MAX_LENGTH)
      src[0,0] = 1 # <SOS>
      src[0,1:l+1] = torch.Tensor(num_prompt)
      src = src.to(dtype=int, device=device)
      print("\n\n")
      print(sampler)
      print(prompt, end="",flush=True)
      for t in range(l+1,MAX_LENGTH):
          out = model(src)

          src[0,t] =  sampler(out[:,t-1,:])
          
          w = vocab.idx2text([src[0,t].cpu().item()])[0]

          if w == "<EOS>":
              break
          if not any(x in w for x in [".",",","\"","'","!","?"]):
              w = " "+w
          
          print(w,  end='',flush=True)
      print("\n")
   sys.exit(1)


def argmaxDecode(scores):
   # TODO
   w_max = torch.argmax(scores).item()
   return w_max

def sampleDecode(scores, temp = 0.5):
   # TODO
   tau_scores = scores / temp
   probs = torch.softmax(tau_scores, dim=-1)
   w_sample = torch.distributions.Categorical(probs).sample().item()
   return w_sample

def nucleusDecode(scores, p=0.9, temp = 0.5):
    # TODO
    # scores = (1, V)
    if scores.dim() == 2:
        scores = scores.squeeze(0)

    tau_scores = scores / temp
    probs = torch.softmax(tau_scores, dim=-1)

    sorted_probs, sorted_indice = probs.sort(descending=True) 

    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    condition = (cum_probs > p)
    nonzero_indices = condition.nonzero(as_tuple=True)[0]
    if len(nonzero_indices)==0:
        k_star = len(probs)
    else:
        k_star = nonzero_indices[0].item() + 1

    k_star = min(k_star, sorted_indice.size(0))
    top_k_indices = sorted_indice[:k_star]

    probs_updated = torch.zeros_like(probs)
    probs_updated[top_k_indices] = probs[top_k_indices]
    probs_updated = probs_updated / probs_updated.sum()

    w_sample = torch.distributions.Categorical(probs_updated).sample().item()
    return w_sample
   
if __name__=="__main__":
    main()