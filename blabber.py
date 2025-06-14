import torch
from v2 import *
import argparse

MAX_NEW_TOKENS = 128

# parser = argparse.ArgumentParser()
model = GPTLanguageModel()
state_dict = torch.load("shakespeare_blabber_state_dict.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
# model = torch.load("shakespeare_blabber.pth", weights_only=False, map_location=device)

# parser.add_argument(
#     "max_new_tokens",
#     type=int,
#     help="Max new tokens for text generation (> 50)"
# )
# args = parser.parse_args()


init_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
# max_new_tokens = args.max_new_tokens

# if max_new_tokens < 50:
#     context = model.generate(init_idx, max_new_tokens)
#     print(decode(context[0].tolist()))
# else:
#     for i in range(max_new_tokens//50):


while True:
    context = model.generate(init_idx, MAX_NEW_TOKENS)
    init_idx = context[:, -1:]
    print(decode(context[0][1:].tolist()), end="")
