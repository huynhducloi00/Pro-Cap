from utils import set_env

set_env()
import torch
from transformers import pipeline
from transformers.generation import EosTokenCriteria

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
prompt = """
Example 1: When you open up theassignments folder on GoogleClassroom and half of themWhen my students ask me ifthings are going to go back tonormal soonare blankI HAVE NO IDEA .  a man in a suit and tie is eating a banana . a black man in a suit and tie is eating a banana on a television set . a man is eating a banana and a woman is eating a banana on a t.v. screen . the person in the image is from the united states and the person in the image is from the united states and the person in the image is from the . he is a christian and he is wearing a tuxedo and he is eating a banana . Meme Humour Image Internet meme Teacher Joke Education 2021 School 2020 .  . Indian Male Middle Eastern Male . . Classification: Good

Example 2: GPANGricatapieWeed Kills Corona VirusScientists are shocked to discover that Weed KillsCorona Virus4coronavirusecoronarivirusLIVEBREAKING NEWSWEED KILLS CORONA VIRUSSCIENTIST ARE SHOCKED TO DISCOVER THAT WEED KILLS CORONA VIRUS458 AM Feb 42020 Titie orAvaoe .  a piece of marijuana in a glass of water . he is a buddhist buddhist buddhist buddhist buddhist . Coronavirus Hemp Coronavirus disease 2019 Cannabis in China Newsweed Cannabis pizza Virus Substance intoxication Corona News .  . unk . Classification: Bad.

Question: ArmaniUSER after hearing trump got covid then realizing he wasthe stage with him 2 days ago1244 AM  Oct 2 2020  Twitter for iPhone . a picture of a man and a woman with their mouths open . a black man with a beard and a white man with a beard and a white man with a beard . he is a man with a beard and a beard and a beard and a beard and a be . america, he is from america, he is from america, he is from america, he is from america . he is a christian and he has a tee shirt with the word christian on it . Steve Harvey Internet meme Meme Image Humour Know Your Meme Imgur GIF good Logo Font Banner Brand Product Graphics Meter .  . Middle Eastern Male Black Male . . Classification:
"""
result = pipe(
    prompt,
    pad_token_id=pipe.model.config.eos_token_id,
    max_length=1000,
    stop_strings=["\n",'.'],
    tokenizer=pipe.tokenizer
)
print(result[len(prompt) :])
