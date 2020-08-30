import tkinter as tk
# from tkinter import *

from generator.shakespeare.shakespeare_model import ShakespeareRNN
from generator.shakespeare import shakespeare_generator

from generator.cooking_recipes.recipes_model import RecipesRNN
from generator.cooking_recipes import recipes_generator

from generator.phone_book.phonebook_model import PhonebookRNN
from generator.phone_book import phonebook_generator

from generator.philosophy.philosophy_model import PhilosophyRNN
from generator.philosophy import philosophy_generator

from generator.openAI.GPT2.model import (GPT2LMHeadModel)
from generator.openAI.GPT2.utils import load_weight
from generator.openAI.GPT2.config import GPT2Config
from generator.openAI.GPT2.sample import sample_sequence
from generator.openAI.GPT2.encoder import get_encoder
from generator.openAI import openAI_generator
from generator.transformers_partei import partei_generation

from gui import gui_functions as gf
# from gui import gui_functions_test as gf

if __name__ == '__main__':
    tk.mainloop()