import os
import csv

import tkinter as tk
from tkinter import *
from tkinter.ttk import Progressbar, Style

import pygame
from threading import Thread
# import queue

# text generation imports
from generator.shakespeare import shakespeare_generator
from generator.cooking_recipes import recipes_generator
from generator.phone_book import phonebook_generator
from generator.philosophy import philosophy_generator

from generator.openAI.GPT2.model import (GPT2LMHeadModel)
from generator.openAI.GPT2.utils import load_weight
from generator.openAI.GPT2.config import GPT2Config
from generator.openAI.GPT2.sample import sample_sequence
from generator.openAI.GPT2.encoder import get_encoder
from generator.openAI import openAI_generator
from generator.transformers_partei import partei_generation

# music generation imports
from music_generator import generate, config, utils
from music_generator.config import device, model as model_config
from music_generator.model import PerformanceRNN
from music_generator.sequence import EventSeq, Control, ControlSeq

root = tk.Tk()

# disable the resize button
# root.resizable(0,0)
root.attributes('-fullscreen',True)
root.configure(background='#F18C8E')

# make the window full screen
class FullScreen(object):
    def __init__(self, root, **kwargs):
        self.root = root
        pad=3
        self._geom = '200x200+0+0'
        root.geometry("{0}x{1}+0+0".format(
            root.winfo_screenwidth()-pad, root.winfo_screenheight()-pad))
        root.bind('<Escape>',self.toggle_geom)            
    def toggle_geom(self,event):
        geom = self.root.winfo_geometry()
        self.root.geometry(self._geom)
        self._geom = geom

full_screen = FullScreen(root)



# --------------------------  TEXT  --------------------------

title_text = Label(text='A PORTRAIT OF THE AI AS A YOUNG CYBER ORACLE',
                    background='#F18C8E',
                    font=("Helvetica", 17))

title_text.place(relx=.5, 
                    rely=.02, 
                    anchor="center")


# a horizontal line
line = Frame(root,
        height=1,
        width=1200,
        bg="black")

line.place(relx=.5, 
        rely=.04, 
        anchor="c")


knowledge_base_text = Label(text='1. CHOOSE THE KNOWLEDGE BASE OF THE ORACLE',
                    background='#F18C8E',
                    font=("Helvetica", 14))

knowledge_base_text.place(relx=.5, 
                    rely=.06, 
                    anchor="c")

knowledge_base = StringVar()
knowledge_base.set('OpenAI')

def knowledge_base_radio(knowledge_base, dataset):
    knowledge = Radiobutton(root, 
                        text=dataset, 
                        variable=knowledge_base, 
                        value=dataset,
                        background='#F18C8E', 
                        activebackground='#F18C8E',
                        activeforeground='#F18C8E',
                        font=("Helvetica", 12))
    return knowledge


openAI = knowledge_base_radio(knowledge_base, 'OpenAI')
openAI.place(relx=.28, 
                rely=.09, 
                anchor="c")

artCrap = knowledge_base_radio(knowledge_base, 'Art Bullsit')
artCrap.place(relx=.33, 
                rely=.09, 
                anchor="c")
				
openAIpartei = knowledge_base_radio(knowledge_base, 'Partei Text')
openAIpartei.place(relx=.38,
                rely=.09, 
                anchor="c")


# shakespeare = knowledge_base_radio(knowledge_base, 'Shakespeare')
# shakespeare.place(relx=.48,
#                 rely=.09,
#                 anchor="c")
#
#
# cooking_recipes = knowledge_base_radio(knowledge_base, 'Cooking Recipes')
# cooking_recipes.place(relx=.58,
#                 rely=.09,
#                 anchor="c")
#
#
# telephone_book = knowledge_base_radio(knowledge_base, 'Telephone Book')
# telephone_book.place(relx=.68,
#                 rely=.09,
#                 anchor="c")


class ValidateContext:
    def __init__(self, root):
        self.context_text_text = Label(text='2. GIVE THE ORACLE SOME CONTEXT, ASK IT A QUESTION OR SHARE WHAT IS ON YOUR MIND', 
                                    background='#F18C8E',
                                    font=("Helvetica", 14))
        
        self.context_text_text.place(relx=.5,
                                 rely=.14, 
                                 anchor="c")

        self.vars = tk.StringVar()
        self.vars.trace('w', self.validate)
        self.context_text = Entry(root, 
                                textvariable=self.vars, 
                                width=100,
                                background='#BADDFE')

        self.context_text.place(relx=.5, 
                            rely=.18, 
                            anchor="c")
 
    def validate(self, *args):
        if not self.vars.get().isalpha():
            corrected = ''.join(self.vars.get()[:90])
            self.vars.set(corrected)

context_obj = ValidateContext(root)


# output temperature
def temperature_slider():
    temperature = Scale(root, 
                        from_=0.5, 
                        to=3, 
                        resolution=0.5, 
                        orient=HORIZONTAL,
                        background='#1E93F0',
                        foreground='#FFFFFF',
                        troughcolor='#70BAFF',
                        activebackground='#1E93F0',
                        length=150)

    temperature.set(1)

    temperature.place(relx=.35, 
                    rely=.27, 
                    anchor="c")

    return temperature

temperature = temperature_slider()
temperature_text = Label(text='3. CHOOSE THE LEVEL OF CHANCE AND CONTINGENCY',
                        background='#F18C8E',
                        font=("Helvetica", 14))

temperature_text.place(relx=.35, 
                    rely=.23, 
                    anchor="c")

# text amount
def text_amount_slider():
    text_amount = Scale(root, 
                    from_=250, 
                    to=1000, 
                    resolution = 250, 
                    orient=HORIZONTAL,
                    background='#1E93F0',
                    foreground='#FFFFFF',
                    troughcolor='#70BAFF',
                    activebackground='#1E93F0',
                    length=150)

    text_amount.set(500)

    text_amount.place(relx=.65, 
                    rely=.27, 
                    anchor="c")

    return text_amount

text_amount = text_amount_slider()

text_amount_text = Label(text='4. CHOOSE THE LENGTH OF THE ANSWER',
                    background='#F18C8E',
                    font=("Helvetica", 14))

text_amount_text.place(relx=.65, 
                    rely=.23, 
                    anchor="c")


# store user choices for text generation
def save_text_stats(choice, text_length, temperature, context, generated_text):
    with open('stats/text_stats.csv', mode='a') as stats:
        stats = csv.writer(stats, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
        stats.writerow([choice, text_length, temperature, context, generated_text])


# generate button
def generate_text():
    choice = knowledge_base.get()
    if context_obj.context_text.get() == '':
        if ((choice == 'OpenAI') or (choice == 'Partei Text')):
            context = 'It was a bright cold day in April, and the clocks were striking thirteen.'

        else:
            context = ' '
    else:
        context = context_obj.context_text.get()

    if choice == 'OpenAI':
        progress.start(50)
        progress_bar_text = Label(text='Please wait while the text is being generated',
                            background='#F18C8E',
                            font=("Helvetica", 12))

        progress_bar_text.place(relx=.5, 
                            rely=.37, 
                            anchor="c")
        
        # def display_AI_text(q):
        output_text = openAI_generator.sample(length=int(text_amount.get()/3), temperature=temperature.get(), context_tokens=context)
        output_text = output_text[:text_amount.get()]
        generated_text.configure(state='normal')
        generated_text.delete(1.0,END)
        generated_text.insert(tk.END, output_text)
        generated_text.configure(state='disabled')

        progress.stop()
        progress_bar_text.place_forget()

    elif choice == 'Partei Text':
        progress.start(50)
        progress_bar_text = Label(text='Please wait while the text is being generated',
                                  background='#F18C8E',
                                  font=("Helvetica", 12))

        progress_bar_text.place(relx=.5,
                                rely=.37,
                                anchor="c")

        # def display_AI_text(q):
        output_text = partei_generation.sample(length=int(text_amount.get() / 3),
                                              context_tokens=context)
        output_text = output_text[:text_amount.get()]
        generated_text.configure(state='normal')
        generated_text.delete(1.0, END)
        generated_text.insert(tk.END, output_text)
        generated_text.configure(state='disabled')

        progress.stop()
        progress_bar_text.place_forget()

    elif choice == 'Shakespeare':
        output_text = shakespeare_generator.sample(shakespeare_generator.model, text_amount.get(), prime=context, temperature=temperature.get(), top_k=5)
        generated_text.configure(state='normal')
        generated_text.delete(1.0,END)
        # generated_text.tag_configure('left', justify='left') 
        generated_text.insert(tk.END, output_text)
        # generated_text.tag_add('left', 1.0, 'end') 
        generated_text.configure(state='disabled')

    elif choice == 'Nietzsche + Friends':
        output_text = philosophy_generator.sample(philosophy_generator.model, text_amount.get(), prime=context, temperature=temperature.get(), top_k=5)
        generated_text.configure(state='normal')
        generated_text.delete(1.0,END)
        generated_text.insert(tk.END, output_text, 'center')
        generated_text.configure(state='disabled')

    elif choice == 'Cooking Recipes':
        output_text = recipes_generator.sample(recipes_generator.model, text_amount.get(), prime=context, temperature=temperature.get(), top_k=5)
        generated_text.configure(state='normal')
        generated_text.delete(1.0,END)
        generated_text.insert(tk.END, output_text)
        generated_text.configure(state='disabled')

    elif choice == 'Telephone Book':
        output_text = phonebook_generator.sample(phonebook_generator.model, text_amount.get(), prime=context, temperature=temperature.get(), top_k=5)
        generated_text.configure(state='normal')
        generated_text.delete(1.0,END)
        generated_text.insert(tk.END, output_text)
        generated_text.configure(state='disabled')

    # save stats in a file
    #save_text_stats(choice, text_amount.get(), temperature.get(), context, output_text)

    # clear the context text from the widget
    context_obj.context_text.delete(0, 'end')



# allocate separate thread for the inference so that the main thread won't wait
def start_thread():
    global t

    generate_button['state'] = 'disable'
    generate_music_button['state'] = 'disable'
    replay_butoon['state'] = 'disable'

    t = Thread(target=generate_text)
    t.start()

    check_thread()


def check_thread():
    if not t.is_alive():
        generate_button['state'] = 'normal'
        generate_music_button['state'] = 'normal'
        replay_butoon['state'] = 'normal'
    else:
        root.after(100, check_thread)



# text generation button
generate_button = Button(root, 
                    text="GENERATE", 
                    # command=generate_text, 
                    command=start_thread,
                    background='#1E93F0', 
                    foreground="#FFFFFF",
                    activebackground='#1E93F0',
                    activeforeground='#FFFFFF',
                    width=22,
                    height=2)

generate_button.place(relx=.5, 
                    rely=.33, 
                    anchor="c")


# text area
generated_text = tk.Text(root, 
                    height=9, 
                    width=70, 
                    background='#CFE7FF',
                    foreground='#000000' ,
                    font=('Helvetica', 14),
                    state='disabled',
                    wrap='word')

generated_text.place(relx=.5,
                rely=.5,
                anchor="c")

# scrollbal
scroll = Scrollbar(command=generated_text.yview)

generated_text['yscrollcommand'] = scroll.set

scroll.place(relx=.705,
            rely=.5,
            anchor="c",
            height=201)

# --------------------------  MUSIC  --------------------------

# store user choices for music generation
def save_music_stats(scale, music_length, temperature, note_density, reaction):
    with open('stats/music_stats.csv', mode='a') as stats:
        stats = csv.writer(stats, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
        stats.writerow([scale, music_length, temperature, note_density, reaction.get()])

# play music config
def music_config():
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024

    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1)
    pygame.mixer.music.load('music_generator/output/output-000.mid')

def generate_music():
    temperature = music_temperature.get()
    amount = music_amount.get()
    scale = musical_scale.get()
    density = note_density.get()
    generate.save_midi(temperature, amount, scale, density)

    # play the music
    music_config()
    pygame.mixer.music.stop()
    pygame.mixer.music.play(-1)

    # save the stats about music generation
    save_music_stats(scale, amount, temperature, density, reaction)
    
    
def stop_music():
    music_config()
    pygame.mixer.music.stop()

# music scales
text_amount_text = Label(text='6. CHOOSE A MUSICAL SCALE',
                    background='#F18C8E',
                    font=("Helvetica", 14))

text_amount_text.place(relx=.5, 
                    rely=.71, 
                    anchor="c")

musical_scale = StringVar()
musical_scale.set('C Minor')

def musical_scale_radio(musical_scale, dataset):
    scale = Radiobutton(root, 
                        text=dataset, 
                        variable=musical_scale, 
                        value=dataset,
                        background='#F18C8E', 
                        activebackground='#F18C8E',
                        activeforeground='#F18C8E',
                        font=("Helvetica", 12))
    return scale


c_minor = musical_scale_radio(musical_scale, 'C Minor')
c_minor.place(relx=.37, 
                rely=.75, 
                anchor="c")

c_minor_p = musical_scale_radio(musical_scale, 'C Minor Pentatonic')
c_minor_p.place(relx=.45, 
                rely=.75, 
                anchor="c")

c_major = musical_scale_radio(musical_scale, 'C Major')
c_major.place(relx=.53, 
                rely=.75, 
                anchor="c")

c_major_p = musical_scale_radio(musical_scale, 'C Major Pentatonic')
c_major_p.place(relx=.61, 
                rely=.75, 
                anchor="c")


# note density
def note_density_slider(start, end):
    note_density = Scale(root, 
                        from_=start, 
                        to=end, 
                        resolution=1, 
                        orient=HORIZONTAL,
                        background='#1E93F0',
                        foreground='#FFFFFF',
                        troughcolor='#70BAFF',
                        activebackground='#1E93F0',
                        length=150)

    note_density.set(1)

    note_density.place(relx=.26, 
                    rely=.86, 
                    anchor="c")

    return note_density

note_density = note_density_slider(5, 7)
note_density_text = Label(text='7. CHOOSE A NOTE DENSITY',
                        background='#F18C8E',
                        font=("Helvetica", 14))

note_density_text.place(relx=.26, 
                    rely=.81, 
                    anchor="c")


# music temperature
def music_temperature_slider():
    music_temperature = Scale(root, 
                        from_=0.5, 
                        to=3, 
                        resolution=0.5, 
                        orient=HORIZONTAL,
                        background='#1E93F0',
                        foreground='#FFFFFF',
                        troughcolor='#70BAFF',
                        activebackground='#1E93F0',
                        length=150)

    music_temperature.set(1)

    music_temperature.place(relx=.5, 
                    rely=.86, 
                    anchor="c")

    return music_temperature

music_temperature = music_temperature_slider()
music_temperature_text = Label(text='8. CHOOSE THE LEVEL OF CHANCE AND CONTINGENCY',
                        background='#F18C8E',
                        font=("Helvetica", 14))

music_temperature_text.place(relx=.5, 
                    rely=.81, 
                    anchor="c")


# music amount
def music_amount_slider():
    music_amount = Scale(root, 
                        from_=250, 
                        to=1000, 
                        resolution=250, 
                        orient=HORIZONTAL,
                        background='#1E93F0',
                        foreground='#FFFFFF',
                        troughcolor='#70BAFF',
                        activebackground='#1E93F0',
                        length=150)

    music_amount.set(1)

    music_amount.place(relx=.8, 
                    rely=.86, 
                    anchor="c")

    return music_amount

music_amount = music_amount_slider()
music_amount_text = Label(text='9. CHOOSE THE LENGTH OF THE MUSIC TO BE GENERATED',
                        background='#F18C8E',
                        font=("Helvetica", 14))

music_amount_text.place(relx=.8, 
                    rely=.81, 
                    anchor="c")


generate_music_button = Button(root, 
                            text="Generate and Play", 
                            command=generate_music,
                            background='#1E93F0', 
                            foreground="#FFFFFF",
                            activebackground='#1E93F0',
                            activeforeground='#FFFFFF',
                            width=15,
                            height=3)

generate_music_button.place(relx=.44, 
                        rely=.94, 
                        anchor="c")


replay_butoon = Button(root, 
                    text="Stop Playing", 
                    command=stop_music,
                    background='#1E93F0', 
                    foreground="#FFFFFF",
                    activebackground='#1E93F0',
                    activeforeground='#FFFFFF',
                    width=22,
                    height=3)

replay_butoon.place(relx=.56, 
                rely=.94, 
                anchor="c",
                width=110)


# progress bar
# style
progressbar_style = Style()
progressbar_style.theme_use('clam')
progressbar_style.configure('TProgressbar', 
                        troughcolor='#BADDFE', 
                        background='#1E93F0')

progress = Progressbar(root, 
                    orient = HORIZONTAL, 
                    length = 250,
                    style='TProgressbar',
                    mode = 'indeterminate') 

progress.place(relx=.5, 
            rely=.39, 
            anchor="c")



# user reaction radio button
reaction = StringVar()
reaction.set('happy')
c_major.configure(state = DISABLED)
c_major_p.configure(state = DISABLED)

def check_reaction():
    global note_density
    global musical_scale

    c_minor.configure(state = NORMAL)
    c_minor_p.configure(state = NORMAL)
    musical_scale.set('C Minor')

    if reaction.get() == 'happy':
        c_minor.configure(state = NORMAL)
        c_minor_p.configure(state = NORMAL)
        c_major.configure(state = DISABLED)
        c_major_p.configure(state = DISABLED)

        musical_scale.set('C Minor')
        # musical_scale = musical_scale(['C Major Scale', 'C Major Pentatonic Scale'])
        note_density = note_density_slider(5, 7)
    else:
        c_major.configure(state = NORMAL)
        c_major_p.configure(state = NORMAL)
        c_minor.configure(state = DISABLED)
        c_minor_p.configure(state = DISABLED)

        musical_scale.set('C Major')
        # musical_scale = musical_scale(['C Minor Scale', 'C Minor Pentatonic Scale'])
        note_density = note_density_slider(1, 3)

happy_reaction = Radiobutton(root, 
                    text='Happy', 
                    variable=reaction, 
                    value='happy',
                    command=check_reaction,
                    background='#F18C8E', 
                    activebackground='#F18C8E',
                    activeforeground='#F18C8E',
                    font=("Helvetica", 12))

sad_reaction = Radiobutton(root, 
                    text='Sad', 
                    variable=reaction, 
                    value='sad',
                    command=check_reaction,
                    background='#F18C8E', 
                    activebackground='#F18C8E',
                    activeforeground='#F18C8E',
                    font=("Helvetica", 12))

happy_reaction.place(relx=.45, 
                rely=.66,
                anchor="c")

sad_reaction.place(relx=.55, 
                rely=.66, 
                anchor="c")


reaction_text = Label(text='5. HOW DID THIS REVELATION BY THE ORACLE MAKE YOU FEEL?',
                        background='#F18C8E',
                        font=("Helvetica", 14))

reaction_text.place(relx=.5, 
                rely=.62, 
                anchor="c")