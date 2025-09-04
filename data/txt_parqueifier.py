#parqueshim.py

import os
import pandas as panddd
import csv
import numpy
import argparse
    
ddir = os.path.dirname(__file__)
tfile = r"txt\TinyStoriesV2-GPT4-valid.txt"
tprefix = str(os.path.splitext(tfile)[0])
filename = os.path.join(ddir,tfile)


tstories_sample = r"""<|endoftext|>u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>


Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
"Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
"No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
"Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
"Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
"We're okay, Mommy. But our toys are broken," Lily said.
"I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
"Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
<|endoftext|>

Once 
"""

"""
with open(filename, "w") as fille:
    fille.write(tstories_sample)
"""

delimiter_r=r'\<\|endoftext\|\>'
delimiter_regx = '/\<\|endoftext\|\>/gm'
delimiter_literal = r'<|endoftext|>'

"""
okay this does not work simply. 
it is a nontrivial text processing job to read text files linewise,
fuse line ends, and chunk along delimiter.
so it is time to write nontrivial text processing code!
"""


df = panddd.read_csv(filename, sep=delimiter_regx, skip_blank_lines=False, keep_default_na=False, header=None)
header_nommetext={0:"text"}
delimited = df.isin([delimiter_literal])
df = df.rename(columns=header_nommetext)

header_nommedelimit={0:"delimited"}
delimited = delimited.rename(columns=header_nommedelimit)

result = panddd.concat([delimited, df], axis=1)


index_of_delimits = result[result["delimited"]].index.values

print(len(index_of_delimits))
#print(index_of_delimits[1])


#we would have to preallocate memory and preshard and prechunk and stuff if we watned to whole numpy it.
folded_data=[]
dlabel="text"
nilstring=""
for metaidx in range(len(index_of_delimits)):
    subalter_data=[]
    #if index_of_delimits[metaidx]<df.index.values[metaidx]:
    #    first=0
    #    segund=index_of_delimits[metaidx]-1
    if metaidx<len(index_of_delimits)-1:
        first=index_of_delimits[metaidx]
        segund=index_of_delimits[metaidx+1]
    else:
        first=index_of_delimits[metaidx]
        segund=len(df)
    for subalterindex in range(first,segund):
        subalter_data.append(df.at[subalterindex,dlabel])
    if first==segund:
        subalter_data.append(df.at[first,dlabel])
        print(subalter_data)
    folded_data.append('\n'.join(subalter_data))


np_folded_data=numpy.array(folded_data)

#dataset is now a 1d nparray of multiline strings.
pd_dset = panddd.DataFrame(np_folded_data, columns=["text"])
print(pd_dset)

pqeterminator = ".parquet"
pqname = os.path.join(ddir,tprefix+pqeterminator)
print(pqname)
pd_dset.to_parquet(pqname)