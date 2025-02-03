# forenote:
this is the third major patch or so!
we went from 'something like a nanogpt' to 'something a lot weirder than a nanogpt' in a few major movements.
the /data/ tools will prepare end-of-sequence delimited texts into parquet files, and the loader.py main training utility will train on them correctly.

attention-II is strange. if this was work done for an employer or a research lab, you would see a report before code.
instead, code before report. such is the way of the world.

finally, these models have been tested, and found compilable. compiling makes the models run good. you can train more model per second, (or serve more batches of user inputs per second) if you compile your models good. get in touch (you can find this account's handle on the search engines) if you are interested in hosting these models but do not instantly figure out your integrations from pure gnosis and skimming provided code.

# instructions for use:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
`pip install -r reqqies`
maybe you might need to
`pip install bitsandbytes`
but who can say for sure?

then, download a text corpus to `./data/`
tinystories, which can be found at https://huggingface.co/datasets/roneneldan/TinyStories/, is one such dataset.

`cd data`
`python prepare.py --trainfile TinyStoriesV2-GPT4-train.txt --valfile TinyStoriesV2-GPT4-valid.txt -p tinystories-gpt4`
`cd ..`
`python loader.py`.

but it would be for the best to examine the source and comments of 'loader.py' and 'prepare.py' first, wouldn't it?
a future commit might suggest a package of pre-binarized tinystories entries to enable a simpler and more carefree dry run.

# sampling:
oh yeah you might want to be able to look at outputs from the model. haha.

if you have such an inclination, you will need to edit the simplistic hardcodes in sampling.py to match your trained pgpt_lformer checkpoint.
edit the prompt.txt default sampling cue to literally anything you feel like if you want a less tinystories oriented autoregressive decoding chain.
sampler.py version 0.0 is more of an existence proof than a tool, if you can read this sentence you can write a better one!

# ATTENTION-II:
[redacted]

# LICENSE NOTICE: 
all code presented *without* license :)
do not construe the availability of this source code for authorization of any sort!
or in fact any warranty or guarantee to the behavior, meaning, appropriate deployment, or social value of the tools demonstrated herein.
get in touch personally if you think that code licensing is something that matters in your folkway, and we can work out something, together, which is better than what you originally had in mind. peace out~ <3

# UV:
prepend 'uv' basically.
`uv sync`
`uv run python loader.py`

...
actually there's some 'hell' to navigate too.
running
`uv add flash-attn==2.7.2.post1 --no-build-isolation`
took:
Prepared 1 package in 87m 25s
Installed 2 packages in 349ms.

# triton: 
...
...\VC\Auxiliary\Build>vcvars64.bat
...
if you enjoy compiling code, you will *love* running triton. expect a 4x reduction in gpu memory utilization and a 4x increase in training speed if you compile your models. however, compiling is literal; you must have a c++ compiler configured in your system. a lot of the project notes attached to this repository will guide you towards a combination of dependencies which *permit* compilation, but compilation is never a sure thing in contemporary computing.

edit the torch_compile flags in sample.py and loader.py to 'False' if configuring compilers isn't your jam.