# forenote:
initial patch is really not pleasant to work through.
fixing problems with distributed training library calls meant vivisecting much of each implementation, and a lot of the code should be trimmed.
and that's what further commits are for!
the first working release of a trainer should train and train alone!

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