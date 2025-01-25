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

# sa,[;oming]
oh yeah you might want to be able to look at outputs from the model. haha.