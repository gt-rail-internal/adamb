### Adaptive Mini-batching for Faster Convergence over complex data

This is the implementation of AdaMB training Tensorflow 

## Prerequisites
Following the instructions for Tensorflow on Docker
 - Have a GPU and an NVIDIA driver of >384.10
 - Install Docker CE
 - Install nvidia-docker2

## Run training
Using the Dockerfile provided with this repo, build the custon Docker container:
`docker build -f Dockerfile -t tf_models .`

Now using docker, run `train.py` over a container of the image you just built with the current directory mounted, for example:
`docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/ada_mb -it tf_models python /ada_mb/train.py`

_Caution_: Copy and pasting this direct *may* sometimes fail depending on your screen as Github or your text editor may insert a neline character.
