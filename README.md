# pokemonGAN
Generate pokemon sprites using a DCGAN

Training data is downloaded from https://veekun.com/dex/downloads

The network is trained for 450 epochs with a batch size of 32 
Note: the training process gifs are split into 3 because the gif is too large to be uploaded. The generated samples are different between the first 2 gifs and the last one because the latter was trained at a later time, causing the seed used to generate these images to be different.

This is an implementation of DCGAN following the steps from https://www.tensorflow.org/tutorials/generative/dcgan

<img src="/generated_fakes/0.png" width="200" height="200"> <img src="/generated_fakes/1.png" width="200" height="200"> <img src="/generated_fakes/2.png" width="200" height="200">

The generated images have the general form of a Pokemon sprite correct but may require more training epochs to get the details right.  
