# Semi-Amortized Variational Autoencoders
Code for the paper:  
[Semi-Amortized Variational Autoencoders](https://arxiv.org/pdf/1802.02550.pdf)  
Yoon Kim, Sam Wiseman, Andrew Miller, David Sontag, Alexander Rush

## Dependencies
The code was tested in `python 3.6` and `pytorch 0.2`. We also require the `h5py` package.

## Data
The raw datasets can be downloaded from [here](https://drive.google.com/file/d/1PecZKhrPkMZmvMyiOJfMS-FsHW3FE0rH/view?usp=sharing).

Text experiments use the Yahoo dataset from [Yang et al. 2017](https://arxiv.org/pdf/1702.08139.pdf), which is itself derived from [Zhang et al. 2015](https://arxiv.org/abs/1509.01626). 

Image experiments use the OMNIGLOT dataset [Lake et al. 2015](https://cims.nyu.edu/~brenden/LakeEtAl2015Science.pdf) with preprocessing from [Burda et al. 2015](https://arxiv.org/pdf/1509.00519.pdf).

Please cite the original papers when using the data.

## Text
After downloading the data, run
```
python preprocess_text.py --trainfile data/yahoo/train.txt --valfile data/yahoo/val.txt
--testfile data/yahoo/test.txt --outputfile data/yahoo/yahoo
```
This will create the `*.hdf5` files (data tensors) to be used by the model, as well as the `*.dict`
file which contains the word-to-integer mapping for each word.

The basic model command is
```
python train_text.py --train_file data/yahoo/yahoo-train.hdf5 --val_file data/yahoo/yahoo-val.hdf5
--gpu 1 --checkpoint_path model-path
```
where `model-path` is the path to save the best model and the `*.hdf5` files are obtained from running `preprocess_text.py`. You can specify which GPU to use by changing the input to the `--gpu` command.

To train the various models, add the following:  
- Autoregressive (i.e. language model): `--model autoreg`  
- VAE: `--model vae`  
- SVI: `--model svi --svi_steps 20 --train_n2n 0`  
- VAE+SVI: `--model savae --svi_steps 20 --train_n2n 0 --train_kl 0`  
- VAE+SVI+KL: `--model savae --svi_steps 20 --train_n2n 0 --train_kl 1`  
- SA-VAE: `--model savae --svi_steps 20 --train_n2n 1`  

Number of SVI steps can be changed with the `--svi_steps` command. 

To evaluate, run
```
python train_text.py --train_from model-path --test_file data/yahoo/yahoo-test.hdf5 --test 1 --gpu 1
```
Make sure the append the relevant model configuration at test time too.

## Images
After downloading the data, run
```
python preprocess_img.py --raw_file data/omniglot/chardata.mat --output data/omniglot/omniglot.pt
```

To train, the basic command is
```
python train_img.py --data_file data/omniglot/omniglot.pt --gpu 1 --checkpoint_path model-path
```

To train the various models, add the following:  
- Autoregressive (i.e. Gated PixelCNN): `--model autoreg`  
- VAE: `--model vae`  
- SVI: `--model svi --svi_steps 20`  
- VAE+SVI: `--model savae --svi_steps 20 --train_n2n 0 --train_kl 0`    
- VAE+SVI+KL: `--model savae --svi_steps 20 --train_n2n 0 --train_kl 1`  
- SA-VAE: `--model savae --svi_steps 20 --train_n2n 1`  

To evaluate, run
```
python train_img.py --train_from model-path --test 1 --gpu 1
```
Make sure the append the relevant model configuration at test time too.

## Acknowledgements
Some of our code is based on [VAE with a VampPrior](https://github.com/jmtomczak/vae_vampprior).

## License
MIT