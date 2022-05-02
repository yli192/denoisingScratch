# denoisingScratch
Implementation of "A Noise-level-aware Framework for PET Image Denoising" in PyTorch
https://arxiv.org/pdf/2203.08034.pdf

## Training

First prepare patched dataset:

```bash
python make_patches.py 
```

Then run training looop!


```bash
python train.py --n_gpu [NUMBER OF GPUS FOR TRAINING] --conf config/diffusion.conf 
```
## Testing

```bash
python test.py --save_images
```
