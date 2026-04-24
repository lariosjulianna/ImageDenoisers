# Image Denoising: CNN and Transfer Learning Autoencoders

Built and compared two deep learning models for image denoising using a 
fruit and vegetable image dataset (36 classes, ~100 training images per class).

## Models

**Deep Convolutional Denoising Autoencoder**
- 10-layer encoder-decoder architecture (5 conv layers + 3 max pooling / 
5 conv layers + 3 upsampling)
- Regularization: dropout (0.2), L2 weight decay (λ=0.001), batch normalization

**Transfer Learning Denoising Autoencoder (VGG16)**
- Pretrained VGG16 encoder (frozen) connected to a custom decoder
- Achieved a 44% MSE reduction over the noisy baseline, outperforming 
the custom CNN (25% MSE reduction)

## Results

| Model | Test MAE | Test MSE | MSE Improvement |
|---|---|---|---|
| Custom CNN | 0.2261 | 0.0663 | 25.75% |
| VGG16 Transfer Learning | 0.1666 | 0.0497 | 44.29% |

## Limitations
- Models trained for 5 epochs due to compute constraints — 
further training would likely improve performance based on validation trends

## How to Run

Data is downloaded within the notebook via the Kaggle API.
You will be prompted to upload your `kaggle.json` credentials.

```bash
jupyter nbconvert --to script ImageDenoiser.ipynb
python ImageDenoiser.py
```