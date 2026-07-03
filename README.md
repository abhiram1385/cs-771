# Hamming PUF Attack with Linear Models
CS771 — Introduction to Machine Learning, IIT Kanpur

## Overview
Designed a 288-dimensional feature map that makes Hamming PUF 
responses linearly separable, allowing a simple linear SVM to 
predict PUF responses with 99.76% test accuracy.

## Key Idea
The Hamming PUF computes h_e * h_o (product of even and odd 
Hamming weights of c XOR s). By expanding this product, we show 
it is linear in 288 features derived from the challenge bits alone:
- 256 cross-product features (even x odd bit pairs)
- 16 even-index bits
- 16 odd-index bits

## Results
- Feature dimensionality: D = 288
- Test accuracy: 99.76%
- Training time: ~0.12s
- Feature map time: ~0.05s

## Tech Stack
Python, NumPy, scikit-learn
