# My Handwritten-Style Digits Dataset

A small, MNIST-inspired dataset of digit images (0-9), 28x28 grayscale, generated
programmatically with randomized fonts, rotation, blur, and noise to introduce
handwriting-like variation -- similar in structure to the original MNIST dataset.

## Structure
- Folders 0/ through 9/  -> contain PNG images for each digit class
- labels.csv              -> MNIST-style CSV: label + 784 flattened pixel values per row

## Stats
- Classes: 10 (digits 0-9)
- Samples per class: 40
- Total images: 400
- Image size: 28x28 pixels, grayscale

## Notes
Images were synthetically generated (not scanned handwriting) to build a compact
practice dataset that mirrors MNIST's format for learning purposes.
