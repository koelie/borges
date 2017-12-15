# Borges assessment

This contains my submission for the borges machine learning assessment.

## installation

This assumes OpenCV and python3.5 (and Cuda for gpu training) are already installed (e.g. like on anaconda)

Install the pip dependencies (Keras, tensorflow, etc) with:
```
pip install -r requirements.txt
```

# Pots

The pots package is the bowl/vase classifier.

## training
To train a classifier run the following:
```
python -m pots.train <datadir> <name> 
```

where `datadir` is the path to the assignment datadir (with bowl/vase subdirs) and `name` is any name you want for the trained classifier. This will save a file called `name.json` with the model description, `name.hdf5` with the network weights, and  `name.log` with the training log (accuracy in each epoch).

The training function has some more options, see them with:
```
python -m pots.train -h
```

## predicting

To predict new samples with your trained classifier, run the following:
```
python -m pots.predict <model.json> <filenames>
```

## additional info

### data augmentation

All images in the given dataset have the bowls/vases in exactly the same placing. Since vases are higher, I suspect that some higher-up pixels are vastly more likely to be background in bowl images than in vase images. This is not the kind of thing we want our classifier to learn.
In addition, the dataset is quite small for the purposes of training a neural network.

To help with both, I added some data augmentation where the images are randomly resized and rotated during training/testing. This makes the classifier more likely to learn actual shape-like features, and gives us a lot of extra unique training data. To see some example transformed images, run:

```
python -m pots.data <datadir> -t
```

### crawler

I expanded the dataset with images of real bowls and vases crawled from google images. This was to see if the classifier could be made generic enough to recognize real-world images. To run it, use:

```
python -m pots.im_crawler -n <num_to_download> <out_dir> <query>
```

If you add crawled images to the png folders in the datadir, they will be added during training.

### example model

I've included an example model called `example_model.json`. For this I crawled 1272 real-world bowl/vase images and added them to the dataset.
The model is trained on 2454 images of the dataset (resized to 128x128 pixels), and achieves 82% accuracy on the remaining unused 818 images.
Which I'd say is pretty good given the limited training data, and large variety of the real-world images (I didn't have a chance to check them all, I'm sure there's a significant number of nonsense images in there).

