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

## example model

I've included an example model called `example_model.json`. 
The model is trained on the 2000 images of the dataset (resized to 128x128 pixels) plus 1272 real-world bowl/vase images I crawled from google images.

## predicting

To predict new samples with your trained classifier, run the following:
```
python -m pots.predict <model.json> <filenames>
```
This will print the coonfidence the classifier has for each class.

## data augmentation

All images in the given dataset have the bowls/vases in exactly the same placing. Since vases are higher, I suspect that some higher-up pixels are vastly more likely to be background in bowl images than in vase images. This is not the kind of thing we want our classifier to learn.
In addition, the dataset is quite small for the purposes of training a neural network.

To help with both, I added some data augmentation where the images are randomly resized and rotated during training/testing. This makes the classifier more likely to learn actual shape-like features, and gives us a lot of extra unique training data. To see some example transformed images, run:

```
python -m pots.data <datadir> -t
```

## crawler

I expanded the dataset with images of real bowls and vases crawled from google images. This was to see if the classifier could be made generic enough to recognize real-world images. To run it, use:

```
python -m pots.im_crawler -n <num_to_download> <out_dir> <query>
```

If you add crawled images to the png folders in the datadir, they will be added during training.

# Clay

The clay package trains a GAN model to generate new vases/bowls, based on the expression inputs.

## training
To train a classifier run the following:
```
python -m clay.train <datadir> <name> <class_name>
```

where `datadir` is the path to the assignment datadir (with bowl/vase subdirs), `name` is any name you want for the trained classifier, and `class_name` is the class you want to generate (`vase`, `bowl`, or `all`). This will save a file called `name.json` with the model description, `name.hdf5` with the network weights, and  `name.log` with the training log (accuracy in each epoch). It will also save the generator and discriminator model separately as `name.generator.hdf5` and `name.discriminator.hdf5`.

The training function has some more options, see them with:
```
python -m clay.train -h
```

## example models

I've added a trained model for bowls `dcgan_bowl.json` and one for vases `dcgan_vase.json`. You can use these to generate images (training a model takes a loooooooong time).

## generating

To generate new random bowl/vases with your trained model, run the following:
```
python -m clay.generate <model.json> <out_path> -n <num>
```
This will generate `num` samples and save them in `out_path`.

To generate a random bowl/vase based on one or more particular expressions, run the following:
```
python -m clay.generate <model.json> <out_path> -f <field_files>
```
This will use existing `.fields` values as input to generate vases/bowls.

To interpolate between a set of existing expressions, run the following:
```
python -m clay.generate <model.json> <out_path> -n <num> -f <field_files>
```
This will interpolate between the successive `.fields` files in `num` steps, and use the interpolated expressions as input to the model.
It will also interpolate from the last `.fields` back to the first one, so the images can be used in a loop.

The generate function has some more options, see them with:
```
python -m clay.generate -h
```

## retrieving more data from the api

There's also a little script to request more vase or bowl images from the borges API.
You can run it with:

```
python -m clay.get_more_data <datadir> <class> <num>
```
where `class` can be vase or bowl. This will interpolate `num` new bowls or vases.
It generates a new one by taking two random existing expressions and averaging the x/y values of the curve controls. 
For the required boolean values it randomly picks from one of the existing expressions.
Then a new `.bom` file and a `.fields` file is created, and a `.png` requested from the borges API.


