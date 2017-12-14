# Borges assessment

This contains my submission for the borges machine learning assessment.

## pots

The pots package is the bowl/vase classifier.
To train a classifier run the following:

```
python -m pots.train <datadir> <name> 
```

where `datadir` is the path to the assignment datadir (with bowl/vase subdirs) and `name` is any name you want for the trained classifier. This will save a file called `name.hdf5` with the network weights, and  `name.log` with the training log (accuracy in each epoch).

The training function has some more options, see them with:
```
python -m pots.train -h
```

There's no prediction function yet, that's next on the list.

TBC
