# Classic ML vs Deep Learning
To explain, let's look at a problem: we want an ML model that looks at images of breast cancer and determines the survivability of the patient.

## Classic ML
1. Pool together experts in ML and breast cancer
2. In each image, the experts in breast cancer will identify what aspects of an image of breast cancer affect survivorship
3. The ML team then codes that into the model. These concepts that the model is learning are called "features"

- this approach is slower since humans need to do the identification manually

## Deep Learning

- this approach does not require us to identify the aspects of the images that affect survivorship
- the neural network identifies then builds the "features" for us

1. start off with a random neural network
2. feed it examples so it can learn to recognize patterns 
3. it creates features for itself (it learns on its own)

- the reason deep learning is so much faster is because we do not have to hand-code the features we need
- we just let the neural network build the features for us

# Image Recognizer
- can be used for other things aside from recognizing images
  - requires creativity

- can also be used for classifying sounds
  - this can be done by converting wave forms (sound) into images

- can also be used for classifying mouse movements
  - this can be done by converting the mouse movements into images
  - movements can be lines and clicks can be circles

- to train an image recognizer model, you don't need big images
- small images are enough to train the model
- moreover, big images take time to load into the RAM so there's no need for big images
- that's why resizing them makes a lot of sense before feeding them into the model

# Fast AI
- python package used for creating ml models
- library built on top of pytorch

