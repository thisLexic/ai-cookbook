# What is a model?
- ML models fit functions to data
- specifically, the functions used are infinitely flexible such that it can do a particular thing (e.g. recognize patterns in input data)

### What does "fit functions to data" actually mean?
To explain how functions fit to data, let's take a look at quadratic functions and slowly connect them to the concept of fitting an infinitely flexible function to data.

##### Starting with Quadratic Function

1. We want to approximate the function `F` that maps `x` to `y`. It has some level of noise but the values more or less fit in a quadratic function. How does this relate to the AI model? Imagine if `x` were pictures of dogs or cats. `y` is the string dog or cat. `F` maps pictures of dogs/cats to the strings dog/cat.

2. This means, we want to identify the coefficients `a`, `b`, and `c` for the quadratic function `f` such that it approximates `F` well enough. This means we want an AI model that maps pictures of dogs/cats to the strings dog/cat well enough.

3. To do this, we iteratively and manually fiddle around with the values of `a`, `b`, and `c` in the function `f` then plug in various `x` values to check if `f(x)` is close to the `y` values from `F(x)`. Simply put, we fiddle around with our AI model's parameters until it classifies the dog/cat images sufficiently correctly.

4. We rigorously check for closeness of `y` values by defining a loss function. This determines whether our approximated `f(x)` is close to `F(x)`. A simple loss function would be something like mean squared error: We get all predictions and subtract them by each of their actual values. Afterwards, we square all of the differences. We get the mean of all of the squared differences. The higher this value is, the greater the loss, meaning the worse the model. So, if we increase one parameter and the loss increases, we better try decreasing that parameter to lower our loss.

```
loss = (preds - acts)^2.mean()
```

##### Automating Optimization of the parameters

5. Now that we get the idea, we want to automate fiddling with the parameters. One approach would be to programmatically fiddle with each parameter then check the loss each time then keep on fiddling and checking. 

6. However, there is a much faster way. 
- It is called getting the derivative or rate of change of the loss function with respect to the coefficients/parameters `a`, `b`, and `c`. 
- For example, if we take the derivative of the loss function with respect to `a`, then that value would tell us the rate of change of the loss. So if the derivative/gradient were negative, then that means increasing `a` will decrease the loss. We want loss to decrease since that makes the AI model better! 
- Therefore, in the given scenario, we want to increase `a` so that we can decrease the loss of our model to make it smarter! 
- As for how much to increase/decrease it by, there is no hard and fast rule but the tutorial just changes it by the arbitrary value of `0.01 * gradient` where `0.01` is called the learning rate.
- After some time though, the parameters will get really close to the right values so it will jump over/under the correct values since we are changing the parameters by too big of a leap (`0.01 * gradient`). That is why most frameworks decrease the value of the leaps/learning rate after a certain point.
- This is an implementation of the more general concept called Gradient Descent: We get the gradient (derivative) of the loss function and just decrease the loss based on the gradients.

##### Quadratics -> Rectified Linear Functions

7. So far, we've been trying to figure out parameters for our quadratic function. However, that function is not infinitely flexible such that it can map arbitrary data. Hence, we need to replace it with a function that is infinitely flexible such that it can approximate any arbitrary data. We need to replace it with a rectified linear unit: `ReLU`. This is simply two things:
    1. matrix multiplication - multiplying things then adding them up
    2. getting the max of two values
```
def ReLU(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.)
```
where `m` and `b` are the parameters and `x` is the input.

8. We can add together as many ReLUs as needed in order to approximate any computable function `F` such that we can map arbitrary data:
```
ReLF = ReLU(m1, b1, x) + ReLU(m2, b2, x) + ReLU(m3, b3, x) + ...
```
- `ReLF` or rectified linear function is an arbitrarily squiggly function that can approximate `F`
- Imagine an `ReLF` that has millions of `ReLU`s
- Note that each `ReLU` has its own `m` and `b`. These are initially randomized but they will get closer and closer to their correct values via gradient descent.
- if we were to make an image recognizer, each pixel would be its own `x` variable so you can imagine how big the ReLF gets:
```
ReLF = ReLU(m1, b1, pixelA) + ReLU(m2, b2, pixelA) + ReLU(m3, b3, pixelA) + ... +
       ReLU(m1, b1, pixelB) + ReLU(m2, b2, pixelB) + ReLU(m3, b3, pixelB) + ... +
       ReLU(m1, b1, pixelC) + ReLU(m2, b2, pixelC) + ReLU(m3, b3, pixelC) + ...
```    
- for a refresher on how we got here, you can look at the portion of this video (https://course.fast.ai/Lessons/lesson3.html) that explains how to create an AI model via Excel using linear regression

9. Summary: by adding more `ReLU`s; optimizing parameters with gradient descent; and samples of inputs and outputs we want, the computer creates the model for you.

##### How can we expand on this?

10. This above `ReLF` works for 1 input `x` but the same idea works for multiple inputs as well. With this foundation, we can construct any arbitrary model that is sufficiently accurate. 

11. Everything past these are just optimizations: making the learning faster, making it need less data, etc. Everything above is all we need to train a model, given enough time and enough data.

- The devil, I guess, is in the "given enough time and enough data" part of the above sentence. There's a lot of tweaks we can make to reduce both of these things. 
- For instance, instead of running our calculations on a normal CPU, as we've done above, we could do thousands of them simultaneously by taking advantage of a GPU. 
- We could greatly reduce the amount of computation and data needed by using a convolution instead of a matrix multiplication, which basically means skipping over a bunch of the multiplications and additions for bits that you'd guess won't be important. 
- We could make things much faster if, instead of starting with random parameters, we start with parameters of someone else's model that does something similar to what we want (this is called transfer learning).

Source: https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work

# Identifying the best pre-trained model
Note: looking for better pre-trained models is the last stage of the model dev process

### Option 1 (rarely do)
- create a model for each pre-trained model
- there are a lot of pre-trained models so this will take a long time
- you can do this by literally looping through each pre-trained model then training each on your data

### Option 2
- use a reference
- someone has already tested all of the pre-trained models for accuracy and speed of training. Just look at these references
- Ex: https://www.kaggle.com/code/jhoward/which-image-models-are-best/

- when you're still in the dev stage (data augmentation, data cleaning, dif external data), use the fast-to-train models
- when you're creating the finished model, use the better models
    - more accurate models since they have been pre-trained on more varied data
    - faster inference - meaning these models literally execute the neural network logic faster so you get the output faster. This may be due to less neurons

# Do I need more/better data
- you need to train your model on your existing data even if you might feel like there isn't enough data
    - this may show that you actually have enough data
    - this may show that you actually can't solve your problem with AI
    - it is really fast to train a prototype model so just train your model right off the bat

- you might not even need more/better data if you know how to get the most out of your existing data
    - semi-supervised learning - get dramatically more from your data
    - data augmentation

- consider cost/time when getting more/better data
    - labeled data is kinda easy to get online
    - sementation masks/pixel bounderies are expensive/time consuming

# Learning rate
- rate at which the model's parameters change
- in the example, this was `0.01`
- this is a hyperparameter - a parameter that calculates other parameters

- fastai generally picks a good default value for this
- you can set this manually which would mean digging for a good value

- a big learning rate is bad
    - because your parameters will jump too far forward and backward from the sweet spot again and again
    - this means that your parameters will diverge from the sweet spot
    - you'll see this happen when you train a model for awhile then the model gets worse and worse

- a small learning rate is bad
    - because it means reaching the sweet spot will take long
    - depending on how small the learning rate is, it may be take stupidly long to train your model
    - This is because as you get closer to the sweet spot, the smaller the gradient will be. This makes reaching the sweet spot by a step of `small_learning_rate * gradient` even smaller
    - you may make infinitely smaller steps towards the sweet spot until it computationally takes forever to reach the sweet spot

- findings a good learning rate is a compromise between
    - the possibility of shooting past the sweet spot and
    - taking a long time to reach the answer

