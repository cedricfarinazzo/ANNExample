## 03 Handwritten Digits Recognition

### Source tree
```
.
├── Makefile
├── src/
│   ├── main.c
│   ├── nn.c
│   ├── nn.h
│   ├── dataset.c
│   ├── dataset.h
│   ├── sdl.c
│   ├── sdl.h
├── dataset/
│   │── training/
│       │── 60000 digits images for training (source: https://github.com/cedricfarinazzo/HandwrittenDatasetMNIST)
│   │── testing/
│       │── 60000 digits images for testing (source: https://github.com/cedricfarinazzo/HandwrittenDatasetMNIST)
│   │── custom/
│       │── 10 digits images

```

### Requirements

- gcc or clang
- make
- SDL\_image
- [ANN](https://gitlab.com/cedricfarinazzo/adaptativeneuralnetwork) installed

### How it works ?

So we use make to build this example according to the [01-make](../01-make) example.
View [Makefile](Makefile).


We use SDL\_image to load images from path.
As you can see, images path follows the next pattern: (digit)\_(arbitrary number).(extension) 
where digit is the digit drawn on the image. Thanks to that, it's easy to determine the number.
View [src/dataset.c](src/dataset.c).


For the neural network, we use a fully connected sigmoid neural network with 2 hidden layers.
The input layer contains 784 neurons because each images is 28x28, the first hidden layer 128 neurons, 
the second 64 neurons and the output layers 10 neurons because we have 10 digits...
The configuration of the neural network are saved and loaded for conf.nn
We use a learning rate between 0.05 and 06 and a momentum of 0.9 .
And finally we got a neural network with on 3.2% error on the testing dataset!
(actual conf.nn file)
```
Testing dataset (./dataset/testing): Loaded 10000 digit images in memory
Init neural network
Load network config from conf.nn
Testing...
Testing done
[Digit 0]: error : 10 / 980 (1.020408%)
[Digit 1]: error : 19 / 1135 (1.674009%)
[Digit 2]: error : 25 / 1032 (2.422481%)
[Digit 3]: error : 26 / 1010 (2.574257%)
[Digit 4]: error : 34 / 982 (3.462322%)
[Digit 5]: error : 32 / 892 (3.587444%)
[Digit 6]: error : 21 / 958 (2.192067%)
[Digit 7]: error : 58 / 1028 (5.642023%)
[Digit 8]: error : 55 / 974 (5.646817%)
[Digit 9]: error : 44 / 1009 (4.360753%)
TOTAL Error: 324 (3.240000%)
```
View [src/nn.c](src/nn.c).

And that's all!

### Arguments

- train the network
```
./main train <training_dataset_path> <testing_dataset_path>
```

- check the network
```
./main check <testing_dataset_path>
```

- get digit from image
```
./main get <image_path>
```

