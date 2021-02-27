# Udacity-Deep_Learning

The repository is setup to house the exercises and projects that I complete while pursing the Deep Learning Nanodegree program from Udacity. The program curriculum was broken up into 6 parts which contained 7 main projects in addition to miscellaneous exercises completed throughout the program. 

The project and exercise content is as follows:
- Autoencoder
- Neural Networks:

  - Project 1: Predicting Bike-Sharing Data 
    - The purpose of the project is to build a neural network from scratch to carry out the prediction problem on the dataset. Once completed I'll have a better understanding of gradient descent, backpropagation, and other neural network properties. 
    - The data comes the UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    - Project/Network parameters to adhere to:
      - The activation function should be a sigmoid function
      - The number of epochs should be between 50 and 1500
      - The number of hidden nodes should be between 5 and 100
      - There should be exactly one output node 
      - The learning rate should be between 0.05 and 5
      - Produce good results when running the network on the full dataset, with requirements being:
        - Training loss should be less than 0.09 
        - Validation loss should be less than 0.18

- Convolutional Neural Networks:

  - Project 2: Dog Breed Classifier 
    - The purpose of this project is to build a convolutional neural network (CNN). For this project, I had to learn how to build a pipeline to process user-supplied images. Given this use case, it will be an image of a dog where the algorithm will identify an estimate of the canine's breed. If, however, an image of a human face is provided then the algorithm will attempt to identify the resembling dog breed. 
    - Project specifications:
      - Develop a dog_dector function that returns True if a dog is detected in an image and False if not. This will be accomplished by using the pre-trained VGG16 Network to find the predicted class for a given image. 
      - Create a CNN to Classify Dog Breeds (from scratch):
        - Write 3 separate data loaders for training, validation, and test datasets of dog images. These images should be pre-processed to be of the correct size, which will be accomplished using various transform techniques. 
        - Choose appropriate loss and optimization functions for this classification task. Train the model for a number of epochs and save the best result.
        - The trained model attains at least 10% accuracy on the test set. 
      - Create a CNN Using Transfer Learning:
        - Train the model for a number of epochs and save the result with the lowest validation loss
        - Achieve accuracy on the test set greater or equal to 60%

  - Project 3: Optimize You GitHub Profile
    - This is an optional project with the goal of helping one make sure that their profile is complete and professionally presented

- Recurrent Neural Networks

  - Project 4: Generate TV Scripts
    - The purpose of this project is to generate your own Seinfeld TV scripts using RNNs using a Seinfeld dataset of scripts from 9 seasons. The Neural Network you'll build will generate a new, "fake" TV script.
    - Project Requirements:
      - Pre-Processing Data
        - The function create_lookup_tables is implemented     
          - The function create_lookup_tables create's two dictionaries:
          - Dictionary to go from the words to an id, we'll call vocab_to_int
          - Dictionary to go from the id to word, we'll call int_to_vocab
          - The function create_lookup_tables return these dictionaries as a tuple (vocab_to_int, int_to_vocab)    
        - A special token dictionary is created
          - The function token_lookup returns a dict that can correctly tokenizes the provided symbols
      - Batching Data
        - Data is broken into sequences
          - The function batch_data breaks up word id's into the appropriate sequence lengths, such that only complete sequence lengths are constructed
        - Data is created using TensorDataset
          - In the function batch_data, data is converted into Tensors and formatted with TensorDataset
        - Data is batched correctly
          - batch_data returns a DataLoader for the batched training data
      - Build the RNN 
        - An RNN class has been defined
          - The RNN class has complete __init__, forward , and init_hidden functions
        - The RNN includes at least one LSTM (or GRU) and fully-connected layer
          - The RNN must include an LSTM or GRU and at least one fully-connected layer. The LSTM/GRU should be correctly initialized, where relevant.
      - RNN Training  
        - Reasonable hyperparameters are selected for training
          - Enough epochs to get near a minimum in the training loss, no real upper limit on this. Just need to make sure the training loss is low and not improving much with more training
          - Batch size is large enough to train efficiently, but small enough to fit the data in memory. No real “best” value here, depends on GPU memory usually
          - Embedding dimension, significantly smaller than the size of the vocabulary, if you choose to use word embeddings
          - Hidden dimension (number of units in the hidden layers of the RNN) is large enough to fit the data well. Again, no real “best” value
          - n_layers (number of layers in a GRU/LSTM) is between 1-3
          - The sequence length (seq_length) here should be about the size of the length of sentences you want to look at before you generate the next word
          - The learning rate shouldn’t be too large because the training algorithm won’t converge. But needs to be large enough that training doesn’t take forever
        - The model shows improvement during training
          - The printed loss should decrease during training. The loss should reach a value lower than 3.5
        - Question about hyperparameter choices is answered  
          - There is a provided answer that justifies choices about model size, sequence length, and other parameters
      - Generate the TV Script  
        - The generator code generates a script 
          - The generated script can vary in length, and should look structurally similar to the TV script in the dataset.

- Generative Adversarial Networks

  - Project 5: Generate Faces
    - The purpose of the project is to use generative adversarial networks to generate new images of faces. 
    - Project Requirements:
      - Data Loading and Processing
        - Has get_dataloader been implemented? 
          - The function get_dataloader should transform image data into resized, Tensor image types and return a DataLoader that batches all the training data into an appropriate size
        - Has the scale function been implemented? 
          - Pre-process the images by creating a scale function that scales images into a given pixel range. This function should be used later, in the training loop   
      - Build the Adversarial Networks 
        - Does the discriminator discriminate between real and fake mages?
          - The Discriminator class is implemented correctly; it outputs one value that will determine whether an image is real or fake
        - Does the generator generate fake mages?
          - The Generator class is implemented correctly; it outputs an image of the same shape as the processed training data
        - Is the weight initialization function implemented correctly? 
          -  This function should initialize the weights of any convolutional or linear layer with weights taken from a normal distribution with a mean = 0 and standard deviation = 0.02
      - Optimization Strategy
        - Are the real_loss and fake_loss functions implemented correctly?
          - The loss functions take in the outputs from a discriminator and return the real or fake loss 
        - Are appropriate optimizers defined for the networks? 
          - There are optimizers for updating the weights of the discriminator and generator. These optimizers should have appropriate hyperparameters. 
      - Training and Results 
        - Are all adversarial networks trained correctly?
          - Real training images should be scaled appropriately. The training loop should alternate between training the discriminator and generator networks
        - Do all models and optimizers have reasonable hyperparameters?
          - There is not an exact answer here, but the models should be deep enough to recognize facial features and the optimizers should have parameters that help wth model convergence 
        - Does the project generate realistic faces?
          - The project generates realistic faces. It should be obvious that generated sample images look like faces 
        - How could your model improve?
          - The question about model improvement is answered 
           
  - Project 6: Improve Your LinkedIn Profile
    - This is an optional project with the goal of helping one make sure that their profile is complete and professionally presented 

- Deploying a Model

  - Deploying a Sentiment Analysis Model
    - In this project you will construct a recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. You will create this model using Amazon's SageMaker service. In addition, you will deploy your model and construct a simple web app which will interact with the deployed model.    
    
