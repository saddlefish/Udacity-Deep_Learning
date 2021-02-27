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

- Recurrent Neural Networks

  - Project 4: Generate TV Scripts

- Generative Adversarial Networks

  - Project 5: Generate Faces
  - Project 6: Improve Your LinkedIn Profile

- Deploying a Model

  - Deploying a Sentiment Analysis Model
    
