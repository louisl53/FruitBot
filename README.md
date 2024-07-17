# FruitBot

## Name
FruitBot
Youtube demo : https://youtu.be/oyBFbdQEJt4

## Description

FruitBot is a chatbot that leverages the power of ChatGPT combined with a PyTorch algorithm to predict whether fruits and vegetables are good for consumption, while also suggesting suitable recipes based on the predictions.

The following technologies are used:

- LLM GPT-3.5 Turbo and 4o: Models used to respond to the user.
- Telegram: Messaging application that offers the possibility to create custom bots via an API (available on both computer and mobile).
- PyTorch: Python library for creating and training neural networks (specifically convolutional networks in this case).
- Kaggle: The dataset used to train the network is obtained via the Kaggle API.
- Docker: The application uses two Docker containers to function. The first container runs a FastAPI that houses the PyTorch neural network. The second container runs the Telegram bot, which can call the FastAPI to make predictions.

Using Docker simplifies the potential deployment of this application to the cloud (GCP, Azure, AWS).

## Environment Variables

Rename the ".env.example" file to ".env" and then modify the file to add your variables.

## Installation or prerequisites
- A GPT API key (https://openai.com/index/openai-api/)
- A Telegram bot token (https://core.telegram.org/bots/tutorial)
- Docker Desktop (https://www.docker.com/products/docker-desktop/)
- A Kaggle API key (https://christianjmills.com/posts/kaggle-obtain-api-key-tutorial/)

## Files
### api Folder
In this folder, you will find main.py, which is the script for building the FastAPI with the PyTorch model. The Dockerfile and requirements.txt allow the creation of the container image when the application is launched.

The notebook folder contains all the elements needed to build the PyTorch model:
- data_sep.py script, which retrieves the dataset from Kaggle and splits it into training and testing sets.
- dataset folder, which contains the data for training and testing the model.
- network.py script, which includes the architecture of the PyTorch model.
- The notebook that combines the data preparation and model creation parts.
- nn_model folder, which contains the final model used in the FastAPI.

### bot Folder
In this folder, you will find all the elements needed to build the Telegram bot. The Dockerfile and requirements.txt allow the creation of the container image when the application is launched.

- The Image_message folder temporarily holds images sent in the conversation.
- The bot.py script creates the Telegram bot and manages its functionalities.
- The agent_gpt.py script contains all the agents based on GPT-3.5 Turbo and GPT 4o. Each agent has a specific role.
 

## Application launch

To launch the application, run the command "docker compose up" in a command terminal within this directory (docker desktop must be open in parallel). You can then interact with the bot directly from the Telegram application on your computer or smartphone.

The notebook allows you to see the steps taken to build the PyTorch model. If you decide to make modifications or retrain the model, remember to adjust the batch size and the number of workers during training according to the power of your machine (CPU and GPU). The parameters set in the notebook are suited for a machine with a CPU i9 13900HX with 24 processor cores and an Nvidia RTX 4070 GPU with 8GB of VRAM. To obtain the dataset, please run the data_sep.py file, which will fetch the data from Kaggle and then split it into training and testing datasets.

The current model was trained with the help of the GPU using Nvidia CUDA. To use Nvidia CUDA, please refer to the following links:

Download nvidia cuda toolkit : 
https://developer.nvidia.com/cuda-toolkit
Download pytorch cuda (download the version corresponding to your nvidia toolkit) : 
https://pytorch.org/get-started/locally/
