# Use an official Python 3.8.17 runtime as a parent image
FROM python:3.8.17

# Set the working directory to /app
WORKDIR /app

# Set the environment variable
ENV SUPPRESS_MA_PROMPT=1

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install -r ./requirements.txt

# Install dependencies for maddpg
WORKDIR /app/maddpg
RUN pip install -e .

# Install dependencies for multiagent-particle-envs (previously mpe)
WORKDIR /app/multiagent-particle-envs
RUN pip install -e .

# Reset the working directory to the root
WORKDIR /app/maddpg/experiments

# Define the command to run your Python application (inside the experiments directory)
CMD ["python", "train.py"]