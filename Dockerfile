FROM rasa/rasa:3.6.2

# Install spaCy and other dependencies
USER root
RUN pip install spacy==3.5.0 

# Copy configuration files
COPY config.yml endpoints.yml credentials.yml domain.yml /app/
COPY ./data /app/data/

# Create necessary directories and copy custom components
COPY ./custom_components /app/custom_components

# Copy actions directory
COPY ./actions /app/actions/

# Make sure the model directory exists
RUN mkdir -p /app/models/resume_model

# Copy any models if they exist locally
COPY ./models/resume_model /app/models/resume_model

# For debugging
RUN pip install ipython

WORKDIR /app