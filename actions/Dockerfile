FROM rasa/rasa-sdk:3.6.0

USER root

# Install required packages
RUN pip install --no-cache-dir spacy==3.5.0

# Copy actions code
COPY ./actions /app/actions/

# Make the directory for models
RUN mkdir -p /app/models/resume_model

# Copy models for entity recognition (if needed by actions)
COPY ./models/resume_model /app/models/resume_model

USER 1001