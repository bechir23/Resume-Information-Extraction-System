FROM python:3.9-slim

WORKDIR /app

# Install Flask with compatible Werkzeug version
RUN pip install --no-cache-dir flask==2.2.3 werkzeug==2.2.3 requests==2.28.2

# Copy the application
COPY . /app/

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]