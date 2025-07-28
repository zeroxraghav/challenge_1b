# Use an official lightweight Python image.
FROM --platform=linux/amd64 python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (this happens only once during build)
RUN python -m nltk.downloader stopwords punkt

# Copy the rest of the application code
# This will copy your 'Collection X' folders and the script
COPY . .

# This command runs when the container starts
CMD ["python", "process_1b.py"]
