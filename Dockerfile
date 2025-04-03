# Use an official Python image as the base
FROM python:3.11.0

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies (optional but helps with missing packages)
RUN apt-get update && apt-get install -y git

# # Upgrade pip to avoid warnings
# RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port Flask will run on
EXPOSE 8000

# # Run the Flask application
# CMD ["python", "main.py"]

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]