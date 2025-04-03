# Use an official Python image as the base
FROM python:3.11.0

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# # Run the Flask application
# CMD ["python", "main.py"]

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]