# Use official Python image as the base
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command (can be overridden in CI/CD)
CMD ["python", "src/predict.py"]
