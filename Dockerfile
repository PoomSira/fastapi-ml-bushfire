# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install fastapi uvicorn scikit-learn==1.4

# Copy the machine learning model file
COPY BB_svm_model.pkl /app/

# Expose port 8000 for the API
EXPOSE 8000

# Define environment variable
ENV PORT=8000

# Run the FastAPI app with Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]