FROM python:3.12-slim

# Install system dependencies for pdfplumber (Poppler)
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . /app

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the outputs directory (add this line)
RUN mkdir -p /app/outputs

# Expose port (Render sets $PORT)
EXPOSE 8000

# Run the app
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
