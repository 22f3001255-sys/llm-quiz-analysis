FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml .
COPY main.py .
COPY agent.py .
COPY shared.py .
COPY tools/ tools/
COPY .env .

# Install Python dependencies with uv
RUN pip install uv
RUN uv pip install --system -r pyproject.toml

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Create LLMFiles directory
RUN mkdir -p LLMFiles

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]