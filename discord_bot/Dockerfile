# Discord Bot Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_discord.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_discord.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p discord_data results logs

# Create a non-root user to run the bot
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

# Switch to non-root user
USER botuser

# Run the bot
CMD ["python", "discord_bot.py"]