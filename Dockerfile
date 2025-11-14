# 1. Start from an official, lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory *inside* the container
WORKDIR /app

# 3. Copy the requirements list first (for better caching)
COPY requirements.txt .

# 4. Install all the Python libraries *inside* the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your local files (api.py, *.pkl) into the container
COPY . .

# 6. Tell Docker that the app will run on port 5000
EXPOSE 5000

# 7. The command to run when the container starts
#    This runs your `python api.py` command
CMD ["python", "api.py"]
