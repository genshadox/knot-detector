FROM python:3.11.12-slim as base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \ 
    libglib2.0-0

WORKDIR /src

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Copy dependencies file and install
COPY /src/requirements.txt requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set permissions
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup && useradd -u ${UID} -g ${GID} -m appuser

# Copy the rest of the code (in dev, this is mounted over)
COPY --chown=appuser:appgroup src /src

# DEV
FROM base as dev

# For dev, lets install debugpy
RUN pip install --no-cache-dir debugpy

ENV PYDEVD_DISABLE_FILE_VALIDATION=1

# Debugpy port
EXPOSE 5678

USER appuser

CMD ["debugpy", "--listen", "5678", "--wait-for-client", "./main.py"]


# PROD
FROM base AS prod

USER appuser
 
CMD ["python", "/src/main.py"]
 