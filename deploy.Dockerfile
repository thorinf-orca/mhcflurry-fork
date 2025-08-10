FROM python:3.11.6-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc wget && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt
RUN useradd --create-home --shell /bin/bash appuser
RUN mkdir /tmp/mhcflurry-downloads
COPY mhcflurry/downloads.yml /tmp/mhcflurry-downloads/
RUN python -c 'import yaml, subprocess; d = yaml.safe_load(open("/tmp/mhcflurry-downloads/downloads.yml")); current_release = d["releases"][d["current-release"]]; downloads = current_release["downloads"]; urls = []; [urls.extend(item["part_urls"]) if "part_urls" in item else urls.append(item["url"]) for item in downloads if item.get("default", False)]; [subprocess.check_call(["wget", "-P", "/tmp/mhcflurry-downloads", url]) for url in urls]'
COPY --chown=appuser:appuser . .
RUN pip install -e .
USER appuser
RUN mhcflurry-downloads fetch --already-downloaded-dir /tmp/mhcflurry-downloads
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api_server:app"]