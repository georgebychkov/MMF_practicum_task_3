FROM python:3.8.7-slim

COPY --chown=root:root src /root/src/
COPY --chown=root:root ensembles /root/src/ensembles

WORKDIR /root/src

RUN pip3 install --no-cache-dir -r requirements.txt; chmod +x run.py

ENV SECRET_KEY hello

CMD ["python", "run.py"]
