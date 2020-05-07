FROM pytorch/pytorch:latest

RUN pip install matplotlib /
    scipy \
    anvil-uplink

COPY . .

ARG TOKEN

CMD ["python", "main.py", ${TOKEN}]