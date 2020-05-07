FROM pytorch/pytorch:latest

RUN pip install matplotlib \
    scipy \
    anvil-uplink

COPY . .

ENV TOKEN="change-it"

CMD python main.py ${TOKEN}
