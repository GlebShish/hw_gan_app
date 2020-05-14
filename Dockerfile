FROM pytorch/pytorch:latest

RUN pip install matplotlib \
    scipy \
    anvil-uplink

COPY . .

ENV TOKEN="5B2OO3NHELAIJYRUJHRYZ32M-3XJYGO3FV4QSYTYR"

CMD python main.py ${TOKEN}
