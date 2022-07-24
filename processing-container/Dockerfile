
FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 install Pillow opencv-python tqdm
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
