FROM python:3.10.13-slim

RUN apt update -y
RUN apt install git build-essential -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt install texlive-latex-extra -y

WORKDIR /aalf