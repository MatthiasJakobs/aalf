FROM python:3.10.13-slim

RUN apt update -y && apt install cm-super texlive-latex-extra lmodern git build-essential dvipng --no-install-recommends -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /aalf