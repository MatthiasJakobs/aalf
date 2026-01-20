# Use the official Python image from Docker Hub
FROM python:3.10
WORKDIR /app 

RUN apt update
RUN apt install git build-essential -y

RUN apt install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    cm-super

CMD ["/bin/bash"]
