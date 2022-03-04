FROM ubuntu

RUN apt-get update && apt-get install python3 pip -y

WORKDIR /ivimreduction

ADD . /ivimreduction

CMD ["/bin/bash"]
