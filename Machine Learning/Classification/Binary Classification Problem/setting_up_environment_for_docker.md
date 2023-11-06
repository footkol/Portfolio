# Environment Management - Docker
```
winpty docker run -it --rm python:3.11-slim
```
run the above code to get a python image with Docker

Dockerfile specifies the commands we want to execute and run inside the container. 


### To build a docker image

```
winpty docker build -t project .
```


### To run a docker image 

```
winpty  docker run -it --rm --entrypoint=bash project
```

### Mapping the port with the port on local host machine

```
winpty docker run -it --rm -p 9696:9696 project
```
