# Python Virtual Environment - Pipenv

We will use pipenv for this project.

### How to install dependencies for the first time

```
pip install pipenv
```

Installing the dependencies

```
pipenv install numpy scikit-learn flask
```

You can indicate a particular version of your library if needed, for example scikit-learn==1.3.0

Upon installation the pipenv should create two file Pipfile and Pipfile.lock

If you use windows you can install waitress instead as gunicorn as your Web Server Gateway Interface (WSGI)

```
pipenv install waitress
```

### How to activate the environment

Two options

- First: 
```
pipenv shell
```
Following by 
```
waitress-serve --listen=0.0.0.0:9696 predict:app
```

- Second: combine the above commands into line 
```
pipenv run waitress-serve --listen=0.0.0.0:9696 predict:app
```

When you clone this repo you can simply run the code below in order to install all dependencies from the Pipfile and Pipfile.lock
```
pipenv install
```
