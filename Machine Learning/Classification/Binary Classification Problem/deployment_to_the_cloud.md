# Deployment To The Cloud - AWS Elastic Beanstalk

### To install on virtual environment

Development dependency (only required for development)
```
pipenv install awsebcli --dev
```

First we use virtual environment 
```
pipenv shell
```

following by

```
eb init -p docker -r eu-west-1 prject-serving
```

testing locally

```
eb local run --port 9696
```

Creating environment on the cloud

```
 eb create project-serving-env
```
