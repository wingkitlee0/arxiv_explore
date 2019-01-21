# Exploration of astroph dataset

## Data
The data are collected by *arxiv_collector*


## Deployment

We use AWS to host this project. For large files, we use the Amazon S3.

To upload the files, we first create a `s3 bucket`:
```
aws s3 mb s3://ml-bucket-3
```
Here `ml-bucket-3` is the bucket name. We should also include this name in our `config.json` such that our script can get back our files on s3.




