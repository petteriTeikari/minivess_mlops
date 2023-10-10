# DVC 

## Create the DVC repo for the `minivess` dataset

Init DVC for this repo (be in the root of repo)

```commandline
dvc init
```

Add S3 remote:

```commandline
dvc remote add -d remote_storage s3://minivessdataset
```

Add the files:

```commandline
dvc add data
```

Push these to DVC

```commandline
push dvc
```

## Local debug

If you don't have access to the S3 (or have your own buckets set up), you could use a local data storage as well

Add storage within the repo
The symlink trick does not work with DVC
`sudo ln -s ~/minivess_mlops_artifacts/data/d-bf268b89-1420-476b-b428-b85a913eb523 data`

So at this point of the DVC demo, the python script have copied (duplicated the data to the repo as well)

```commandline
dvc remote add -d remote_storage data
```

And you can later change the remote location with [`remote modify`](https://dvc.org/doc/command-reference/remote/modify), for example to your [S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3):

```commandline
dvc remote modify remote_storage url s3://minivessdataset
```

And you can delete the folder `data`, and re-download the data with `dvc pull`

## Advanced

### Copying files

From https://realpython.com/python-data-version-control/

This raises two questions:

* Doesn’t copying files waste a lot of space?
* Can you put the cache somewhere else?

The answer to both questions is yes. You’ll work through both of these issues in the section [Share a Development Machine.](https://realpython.com/python-data-version-control/#share-a-development-machine)

### Continuous Integration and Deployment for Machine Learning

* DVC can manage data/models and reproducible pipelines, while [CML](https://cml.dev/) can assist with orchestration, testing and monitoring.

**Low friction:** Our sister project CML provides [lightweight machine resource orchestration](https://cml.dev/doc/self-hosted-runners) that lets you use pre-existing infrastructure. DVC and CML both provide abstraction/codification and require no external services.

### Data registry

https://dvc.org/doc/use-cases/data-registry
-> Try our [registry tutorial](https://dvc.org/doc/use-cases/data-registry/tutorial)  to learn how DVC looks and feels firsthand.

We can build a DVC project dedicated to versioning datasets (or data features, 
[ML models](https://dvc.org/doc/use-cases/model-registry), etc.). The repository contains 
the necessary metadata, as well as the entire change history. The data itself is stored in 
one or more [DVC remotes](https://dvc.org/doc/user-guide/data-management/remote-storage). 
This is what we call a **data registry** — data management middleware between ML projects and cloud storage.