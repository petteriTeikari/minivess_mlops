# DVC 

Init DVC for this repo (be in the root of repo)

```commandline
dvc init
```

Add storage within the repo
The symlink trick does not work with DVC
`sudo ln -s ~/minivess_mlops_artifacts/data/d-bf268b89-1420-476b-b428-b85a913eb523 data`

So at this point of the DVC demo, the python script have copied (duplicated the data to the repo as well)

```commandline
dvc remote add -d remote_storage data
```

Add the files:

```commandline
dvc add data
```

Push these to DVC

```commandline
push dvc
```

## Advanced

From https://realpython.com/python-data-version-control/

This raises two questions:

* Doesn’t copying files waste a lot of space?
* Can you put the cache somewhere else?

The answer to both questions is yes. You’ll work through both of these issues in the section [Share a Development Machine.](https://realpython.com/python-data-version-control/#share-a-development-machine)