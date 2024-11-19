## Building Central and HIE Images
This directory has 2 folders, named accordingly for Central and HIE code. Each of those folders contains a file named `dockerfile`, which is the build script for that code. All the code (and any data) is inside [central/app](central/app) and [hie/app](hie/app) folders.

To Build the images, cd into the folders and execute
```bash
docker build -t hie:latest .
```
for building the HIE Image and 
```bash
docker build -t central:latest .
```
for building the Central Image.

`hie:latest` and `central:latest` can be changed to any other name with the following syntax: `<image_name>:<image_version>`

## Running the MNIST test code
### HIE
To run the images, use the following command for HIE
```bash
docker run -it --rm --net=host --name HIE1 hie:latest python3 split_nn_hie.py
```
or follow the below syntax and replace <...> with relevant data
```
docker run -it --rm --net=host --name <container_name> <image_name> python3 split_nn_hie.py
```

### Central
Use below to run the central image
```bash
mkdir saves
docker run -it --rm --net=host --mount type=bind,source="$(pwd)"/saves,target=/app/saves --name central central:latest python3 split_nn_central.py
```

or use the following syntax
```
docker run -it --rm --net=host --mount type=bind,source=<local_saves_folder>,target=/app/saves  --name <container_name> <image_name> python3 split_nn_central.py
```

The program saves the model states in /app/saves folder iteration wise. Mounting it to a local folder will allow saving those models as well as some information locally.

### Notes
* `<custom_port>` should be same for all the nodes, unless custom port forwarding is setup, which is not done with the above commands.
* `<central_address>` is usually `localhost`
