# OF-Image-Tools
a python implementation of ngt, resnet vectorization and mbr-sift for image duplication search

# structure 
* [api-contextual-search]
* [api-image-encoder]
* [api-instance-matching]

# api-image-encoder
```bash
    docker build -t image-encoder:0.0 -f Dockerfile .
    docker run 
        --rm
        --name image-encoder
        -v path2images:/home/solver/images/
        -v path2models:/home/solver/models/
        -e MODELS_NAME=name_of_model_to_usr  # default is resnet18.th
        -e SERVER_PORT=8500
        -p host_port:8500
        image-encoder:0.0
``` 

# api-contextual-search
```bash
    docker build -t contextual-search:0.0 -f Dockerfile .
    docker run 
        --rm 
        --name contextual-search
        -v path2embeddings:/home/solver/location
        -e SERVER_PORT=8800
        -e BATCH_SIZE=1024
        -e DIMENSION=512
        -e DISTANCE_TYPE=Cosine
        -p host_port:8800
        contextual-search:0.0
```

# api-instance-matching
```bash
    docker build -t instance-matching:0.0 -f Dockerfile .
    docker run
        --rm
        --name instance-matching
        -v path2input_volume:/home/solver/source/
        -v path2target_volume:/home/solver/target/
        -e SERVER_PORT=8300
        -e THRESHOLD=0.75
        -e WIDTH=128
        -e HEIGHT=128
        -e NB_WORKERS=8
        -p host_port:8300
        instance-matching:0.0
```