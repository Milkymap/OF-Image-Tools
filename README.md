# OF-Image-Tools
a python implementation of ngt, resnet vectorization and mbr-sift for image duplication search

# structure 
* [api-contextual-search]
* [api-image-encoder]
* [api-instance-matching]

# api-image-encoder
* json schema 
    * input schema 
        ```json
            {
                "request_id": "integer",
                "relative_path": "relative path to the source directory" 
            }
        ```
    * output schema
        ```json
            {
                "global_status": "0 | 1",
                "response": {
                    "request_id": "integer", 
                    "fingerprint": "List[Float]"
                }
            }
        ```

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
* json schema 
    * input schema 
        ```json
            {
                "nb_neighbors": "integer",
                "vec_features": "List[Float]"  
            }
        ```
    * output schema
        ```json
            {
                "global_status": "0 | 1",
                "error_message": "some message if an error was catched by the server",
                "response": {
                    "neighbors": [
                        {
                            "relative_path": "...", 
                            "score" : "float between [0, 1]"
                        }
                    ]
                }
            }
        ```

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
* json schema 
    * input schema 
        ```json
            {
                "candidate": "input image that we wana check if it is a duplication",
                "neighbors": "array of neighbors's paths"   
            }
        ```
    * output schema
        ```json
            {
                "global_status": "0 | 1",
                "error_message": "some message if an error was catched by the server",
                "response": {
                    "duplicated": [{
                        "relative_path": "relative path to duplicated image", 
                        "score": "mbr-sift score between 0 and 1"
                    }]
                }
            }
        ```

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