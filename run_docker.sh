
#!/bin/sh
# replace with your root folder absolute path
DATA_PATH='/data'
ROOT_PATH=$(realpath $(dirname $0))
echo "Test project path: $ROOT_PATH"
# deal with the permision problem
docker build -t juntao/qllm:0.1 .
CONT_NAME="qllm" 
# run if the docker container is not running but has been created
if [ "$(docker ps -aq -f name=$CONT_NAME)" ]; then
    echo "Container $CONT_NAME is running"
    docker start $CONT_NAME
    exit 0
fi
docker run --net=host --gpus all --name $CONT_NAME -v $DATA_PATH:/data -v $ROOT_PATH:/workspace/qllm -it juntao/qllm:0.1 

