define PROJECT_HELP_MSG
Makefile to control project aml_dist
Usage:
    help                        show this message
    build                       build docker image to use as control plane
	bash                        run bash inside runnin docker container
    stop                        stop running docker container
endef
export PROJECT_HELP_MSG
PWD:=$(shell pwd)
PORT:=9999
TBOARD_PORT:=6006
IMAGE_NAME:=horovod_image
NAME:=horovod_container # Name of running container
DATA:=/mnt

BASEDIR:=$(shell dirname $(shell dirname $(shell dirname ${PWD})))
REPODIR:=$(shell dirname ${BASEDIR})

local_code_volume:=-v $(BASEDIR):/workspace
volumes:=-v $(DATA):/data \
		 -v ${HOME}/.bash_history:/root/.bash_history

help:
	echo "$$PROJECT_HELP_MSG" | less
	
build:
	docker build -t $(IMAGE_NAME) -f dockerfile ${REPODIR}

run: 
	@echo ${BASEDIR}
	# Start docker running as daemon
	docker run $(local_code_volume) $(volumes) $(setup_environment_file)  \
	--privileged \
	--shm-size="4g" \
	--runtime=nvidia \
	--name $(NAME) \
	-d \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-e HIST_FILE=/root/.bash_history \
	-it $(IMAGE_NAME) 

	docker exec -it $(NAME) bash 


run-horovod:
	docker exec -it $(NAME) mpirun -np 2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python train_horovod.py

bash:
	docker exec -it $(NAME) bash 

stop:
	docker stop $(NAME)
	docker rm $(NAME)

.PHONY: help build run bash stop