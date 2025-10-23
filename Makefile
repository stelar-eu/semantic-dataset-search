
DOCKER=docker
IMGTAG=petroud/semantic-dataset-search:latest
IMGPATH=.
DOCKERFILE=$(IMGPATH)/Dockerfile
.PHONY: all build push


all: build push

build:
	$(DOCKER) build -f $(DOCKERFILE) $(IMGPATH) -t $(IMGTAG)

push:
	$(DOCKER) push $(IMGTAG)

