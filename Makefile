APP_NAME=ivimreduction-docker
PORT=3333

# Build the container
build: ## Build the container
	docker build -t $(APP_NAME):latest .
	
run: ## Run container on port configured in `config.env`
	docker run -i -t --rm -p=$(PORT):$(PORT) --name="$(APP_NAME)" $(APP_NAME)

commit: ## Commit changes to image
	docker commit $(APP_NAME) $(APP_NAME)
