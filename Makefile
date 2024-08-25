# ---------- Python Configuration
.PHONY: setup
setup: ## create a venv and run install requirements
	@if [ ! -e .venv ]; then \
		python -m venv venv; \
		. venv/bin/activate; \
		pip install -r requirements.txt; \
	fi
	@echo "Environment created"
	@wget "https://download.pytorch.org/models/vgg16-397923af.pth"
	@echo "Model downloaded"
	@mkdir -p data
	@echo "data folder created"

#---------- Data treatment
.PHONY: data
data: # download images from GCP buckets
	@gsutil -m cp -r \
  		"gs://verifymy-ai-trainning/research-dataset/images/age-regression"
	@echo "Download complete"

#---------- Train model
.PHONY: run
run:
	@python train.py

.PHONY: upload
upload: # upload model.pah to bucket
	@gsutil cp vgg_model.pth gs://verifymy-ai-trainning/
