.PHONY: setup setup-backend setup-frontend dev dev-backend dev-frontend clean download-models

setup: setup-backend setup-frontend download-models
	@echo "Setup complete. You can now run 'make dev' to start the application."

download-models:
	@echo "Downloading models..."
	bash backend/scripts/download_models.sh

setup-backend:
	@echo "Setting up backend with uv..."
	cd backend && uv venv && uv pip install -r requirements.txt

setup-frontend:
	@echo "Setting up frontend with npm..."
	cd frontend && npm install

dev:
	@echo "Starting backend and frontend concurrently..."
	# Using -j 2 to run the targets in parallel
	$(MAKE) -j 2 dev-backend dev-frontend

dev-backend:
	cd backend && uv run python app.py

dev-frontend:
	cd frontend && npm run dev

clean:
	@echo "Cleaning up environments..."
	rm -rf backend/.venv
	rm -rf frontend/node_modules
