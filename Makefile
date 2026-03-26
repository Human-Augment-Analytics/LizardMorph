.PHONY: setup setup-backend setup-frontend dev dev-backend dev-frontend clean download-models

# Process substitution in setup-backend requires bash; dash (default /bin/sh on Debian/Ubuntu) does not support it.
SHELL := /bin/bash

# Ensure uv from the official installer is on PATH when make does not load a login shell
export PATH := $(HOME)/.local/bin:$(PATH)

setup: setup-backend setup-frontend download-models
	@echo "Setup complete. You can now run 'make dev' to start the application."

download-models:
	@echo "Downloading models..."
	bash backend/scripts/download_models.sh

setup-backend:
	@echo "Setting up backend with uv..."
	cd backend && uv venv --clear && \
	uv pip install --python .venv/bin/python 'cmake>=3.28,<4' && \
	uv pip install --python .venv/bin/python -r <(grep -vE '^[[:space:]]*dlib' requirements.txt) && \
	uv pip install --python .venv/bin/python --no-build-isolation $$(grep -E '^[[:space:]]*dlib' requirements.txt | tr -d '[:space:]')

setup-frontend:
	@echo "Setting up frontend with npm..."
	cd frontend && npm install

dev:
	@echo "Starting backend and frontend concurrently..."
	# Using -j 2 to run the targets in parallel
	$(MAKE) -j 2 dev-backend dev-frontend

dev-backend:
	@cd backend && \
	PORT="$$(grep -E '^API_PORT=' ../.env 2>/dev/null | head -n1 | cut -d= -f2 | tr -d '\"' )" && \
	PORT="$${PORT:-3005}" && \
	if lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then \
		PIDS="$$(lsof -t -iTCP:"$${PORT}" -sTCP:LISTEN -nP | tr '\n' ' ')" && \
		CMDLINE="$$(ps -o cmd= -p $$(echo "$$PIDS" | awk '{print $$1}') 2>/dev/null || true)" && \
		if echo "$$CMDLINE" | grep -q gunicorn; then \
			echo "Stopping existing gunicorn on port $${PORT} (PIDs: $${PIDS})"; \
			kill $${PIDS} 2>/dev/null || true; \
			for i in 1 2 3 4 5 6 7 8 9 10; do \
				sleep 0.5; \
				if ! lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then break; fi; \
			done; \
			if lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then \
				echo "Gunicorn still listening; sending SIGKILL."; \
				kill -9 $${PIDS} 2>/dev/null || true; \
				sleep 0.5; \
			fi; \
		else \
			echo "Port $${PORT} is in use by a non-gunicorn process; refusing to kill it."; \
			lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP; \
			exit 1; \
		fi; \
	fi && \
	uv run python app.py

dev-frontend:
	@cd frontend && \
	PORT=5173 && \
	if lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then \
		PIDS="$$(lsof -t -iTCP:"$${PORT}" -sTCP:LISTEN -nP | tr '\n' ' ')" && \
		CMDLINE="$$(ps -o cmd= -p $$(echo "$$PIDS" | awk '{print $$1}') 2>/dev/null || true)" && \
		if echo "$$CMDLINE" | grep -Eq '(vite|npm run dev|node.*vite)'; then \
			echo "Stopping existing Vite dev server on port $${PORT} (PIDs: $${PIDS})"; \
			kill $${PIDS} 2>/dev/null || true; \
			for i in 1 2 3 4 5 6 7 8 9 10; do \
				sleep 0.5; \
				if ! lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then break; fi; \
			done; \
			if lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then \
				echo "Vite still listening; sending SIGKILL."; \
				kill -9 $${PIDS} 2>/dev/null || true; \
				sleep 0.5; \
			fi; \
		else \
			echo "Port $${PORT} is in use by a non-vite process; refusing to kill it."; \
			lsof -iTCP:"$${PORT}" -sTCP:LISTEN -nP; \
			exit 1; \
		fi; \
	fi && \
	npm run dev

clean:
	@echo "Cleaning up environments..."
	rm -rf backend/.venv
	rm -rf frontend/node_modules
