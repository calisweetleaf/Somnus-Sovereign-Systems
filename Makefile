SHELL := /bin/bash

.PHONY: diagrams diagrams-svg diagrams-clean check-mmdc install-mermaid-cli

check-mmdc:
	@command -v mmdc >/dev/null 2>&1 || { \
		echo "ERROR: 'mmdc' (Mermaid CLI) not found."; \
		echo "Install Node and Mermaid CLI:"; \
		echo "  npm install -g @mermaid-js/mermaid-cli@10"; \
		exit 1; \
	}

install-mermaid-cli:
	@echo "Installing Mermaid CLI globally via npm..."
	@echo "Run: npm install -g @mermaid-js/mermaid-cli@10"

# Render all Mermaid diagrams in docs/diagrams to PNG
# Outputs go to docs/diagrams/img and sources to docs/diagrams/mmd
# Requires: Python, Node, Mermaid CLI (mmdc)
diagrams: check-mmdc
	@python scripts/render_mermaid.py --format png --theme default

# Render all diagrams to SVG instead of PNG
diagrams-svg: check-mmdc
	@python scripts/render_mermaid.py --format svg --theme default

# Remove generated images and temp .mmd sources
diagrams-clean:
	@rm -f docs/diagrams/mmd/*.mmd 2>/dev/null || true
	@rm -f docs/diagrams/img/*.png 2>/dev/null || true
	@rm -f docs/diagrams/img/*.svg 2>/dev/null || true

