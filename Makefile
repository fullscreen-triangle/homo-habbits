# Makefile for Homo-Habits Development
# Provides convenient commands for common development tasks

.PHONY: help setup check test clean build lint format doc bench audit deps update release

# Default target
.DEFAULT_GOAL := help

# Colors for output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Project information
PROJECT_NAME := homo-habits
VERSION := $(shell grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)

## Display help information
help:
	@echo "$(BOLD)$(BLUE)Homo-Habits Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Setup:$(RESET)"
	@echo "  $(GREEN)setup$(RESET)        - Set up development environment"
	@echo "  $(GREEN)deps$(RESET)         - Install all dependencies"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  $(GREEN)check$(RESET)        - Run all checks (format, lint, test)"
	@echo "  $(GREEN)test$(RESET)         - Run all tests"
	@echo "  $(GREEN)lint$(RESET)         - Run all linters"
	@echo "  $(GREEN)format$(RESET)       - Format all code"
	@echo "  $(GREEN)doc$(RESET)          - Generate documentation"
	@echo ""
	@echo "$(BOLD)Building:$(RESET)"
	@echo "  $(GREEN)build$(RESET)        - Build all components"
	@echo "  $(GREEN)build-rust$(RESET)   - Build Rust components only"
	@echo "  $(GREEN)build-node$(RESET)   - Build Node.js components only"
	@echo ""
	@echo "$(BOLD)Quality:$(RESET)"
	@echo "  $(GREEN)bench$(RESET)        - Run performance benchmarks"
	@echo "  $(GREEN)audit$(RESET)        - Run security audits"
	@echo "  $(GREEN)coverage$(RESET)     - Generate test coverage reports"
	@echo ""
	@echo "$(BOLD)Maintenance:$(RESET)"
	@echo "  $(GREEN)clean$(RESET)        - Clean build artifacts"
	@echo "  $(GREEN)update$(RESET)       - Update dependencies"
	@echo "  $(GREEN)release$(RESET)      - Create a new release"
	@echo ""
	@echo "$(BOLD)Docker:$(RESET)"
	@echo "  $(GREEN)docker-build$(RESET) - Build Docker images"
	@echo "  $(GREEN)docker-up$(RESET)    - Start development services"
	@echo "  $(GREEN)docker-down$(RESET)  - Stop development services"

## Set up development environment
setup:
	@echo "$(BOLD)$(BLUE)Setting up development environment...$(RESET)"
	@chmod +x scripts/dev/setup.sh
	@./scripts/dev/setup.sh
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"

## Install all dependencies
deps:
	@echo "$(BOLD)$(BLUE)Installing dependencies...$(RESET)"
	@echo "$(YELLOW)Installing Rust dependencies...$(RESET)"
	@cargo fetch
	@echo "$(YELLOW)Installing Node.js dependencies...$(RESET)"
	@pnpm install --frozen-lockfile
	@echo "$(GREEN)✓ All dependencies installed!$(RESET)"

## Run all checks (format, lint, test)
check: format-check lint test
	@echo "$(GREEN)✓ All checks passed!$(RESET)"

## Run all tests
test:
	@echo "$(BOLD)$(BLUE)Running tests...$(RESET)"
	@echo "$(YELLOW)Running Rust tests...$(RESET)"
	@cargo test --workspace --all-features
	@echo "$(YELLOW)Running Node.js tests...$(RESET)"
	@pnpm run test
	@echo "$(GREEN)✓ All tests passed!$(RESET)"

## Run specific test suites
test-rust:
	@echo "$(BOLD)$(BLUE)Running Rust tests...$(RESET)"
	@cargo test --workspace --all-features --verbose

test-node:
	@echo "$(BOLD)$(BLUE)Running Node.js tests...$(RESET)"
	@pnpm run test

test-integration:
	@echo "$(BOLD)$(BLUE)Running integration tests...$(RESET)"
	@cargo test --test '*' --all-features

## Generate test coverage
coverage:
	@echo "$(BOLD)$(BLUE)Generating test coverage...$(RESET)"
	@echo "$(YELLOW)Rust coverage...$(RESET)"
	@cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@echo "$(YELLOW)Node.js coverage...$(RESET)"
	@pnpm run test:coverage
	@echo "$(GREEN)✓ Coverage reports generated!$(RESET)"

## Run all linters
lint: lint-rust lint-node
	@echo "$(GREEN)✓ All linting passed!$(RESET)"

lint-rust:
	@echo "$(BOLD)$(BLUE)Running Rust linters...$(RESET)"
	@cargo clippy --workspace --all-targets --all-features -- -D warnings

lint-node:
	@echo "$(BOLD)$(BLUE)Running Node.js linters...$(RESET)"
	@pnpm run lint

## Format all code
format: format-rust format-node
	@echo "$(GREEN)✓ All code formatted!$(RESET)"

format-rust:
	@echo "$(BOLD)$(BLUE)Formatting Rust code...$(RESET)"
	@cargo fmt --all

format-node:
	@echo "$(BOLD)$(BLUE)Formatting Node.js code...$(RESET)"
	@pnpm run format

## Check formatting without changing files
format-check: format-check-rust format-check-node
	@echo "$(GREEN)✓ All formatting checks passed!$(RESET)"

format-check-rust:
	@echo "$(BOLD)$(BLUE)Checking Rust formatting...$(RESET)"
	@cargo fmt --all -- --check

format-check-node:
	@echo "$(BOLD)$(BLUE)Checking Node.js formatting...$(RESET)"
	@pnpm run format:check

## Build all components
build: build-rust build-node
	@echo "$(GREEN)✓ All components built!$(RESET)"

build-rust:
	@echo "$(BOLD)$(BLUE)Building Rust components...$(RESET)"
	@cargo build --workspace --all-features

build-node:
	@echo "$(BOLD)$(BLUE)Building Node.js components...$(RESET)"
	@pnpm run build

## Build for release
build-release:
	@echo "$(BOLD)$(BLUE)Building for release...$(RESET)"
	@cargo build --workspace --all-features --release
	@pnpm run build:production
	@echo "$(GREEN)✓ Release build complete!$(RESET)"

## Generate documentation
doc: doc-rust doc-node
	@echo "$(GREEN)✓ All documentation generated!$(RESET)"

doc-rust:
	@echo "$(BOLD)$(BLUE)Generating Rust documentation...$(RESET)"
	@cargo doc --workspace --all-features --no-deps --document-private-items

doc-node:
	@echo "$(BOLD)$(BLUE)Generating Node.js documentation...$(RESET)"
	@pnpm run docs:build

## Open documentation in browser
doc-open:
	@echo "$(BOLD)$(BLUE)Opening documentation...$(RESET)"
	@cargo doc --workspace --all-features --no-deps --open

## Run performance benchmarks
bench:
	@echo "$(BOLD)$(BLUE)Running performance benchmarks...$(RESET)"
	@cargo bench --workspace

## Run security audits
audit: audit-rust audit-node
	@echo "$(GREEN)✓ Security audits complete!$(RESET)"

audit-rust:
	@echo "$(BOLD)$(BLUE)Running Rust security audit...$(RESET)"
	@cargo audit

audit-node:
	@echo "$(BOLD)$(BLUE)Running Node.js security audit...$(RESET)"
	@pnpm audit

## Update dependencies
update: update-rust update-node
	@echo "$(GREEN)✓ Dependencies updated!$(RESET)"

update-rust:
	@echo "$(BOLD)$(BLUE)Updating Rust dependencies...$(RESET)"
	@cargo update

update-node:
	@echo "$(BOLD)$(BLUE)Updating Node.js dependencies...$(RESET)"
	@pnpm update

## Check for outdated dependencies
outdated:
	@echo "$(BOLD)$(BLUE)Checking for outdated dependencies...$(RESET)"
	@echo "$(YELLOW)Rust dependencies:$(RESET)"
	@cargo outdated || echo "Install with: cargo install cargo-outdated"
	@echo "$(YELLOW)Node.js dependencies:$(RESET)"
	@pnpm outdated

## Clean build artifacts
clean:
	@echo "$(BOLD)$(BLUE)Cleaning build artifacts...$(RESET)"
	@cargo clean
	@rm -rf node_modules/*/dist/
	@rm -rf node_modules/*/build/
	@rm -rf browser_extension/dist/
	@rm -rf browser_extension/build/
	@pnpm run clean
	@echo "$(GREEN)✓ Build artifacts cleaned!$(RESET)"

## Full clean including dependencies
clean-all: clean
	@echo "$(BOLD)$(BLUE)Cleaning all dependencies...$(RESET)"
	@rm -rf node_modules/
	@rm -rf target/
	@rm -rf .pnpm-store/
	@echo "$(GREEN)✓ Everything cleaned!$(RESET)"

## Docker commands
docker-build:
	@echo "$(BOLD)$(BLUE)Building Docker images...$(RESET)"
	@docker-compose build
	@echo "$(GREEN)✓ Docker images built!$(RESET)"

docker-up:
	@echo "$(BOLD)$(BLUE)Starting development services...$(RESET)"
	@docker-compose up -d
	@echo "$(GREEN)✓ Development services started!$(RESET)"

docker-down:
	@echo "$(BOLD)$(BLUE)Stopping development services...$(RESET)"
	@docker-compose down
	@echo "$(GREEN)✓ Development services stopped!$(RESET)"

docker-logs:
	@echo "$(BOLD)$(BLUE)Showing service logs...$(RESET)"
	@docker-compose logs -f

## Development workflow
dev: docker-up
	@echo "$(BOLD)$(BLUE)Starting development environment...$(RESET)"
	@pnpm run dev

## Quick development check
quick-check:
	@echo "$(BOLD)$(BLUE)Running quick checks...$(RESET)"
	@cargo check --workspace
	@cargo clippy --workspace -- -D warnings
	@cargo test --workspace --lib
	@echo "$(GREEN)✓ Quick checks passed!$(RESET)"

## Pre-commit checks
pre-commit: format-check lint test
	@echo "$(GREEN)✓ Pre-commit checks passed!$(RESET)"

## Prepare for release
release-prep:
	@echo "$(BOLD)$(BLUE)Preparing for release...$(RESET)"
	@$(MAKE) clean
	@$(MAKE) deps
	@$(MAKE) check
	@$(MAKE) build-release
	@$(MAKE) doc
	@echo "$(GREEN)✓ Release preparation complete!$(RESET)"

## Create a new release
release:
	@echo "$(BOLD)$(BLUE)Creating release v$(VERSION)...$(RESET)"
	@$(MAKE) release-prep
	@echo "$(YELLOW)Run: git tag v$(VERSION) && git push origin v$(VERSION)$(RESET)"
	@echo "$(GREEN)✓ Ready for release!$(RESET)"

## Show project status
status:
	@echo "$(BOLD)$(BLUE)Project Status$(RESET)"
	@echo "Name: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo ""
	@echo "$(BOLD)Git Status:$(RESET)"
	@git status --porcelain
	@echo ""
	@echo "$(BOLD)Rust Toolchain:$(RESET)"
	@rustc --version
	@cargo --version
	@echo ""
	@echo "$(BOLD)Node.js Environment:$(RESET)"
	@node --version
	@pnpm --version

## Watch for changes and run tests
watch:
	@echo "$(BOLD)$(BLUE)Watching for changes...$(RESET)"
	@cargo watch -x check -x test

## Install development tools
install-tools:
	@echo "$(BOLD)$(BLUE)Installing development tools...$(RESET)"
	@cargo install cargo-watch cargo-audit cargo-outdated cargo-llvm-cov
	@echo "$(GREEN)✓ Development tools installed!$(RESET)"

## Initialize git hooks
init-hooks:
	@echo "$(BOLD)$(BLUE)Setting up git hooks...$(RESET)"
	@echo "#!/bin/sh" > .git/hooks/pre-commit
	@echo "make pre-commit" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)✓ Git hooks configured!$(RESET)" 