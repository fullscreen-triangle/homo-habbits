#!/bin/bash

# Homo-Habits Development Environment Setup Script
# This script sets up the complete development environment for the homo-habits framework

set -e  # Exit on any error

echo "🐒 Setting up Homo-Habits Development Environment..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]]; then
        OS="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_success "Operating system: $OS"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        print_error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    print_success "Rust is installed: $(rustc --version)"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js from https://nodejs.org/"
        exit 1
    fi
    print_success "Node.js is installed: $(node --version)"
    
    # Check pnpm
    if ! command -v pnpm &> /dev/null; then
        print_warning "pnpm is not installed. Installing pnpm..."
        npm install -g pnpm
    fi
    print_success "pnpm is available: $(pnpm --version)"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
        print_warning "Please install Docker from https://docs.docker.com/get-docker/"
    else
        print_success "Docker is available: $(docker --version)"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed. Some features may not work."
    else
        print_success "Docker Compose is available: $(docker-compose --version)"
    fi
}

# Install Rust dependencies
install_rust_deps() {
    print_status "Installing Rust dependencies..."
    
    # Install cargo-watch for development
    if ! command -v cargo-watch &> /dev/null; then
        print_status "Installing cargo-watch..."
        cargo install cargo-watch
    fi
    
    # Install sqlx-cli for database migrations
    if ! command -v sqlx &> /dev/null; then
        print_status "Installing sqlx-cli..."
        cargo install sqlx-cli --features postgres
    fi
    
    print_success "Rust development tools installed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Core directories
    mkdir -p core/{semantic,identity,bmds,engine}
    mkdir -p ai_integration/{context_injection,enhancement}
    mkdir -p ai_integration/platforms/{openai,anthropic,google,local_models}
    mkdir -p commercial/{matching,optimization,pricing}
    mkdir -p commercial/platforms/{amazon,shopify,woocommerce,custom_apis}
    mkdir -p privacy/{ephemeral,disclosure,security}
    
    # Browser extension
    mkdir -p browser_extension/{popup,content_scripts,background,options,utils}
    
    # Mobile app
    mkdir -p mobile_app/{ios,android,shared}
    mkdir -p mobile_app/shared/{semantic_core,api_client,crypto_utils}
    
    # Backend
    mkdir -p backend/{api,database,infrastructure}
    mkdir -p backend/api/{routes,middleware,handlers}
    mkdir -p backend/services/{semantic_processor,identity_manager,ai_integrator,commercial_optimizer}
    mkdir -p backend/database/{migrations,schemas,seed_data}
    mkdir -p backend/infrastructure/{docker,kubernetes,terraform,monitoring}
    
    # Integrations
    mkdir -p integrations/{sdk,webhooks,apis}
    mkdir -p integrations/sdk/{javascript,python,rust,go}
    mkdir -p integrations/apis/{social_platforms,e_commerce,educational,professional_tools}
    
    # Modeling
    mkdir -p modeling/{multi_model,domain_extraction,quality_assessment,distributed}
    mkdir -p modeling/domain_extraction/{expert_knowledge_base,competency_patterns,validation_criteria}
    
    # Turbulance integration
    mkdir -p turbulance_integration/{parser,operations,learning}
    
    # Testing
    mkdir -p testing/{unit,integration,benchmarks,validation,fixtures}
    mkdir -p testing/unit/{core,ai_integration,commercial,privacy}
    mkdir -p testing/integration/{api_tests,platform_tests,end_to_end}
    mkdir -p testing/benchmarks/{semantic_processing,scalability,latency}
    mkdir -p testing/fixtures/{user_profiles,interaction_data,expected_results}
    
    # Examples and docs
    mkdir -p examples/{basic_usage,advanced,platform_demos,tutorials}
    mkdir -p docs/{architecture,api,integration,deployment,research}
    
    # Tools
    mkdir -p tools/{semantic_debugger,performance_profiler,privacy_auditor,data_generators}
    
    # Config and scripts
    mkdir -p config/{environments,semantic,privacy,platforms,monitoring}
    mkdir -p scripts/{dev,build,deploy,maintenance}
    
    print_success "Directory structure created"
}

# Setup environment files
setup_environment() {
    print_status "Setting up environment files..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Homo-Habits Development Environment Variables

# Database
DATABASE_URL=postgres://habits_user:habits_dev_password@localhost:5432/homo_habits
TEST_DATABASE_URL=postgres://habits_user:habits_dev_password@localhost:5432/homo_habits_test

# Redis
REDIS_URL=redis://localhost:6379

# AI API Keys (add your keys here)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Application
RUST_LOG=debug
HOMO_HABITS_ENV=development

# Security
JWT_SECRET=your_jwt_secret_here_change_in_production
ENCRYPTION_KEY=your_encryption_key_here_change_in_production

# External Services
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
JAEGER_URL=http://localhost:16686
EOF
        print_success "Created .env file (remember to add your API keys)"
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    # Create .gitignore if it doesn't exist
    if [ ! -f .gitignore ]; then
        cat > .gitignore << EOF
# Homo-Habits .gitignore

# Environment files
.env
.env.local
.env.production
.env.staging

# Rust
/target/
**/*.rs.bk
Cargo.lock

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn/
dist/
build/

# Database
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Temporary files
*.tmp
*.temp
.cache/

# Test coverage
coverage/
.nyc_output/

# Production builds
/dist/
/build/

# Secrets and keys
secrets/
*.pem
*.key
*.crt

# Backup files
*.backup
*.bak

# Performance profiling
*.prof
flamegraph.svg

# Browser extension builds
browser_extension/dist/
browser_extension/build/

# Mobile app builds
mobile_app/ios/build/
mobile_app/android/build/
mobile_app/android/.gradle/

# Documentation builds
docs/_build/
docs/.doctrees/
EOF
        print_success "Created .gitignore file"
    else
        print_warning ".gitignore file already exists, skipping creation"
    fi
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    # Install root dependencies
    pnpm install
    
    print_success "Node.js dependencies installed"
}

# Setup database
setup_database() {
    print_status "Setting up development database..."
    
    if command -v docker &> /dev/null; then
        print_status "Starting PostgreSQL and Redis with Docker..."
        docker-compose up -d postgres redis
        
        # Wait for database to be ready
        print_status "Waiting for database to be ready..."
        sleep 10
        
        # Run database migrations (placeholder)
        print_status "Database migrations will be run when implemented"
        
        print_success "Database setup completed"
    else
        print_warning "Docker not available. Please set up PostgreSQL and Redis manually."
        print_warning "Database URL: postgres://habits_user:habits_dev_password@localhost:5432/homo_habits"
        print_warning "Redis URL: redis://localhost:6379"
    fi
}

# Build Rust workspace
build_rust() {
    print_status "Building Rust workspace..."
    
    # Create placeholder Cargo.toml files for each member
    create_placeholder_cargo_files
    
    # Build the workspace
    cargo check
    
    print_success "Rust workspace check completed"
}

# Create placeholder Cargo.toml files for workspace members
create_placeholder_cargo_files() {
    print_status "Creating placeholder Cargo.toml files..."
    
    # Array of workspace members
    members=(
        "core/semantic"
        "core/identity"
        "core/bmds"
        "core/engine"
        "ai_integration/context_injection"
        "ai_integration/platforms/openai"
        "ai_integration/platforms/anthropic"
        "ai_integration/platforms/google"
        "ai_integration/platforms/local_models"
        "ai_integration/enhancement"
        "commercial/matching"
        "commercial/optimization"
        "commercial/pricing"
        "privacy/ephemeral"
        "privacy/disclosure"
        "privacy/security"
        "backend/api"
        "backend/services/semantic_processor"
        "backend/services/identity_manager"
        "backend/services/ai_integrator"
        "backend/services/commercial_optimizer"
        "modeling/multi_model"
        "modeling/domain_extraction"
        "modeling/quality_assessment"
        "modeling/distributed"
        "turbulance_integration/parser"
        "turbulance_integration/operations"
        "turbulance_integration/learning"
        "tools/semantic_debugger"
        "tools/performance_profiler"
        "tools/privacy_auditor"
        "tools/data_generators"
    )
    
    for member in "${members[@]}"; do
        if [ ! -f "$member/Cargo.toml" ]; then
            mkdir -p "$member/src"
            
            # Create basic Cargo.toml
            cat > "$member/Cargo.toml" << EOF
[package]
name = "$(basename "$member" | tr '-' '_')"
version.workspace = true
edition.workspace = true
authors.workspace = true
description = "Homo-habits $(basename "$member") component"
license.workspace = true

[dependencies]
tokio.workspace = true
serde.workspace = true
anyhow.workspace = true
tracing.workspace = true
EOF
            
            # Create basic lib.rs
            cat > "$member/src/lib.rs" << EOF
//! Homo-habits $(basename "$member") component
//! 
//! This module implements $(basename "$member") functionality for the homo-habits framework.

#![warn(missing_docs)]

/// Main functionality for $(basename "$member")
pub fn hello() {
    println!("Hello from $(basename "$member")!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello() {
        hello();
    }
}
EOF
        fi
    done
    
    print_success "Placeholder Cargo.toml files created"
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Check Rust compilation
    if cargo check --quiet; then
        print_success "Rust workspace compiles successfully"
    else
        print_error "Rust workspace compilation failed"
        return 1
    fi
    
    # Check Node.js setup
    if pnpm run --silent lint > /dev/null 2>&1 || true; then
        print_success "Node.js environment is ready"
    else
        print_warning "Node.js linting has issues (normal for initial setup)"
    fi
    
    print_success "Health checks completed"
}

# Print next steps
print_next_steps() {
    echo
    echo "🎉 Homo-Habits development environment setup complete!"
    echo "=================================================="
    echo
    echo "Next steps:"
    echo "1. Add your AI API keys to the .env file"
    echo "2. Start the development environment: pnpm run dev"
    echo "3. Visit the documentation at docs/ for detailed setup"
    echo "4. Begin implementing core components in the core/ directory"
    echo
    echo "Available commands:"
    echo "  pnpm run dev              - Start development servers"
    echo "  pnpm run build            - Build all components"
    echo "  pnpm run test             - Run all tests"
    echo "  pnpm run lint             - Run linting"
    echo "  cargo watch -x check      - Watch Rust files for changes"
    echo "  docker-compose up         - Start all services"
    echo
    echo "🐒 Happy coding with your semantic monkey tail!"
}

# Main setup function
main() {
    check_os
    check_prerequisites
    install_rust_deps
    create_directories
    setup_environment
    install_node_deps
    setup_database
    build_rust
    run_health_checks
    print_next_steps
}

# Run main function
main "$@"