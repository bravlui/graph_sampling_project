#!/bin/bash
# setup.sh — Configura ambiente virtual e git para o projeto
# Uso: bash setup.sh

set -e

echo "=== Projeto 6: Amostragem de Redes Complexas — Setup ==="

# 1. Criar e ativar venv
if [ ! -d "venv" ]; then
    echo "[1/4] Criando ambiente virtual..."
    python3 -m venv venv
else
    echo "[1/4] venv já existe."
fi

echo "[2/4] Ativando venv e instalando dependências..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Git
if [ ! -d ".git" ]; then
    echo "[3/4] Inicializando repositório git..."
    git init
    git checkout -b projeto6-amostragem
else
    echo "[3/4] Git já inicializado."
    # Criar branch se não existir
    if ! git rev-parse --verify projeto6-amostragem >/dev/null 2>&1; then
        git checkout -b projeto6-amostragem
    else
        echo "  Branch projeto6-amostragem já existe."
    fi
fi

# 4. .gitignore
if [ ! -f ".gitignore" ]; then
    echo "[4/4] Criando .gitignore..."
    cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
results/raw/
results/processed/
results/figures/
results/models/
*.pt
*.joblib
.DS_Store
EOF
else
    echo "[4/4] .gitignore já existe."
fi

echo ""
echo "=== Setup concluído! ==="
echo "Para ativar o ambiente: source venv/bin/activate"
echo "Para rodar o Nível 1:  python scripts/run_level1_benchmark.py --config configs/default.yaml"
