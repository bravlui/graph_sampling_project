#!/bin/bash
# setup.sh — Configura ambiente virtual para o projeto
# Uso: bash setup.sh

set -e

echo "=== Amostragem de Redes Complexas — Setup ==="

# 1. Criar e ativar venv
if [ ! -d "venv" ]; then
    echo "[1/3] Criando ambiente virtual..."
    python3 -m venv venv
else
    echo "[1/3] venv já existe."
fi

echo "[2/3] Ativando venv e instalando dependências..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Verificar git
if [ ! -d ".git" ]; then
    echo "[3/3] Inicializando repositório git..."
    git init
else
    echo "[3/3] Git já inicializado."
fi

echo ""
echo "=== Setup concluído! ==="
echo "Para ativar o ambiente: source venv/bin/activate"
echo "Para rodar o benchmark: python scripts/run_benchmark.py --config configs/default.yaml"
echo "  (aviso: experimento completo leva 4-8 horas)"
