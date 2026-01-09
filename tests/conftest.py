import sys
from pathlib import Path

# Aggiunge la root del progetto al sys.path, così "import app" funziona
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
