import os 
import sys
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from embedding import  run_embedding

if __name__ == "__main__":
    run_embedding()
