
import sys
import os

# Añadir la ruta absoluta de la carpeta externa al sys.path
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from crawler import EventScraper
def test_crawler():
    scraper = EventScraper()
    total = scraper.run_all_scrapers()
    print(f"[Crawler] Eventos descargados: {total}")

if __name__ == "__main__":
    test_crawler()
   
