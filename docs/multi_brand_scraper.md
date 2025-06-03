# Multi-Brand Scraper

The `multi_brand_scraper.py` script automatically discovers every make listed on Classic.com and scrapes listings for each brand sequentially. It relies on the `ClassicComScraper` class which performs the actual extraction and CSV export.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the scraper:
   ```bash
   python scrapers/multi_brand_scraper.py
   ```
   Each brand will be processed in turn and the results saved to `<brand>_listings_enhanced_<timestamp>.csv`.

## Files
- `scrapers/multi_brand_scraper.py` – orchestrates scraping across all brands.
- `scrapers/scraper.py` – core scraping logic with robots.txt checks and data quality reporting.

Review Classic.com's terms and robots.txt before running this script.
