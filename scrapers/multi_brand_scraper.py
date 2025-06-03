import logging
from scraper import ClassicComScraper, get_all_brands


def main():
    print("\U0001F699  Classic.com Multi-Brand Scraper")
    print("=" * 60)

    brands = get_all_brands()
    if not brands:
        print("No brands found. Check network connectivity or website structure.")
        return

    print(f"Found {len(brands)} brands. Starting scrape...")
    for brand in brands:
        print(f"\n-- Scraping {brand} --")
        scraper = ClassicComScraper(brand=brand)
        listings = scraper.scrape_all_listings_parallel(max_pages=100, delay_range=(3,6), max_workers=3)
        if listings:
            scraper.save_to_csv()
        else:
            print(f"No data extracted for {brand}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
