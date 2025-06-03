import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import random
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import logging
from datetime import datetime
import csv
import re
import concurrent.futures
import threading
from queue import Queue

def get_all_brands(base_url="https://www.classic.com"):
    """Fetch the list of all brands available on Classic.com."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    url = urljoin(base_url, "/makes/")
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        brand_links = soup.select('a[href^="/m/"]')
        brands = sorted({link.get_text(strip=True) for link in brand_links if link.get_text(strip=True)})
        return brands
    except Exception:
        return []

class ClassicComScraper:
    def __init__(self, brand="McLaren", base_url="https://www.classic.com"):
        self.base_url = base_url
        self.brand = brand
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.brand.lower()}_scraper.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.listings = []
        self.failed_pages = []
        
        # Threading controls
        self.rate_limiter = Queue()
        self.rate_limit_lock = threading.Lock()
        
        # Country code mapping
        self.country_codes = {
            'usa': 'US', 'us.png': 'US', 'united states': 'US',
            'canada': 'CA', 'ca.png': 'CA', 'can': 'CA',
            'uk.png': 'UK', 'united kingdom': 'UK', 'gb.png': 'UK',
            'germany': 'DE', 'de.png': 'DE',
            'france': 'FR', 'fr.png': 'FR',
            'italy': 'IT', 'it.png': 'IT',
            'spain': 'ES', 'es.png': 'ES',
            'netherlands': 'NL', 'nl.png': 'NL',
            'australia': 'AU', 'au.png': 'AU',
            'japan': 'JP', 'jp.png': 'JP',
        }
        
    def check_robots_txt(self):
        """Check robots.txt compliance - CRITICAL FIRST STEP"""
        try:
            rp = RobotFileParser()
            rp.set_url(f"{self.base_url}/robots.txt")
            rp.read()
            can_fetch = rp.can_fetch('*', f"{self.base_url}/search")
            
            self.logger.info(f"Robots.txt check: Can fetch search page: {can_fetch}")
            
            # Get crawl delay if specified
            crawl_delay = rp.crawl_delay('*')
            if crawl_delay:
                self.logger.info(f"Robots.txt specifies crawl delay: {crawl_delay} seconds")
                return can_fetch, crawl_delay
            
            return can_fetch, 2  # Default 2 second delay
            
        except Exception as e:
            self.logger.error(f"Could not check robots.txt: {e}")
            return False, 2
    
    def rate_limited_request(self, url, params=None, delay=2):
        """Make rate-limited request"""
        with self.rate_limit_lock:
            time.sleep(delay)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
    
    def get_search_page(self, query=None, page=1, delay=2):
        """Fetch a search page with proper error handling"""
        url = f"{self.base_url}/search"
        params = {
            'q': query or self.brand,
            'page': page
        }
        
        response = self.rate_limited_request(url, params, delay)
        if response:
            self.logger.info(f"Successfully fetched page {page}")
        else:
            self.failed_pages.append(page)
        
        return response
    
    def extract_listings_from_html(self, html_content, page_num):
        """Extract listing data from HTML content using actual classic.com structure"""
        soup = BeautifulSoup(html_content, 'html.parser')
        page_listings = []
        
        # Find listings by looking for divs with numeric IDs (the actual pattern used by classic.com)
        all_divs = soup.find_all('div', id=re.compile(r'^\d+$'))
        
        self.logger.info(f"Found {len(all_divs)} potential listing containers on page {page_num}")
        
        for idx, element in enumerate(all_divs):
            try:
                # Each listing should have a nested structure with vehicle information
                listing_data = self.extract_single_listing(element, page_num, idx)
                if listing_data and listing_data.get('title'):  # Only add if we got meaningful data
                    page_listings.append(listing_data)
                    
            except Exception as e:
                self.logger.warning(f"Error extracting listing {idx} on page {page_num}: {e}")
                continue
        
        return page_listings
    
    def extract_single_listing(self, element, page_num, listing_idx):
        """Extract data from a single listing element based on classic.com structure"""
        listing = {
            'page_number': page_num,
            'listing_index': listing_idx,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Title extraction - look for h3 > a structure
        title_element = element.select_one('h3 a')
        if title_element:
            listing['title'] = title_element.get_text(strip=True)
            # Extract brand, model, year from title
            title_parts = listing['title'].split()
            if len(title_parts) >= 3:
                try:
                    listing['year'] = int(title_parts[0])
                    listing['brand'] = title_parts[1]
                    listing['model'] = ' '.join(title_parts[2:])
                except:
                    listing['year'] = None
                    listing['brand'] = self.brand
                    listing['model'] = listing['title']
        
        # Listing URL - from the title link or other links in the container
        url_element = element.select_one('a[href*="/veh/"]')
        if url_element and url_element.get('href'):
            listing['listing_url'] = urljoin(self.base_url, url_element['href'])
        
        # Price extraction - look for price display elements
        price_element = element.select_one('div[id*="price"]') or element.find(string=re.compile(r'\$[\d,]+'))
        if price_element:
            if hasattr(price_element, 'get_text'):
                price_text = price_element.get_text(strip=True)
            else:
                price_text = str(price_element).strip()
            listing['price'] = price_text
        
        # Enhanced status extraction - look for "For Sale", "Sold", "Auction", etc.
        element_text = element.get_text().lower()
        
        # Check for sold status
        if re.search(r'\bsold\b', element_text):
            listing['sold'] = 'y'
            listing['active_listing'] = 'n'
        elif re.search(r'\bfor sale\b', element_text):
            listing['sold'] = 'n'
            listing['active_listing'] = 'y'
        elif re.search(r'\bauction\b', element_text):
            listing['sold'] = 'n'
            listing['active_listing'] = 'y'
        else:
            # Check for status badges or indicators
            status_element = element.find('div', class_=re.compile(r'(status|badge)', re.I))
            if status_element:
                status_text = status_element.get_text().lower()
                listing['sold'] = 'y' if 'sold' in status_text else 'n'
                listing['active_listing'] = 'n' if 'sold' in status_text else 'y'
            else:
                listing['sold'] = 'n'  # Default assumption
                listing['active_listing'] = 'y'  # Default assumption
        
        # Seller information extraction
        seller_element = element.select_one('a[href*="/s/"]')  # Seller links typically have /s/ pattern
        if seller_element:
            listing['seller_name'] = seller_element.get_text(strip=True)
            listing['seller_url'] = urljoin(self.base_url, seller_element.get('href', ''))
        
        # Alternative seller extraction - look for dealer/seller names
        if not listing.get('seller_name'):
            # Look for patterns like "Earth MotorCars", "August Luxury Motorcars"
            seller_text = element.find(string=re.compile(r'(Motor|Car|Auto|Classic|Luxury|Exotic)', re.I))
            if seller_text:
                # Get the parent element to find the complete seller name
                seller_parent = seller_text.parent if hasattr(seller_text, 'parent') else None
                if seller_parent:
                    seller_name = seller_parent.get_text(strip=True)
                    # Clean up seller name
                    seller_name = re.sub(r'\s+', ' ', seller_name)
                    if len(seller_name) < 50:  # Reasonable seller name length
                        listing['seller_name'] = seller_name
        
        # Seller verification status
        verified_element = element.select_one('i[class*="verified"]') or element.find(string=re.compile(r'verified', re.I))
        listing['seller_verified'] = 'y' if verified_element else 'n'
        
        # Mileage extraction
        mileage_element = element.find(string=re.compile(r'\d+[k]?\s*mi', re.I))
        if mileage_element:
            listing['mileage'] = mileage_element.strip()
        
        # Transmission extraction
        transmission_element = element.find(string=re.compile(r'(manual|automatic|dct)', re.I))
        if transmission_element:
            listing['transmission'] = transmission_element.strip()
        
        # Location and country extraction - improved logic
        location_img = element.select_one('img[src*="/flags/"]')
        if location_img:
            flag_src = location_img.get('src', '')
            country_from_flag = self.extract_country_code_from_flag(flag_src)
            listing['country_code'] = country_from_flag
            
            # Look for location text near the flag - try multiple approaches
            location_text = None
            
            # Method 1: Look for div containing the flag and location text
            location_div = location_img.find_parent('div')
            if location_div:
                location_text = location_div.get_text(strip=True)
                # Remove flag alt text and clean up
                location_text = re.sub(r'^[A-Z]{2,3}\s*', '', location_text)
            
            # Method 2: Look for text immediately following the flag
            if not location_text:
                next_sibling = location_img.find_next_sibling(string=True)
                if next_sibling:
                    location_text = next_sibling.strip()
            
            # Method 3: Look in the parent container for location patterns
            if not location_text:
                parent_text = location_img.parent.get_text()
                location_match = re.search(r'([A-Za-z\s]+,\s*[A-Za-z\s]+)', parent_text)
                if location_match:
                    location_text = location_match.group(1)
            
            if location_text:
                listing['location'] = location_text
                
                # Extract state if it's US/Canada
                if country_from_flag in ['US', 'CA']:
                    # Look for state patterns like "City, State" or "City, ST"
                    state_match = re.search(r',\s*([A-Z]{2,}|[A-Za-z\s]+)(?:\s*,|$)', location_text)
                    if state_match:
                        state = state_match.group(1).strip()
                        # Convert full state names to abbreviations for US states
                        if len(state) > 2 and country_from_flag == 'US':
                            state_abbrevs = {
                                'california': 'CA', 'texas': 'TX', 'florida': 'FL', 'new york': 'NY',
                                'illinois': 'IL', 'pennsylvania': 'PA', 'ohio': 'OH', 'georgia': 'GA',
                                'north carolina': 'NC', 'michigan': 'MI', 'new jersey': 'NJ', 'virginia': 'VA',
                                'washington': 'WA', 'arizona': 'AZ', 'massachusetts': 'MA', 'tennessee': 'TN',
                                'indiana': 'IN', 'missouri': 'MO', 'maryland': 'MD', 'wisconsin': 'WI',
                                'colorado': 'CO', 'minnesota': 'MN', 'south carolina': 'SC', 'alabama': 'AL',
                                'louisiana': 'LA', 'kentucky': 'KY', 'oregon': 'OR', 'oklahoma': 'OK',
                                'connecticut': 'CT', 'utah': 'UT', 'iowa': 'IA', 'nevada': 'NV',
                                'arkansas': 'AR', 'mississippi': 'MS', 'kansas': 'KS', 'new mexico': 'NM',
                                'nebraska': 'NE', 'west virginia': 'WV', 'idaho': 'ID', 'hawaii': 'HI',
                                'new hampshire': 'NH', 'maine': 'ME', 'montana': 'MT', 'rhode island': 'RI',
                                'delaware': 'DE', 'south dakota': 'SD', 'north dakota': 'ND', 'alaska': 'AK',
                                'vermont': 'VT', 'wyoming': 'WY'
                            }
                            state_lower = state.lower()
                            if state_lower in state_abbrevs:
                                state = state_abbrevs[state_lower]
                        listing['state'] = state
        
        # Engine extraction - improved logic
        engine_text = element.get_text()
        engine_patterns = [
            r'(\d+\.?\d*L\s*V\d+(?:\s*Twin\s*Turbo)?)',  # 3.8L V8 Twin Turbo
            r'(\d+\.?\d*L\s*(?:Twin\s*)?Turbo)',         # 3.8L Twin Turbo
            r'(V\d+\s*(?:Twin\s*)?Turbo)',               # V8 Twin Turbo
            r'(\d+\.?\d*L)',                             # 3.8L
            r'(V\d+)',                                   # V8
        ]
        
        for pattern in engine_patterns:
            engine_match = re.search(pattern, engine_text, re.I)
            if engine_match:
                listing['engine'] = engine_match.group(1)
                break
        
        # Image URL
        img_element = element.select_one('img[src*="images.classic.com"]')
        if img_element and img_element.get('src'):
            listing['image_url'] = img_element['src']
        
        return listing
    
    def extract_country_code_from_flag(self, flag_src):
        """Extract country code from flag image source"""
        for key, code in self.country_codes.items():
            if key in flag_src.lower():
                return code
        return None
    
    def detect_pagination_info(self, soup):
        """Detect pagination information from the page"""
        pagination_info = {
            'total_pages': None,
            'current_page': None,
            'total_results': None,
            'has_next': False
        }
        
        # Look for pagination text like "1 / 100"
        pagination_text = soup.find(string=re.compile(r'\d+\s*/\s*\d+'))
        if pagination_text:
            match = re.search(r'(\d+)\s*/\s*(\d+)', pagination_text)
            if match:
                pagination_info['current_page'] = int(match.group(1))
                pagination_info['total_pages'] = int(match.group(2))
        
        # Check for next page link
        next_link = soup.select_one('a[href*="page="]')
        pagination_info['has_next'] = next_link is not None
        
        return pagination_info
    
    def scrape_page_batch(self, page_numbers, delay=2):
        """Scrape a batch of pages"""
        batch_listings = []
        
        for page in page_numbers:
            try:
                response = self.get_search_page(page=page, delay=delay)
                if response:
                    page_listings = self.extract_listings_from_html(response.text, page)
                    batch_listings.extend(page_listings)
                    self.logger.info(f"Extracted {len(page_listings)} listings from page {page}")
                else:
                    self.logger.warning(f"Failed to fetch page {page}")
            except Exception as e:
                self.logger.error(f"Error processing page {page}: {e}")
        
        return batch_listings
    
    def scrape_all_listings_parallel(self, max_pages=100, delay_range=(2, 5), max_workers=3):
        """Main scraping function with safe parallelization"""
        
        # CRITICAL: Check robots.txt first
        can_fetch, min_delay = self.check_robots_txt()
        if not can_fetch:
            self.logger.error("Robots.txt prohibits scraping. Please respect the website's rules.")
            print("\n⚠️  IMPORTANT: Robots.txt check failed or prohibits scraping.")
            print("Please verify compliance with the website's terms before proceeding.")
            return None
        
        print(f"\n✅ Robots.txt compliance check passed. Minimum delay: {min_delay}s")
        print(f"🚀 Starting parallel {self.brand} data extraction...")
        
        delay_min, delay_max = delay_range
        if delay_min < min_delay:
            delay_min = min_delay
            delay_max = max(delay_max, min_delay + 1)
        
        # First, get pagination info from page 1
        response = self.get_search_page(page=1, delay=delay_min)
        if not response:
            self.logger.error("Failed to fetch first page")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        pagination_info = self.detect_pagination_info(soup)
        
        if pagination_info['total_pages']:
            max_pages = min(max_pages, pagination_info['total_pages'])
            self.logger.info(f"Detected {pagination_info['total_pages']} total pages, will scrape {max_pages}")
        
        # Extract from first page
        first_page_listings = self.extract_listings_from_html(response.text, 1)
        self.listings.extend(first_page_listings)
        print(f"📄 Page 1: {len(first_page_listings)} listings")
        
        # Create batches for parallel processing
        remaining_pages = list(range(2, max_pages + 1))
        batch_size = max(1, len(remaining_pages) // max_workers)
        page_batches = [remaining_pages[i:i + batch_size] for i in range(0, len(remaining_pages), batch_size)]
        
        print(f"🔄 Processing {len(remaining_pages)} pages in {len(page_batches)} batches using {max_workers} workers")
        
        # Process batches in parallel with rate limiting
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            
            for i, batch in enumerate(page_batches):
                # Stagger the start times to respect rate limits
                delay = random.uniform(delay_min, delay_max) * (i + 1)
                future = executor.submit(self.scrape_page_batch, batch, delay)
                future_to_batch[future] = batch
            
            # Collect results as they complete
            total_listings = len(first_page_listings)
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_listings = future.result()
                    self.listings.extend(batch_listings)
                    total_listings += len(batch_listings)
                    print(f"📦 Batch {batch[0]}-{batch[-1]}: {len(batch_listings)} listings (Total: {total_listings})")
                except Exception as e:
                    self.logger.error(f"Batch {batch} failed: {e}")
        
        print(f"\n✅ Parallel scraping completed!")
        print(f"📊 Total listings extracted: {len(self.listings)}")
        print(f"📄 Pages processed: {max_pages}")
        print(f"❌ Failed pages: {len(self.failed_pages)}")
        
        return self.listings
    
    def run_data_quality_checks(self):
        """Comprehensive data quality assurance"""
        if not self.listings:
            print("❌ No data to check")
            return
        
        print("\n" + "="*60)
        print("🔍 DATA QUALITY ASSURANCE REPORT")
        print("="*60)
        
        df = pd.DataFrame(self.listings)
        
        # Basic stats
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total records: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Required fields coverage
        print(f"\n✅ REQUIRED FIELDS COVERAGE:")
        required_fields = [
            'brand', 'model', 'year', 'price', 'mileage', 'location', 
            'country_code', 'state', 'listing_url', 'engine', 'transmission', 
            'sold', 'active_listing', 'seller_name', 'seller_verified'
        ]
        
        coverage_issues = []
        for field in required_fields:
            if field in df.columns:
                non_null = df[field].notna().sum()
                percentage = (non_null / len(df)) * 100
                status = "✅" if percentage > 80 else "⚠️" if percentage > 50 else "❌"
                print(f"   {status} {field:15}: {non_null:4d}/{len(df)} ({percentage:5.1f}%)")
                if percentage < 80:
                    coverage_issues.append(f"{field}: {percentage:.1f}%")
            else:
                print(f"   ❌ {field:15}: MISSING COLUMN")
                coverage_issues.append(f"{field}: MISSING")
        
        # Data consistency checks
        print(f"\n🔍 DATA CONSISTENCY CHECKS:")
        
        # Year range check
        if 'year' in df.columns:
            year_issues = df[(df['year'] < 1970) | (df['year'] > 2025)]['year'].count()
            print(f"   Years outside 1970-2025: {year_issues} ({year_issues/len(df)*100:.1f}%)")
        
        # Price format check
        if 'price' in df.columns:
            price_with_dollar = df['price'].str.contains(r'\$', na=False).sum()
            print(f"   Prices with $ symbol: {price_with_dollar}/{df['price'].notna().sum()} ({price_with_dollar/df['price'].notna().sum()*100:.1f}%)")
        
        # URL validity check
        if 'listing_url' in df.columns:
            valid_urls = df['listing_url'].str.startswith('http', na=False).sum()
            print(f"   Valid URLs: {valid_urls}/{df['listing_url'].notna().sum()} ({valid_urls/df['listing_url'].notna().sum()*100:.1f}%)")
        
        # Brand distribution
        print(f"\n🏷️  BRAND DISTRIBUTION:")
        if 'brand' in df.columns:
            brand_counts = df['brand'].value_counts().head(10)
            for brand, count in brand_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {brand}: {count:,} ({percentage:.1f}%)")
        
        # Geographic distribution
        print(f"\n🌍 GEOGRAPHIC DISTRIBUTION:")
        if 'country_code' in df.columns:
            country_counts = df['country_code'].value_counts()
            for country, count in country_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {country}: {count:,} ({percentage:.1f}%)")
        
        # Listing status
        print(f"\n📋 LISTING STATUS:")
        if 'sold' in df.columns:
            sold_counts = df['sold'].value_counts()
            print(f"   Sold: {sold_counts.get('y', 0):,}")
            print(f"   Available: {sold_counts.get('n', 0):,}")
        
        if 'active_listing' in df.columns:
            active_counts = df['active_listing'].value_counts()
            print(f"   Active: {active_counts.get('y', 0):,}")
            print(f"   Inactive: {active_counts.get('n', 0):,}")
        
        # Seller information
        print(f"\n👥 SELLER INFORMATION:")
        if 'seller_name' in df.columns:
            sellers_with_name = df['seller_name'].notna().sum()
            print(f"   Records with seller name: {sellers_with_name:,} ({sellers_with_name/len(df)*100:.1f}%)")
            
            if sellers_with_name > 0:
                top_sellers = df['seller_name'].value_counts().head(5)
                print(f"   Top sellers:")
                for seller, count in top_sellers.items():
                    print(f"     {seller}: {count:,} listings")
        
        if 'seller_verified' in df.columns:
            verified_count = df[df['seller_verified'] == 'y'].shape[0]
            print(f"   Verified sellers: {verified_count:,} ({verified_count/len(df)*100:.1f}%)")
        
        # Data quality score
        total_checks = len(required_fields)
        passed_checks = sum(1 for field in required_fields if field in df.columns and (df[field].notna().sum() / len(df)) * 100 > 80)
        quality_score = (passed_checks / total_checks) * 100
        
        print(f"\n🎯 OVERALL DATA QUALITY SCORE: {quality_score:.1f}%")
        
        if coverage_issues:
            print(f"\n⚠️  COVERAGE ISSUES TO ADDRESS:")
            for issue in coverage_issues:
                print(f"   - {issue}")
        
        print("\n" + "="*60)
    
    def save_to_csv(self, filename=None):
        """Save extracted listings to CSV file with all required fields"""
        if not self.listings:
            self.logger.error("No listings to save")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.brand.lower()}_listings_enhanced_{timestamp}.csv"
        
        # Define the required columns in a specific order
        required_columns = [
            'brand', 'model', 'year', 'price', 'mileage', 'location', 
            'country_code', 'state', 'listing_url', 'engine', 'transmission', 
            'sold', 'active_listing', 'seller_name', 'seller_url', 'seller_verified',
            'title', 'image_url', 'page_number', 'listing_index', 'scraped_at'
        ]
        
        # Ensure all listings have all required fields
        for listing in self.listings:
            for col in required_columns:
                if col not in listing:
                    listing[col] = None
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=required_columns)
                writer.writeheader()
                writer.writerows(self.listings)
            
            self.logger.info(f"Data saved to {filename}")
            print(f"💾 Data saved to: {filename}")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            return None
    
    def generate_synthetic_data(self, count=2380):
        """Generate synthetic data for testing purposes"""
        print(f"\n🧪 Generating synthetic {self.brand} data for testing...")
        
        sample_models = [
            'F1', 'F1 LM', 'F1 GT', 'P1', 'P1 GTR', '720S', '720S Spider',
            '750S', '750S Spider', 'Artura', '600LT', '600LT Spider',
            '570S', '570GT', '650S', '650S Spider', 'MP4-12C', 'Senna',
            'Senna GTR', 'Speedtail', 'Elva', 'GT'
        ]
        
        locations = [
            'Beverly Hills, CA', 'Miami, FL', 'New York, NY', 'Dallas, TX',
            'London, UK', 'Monaco', 'Dubai, UAE', 'Tokyo, Japan',
            'Los Angeles, CA', 'Chicago, IL', 'Atlanta, GA', 'Phoenix, AZ'
        ]
        
        sellers = [
            'Prestige Motors', 'Elite Auto Gallery', 'Classic Car Depot',
            'Luxury Motors Inc', 'Heritage Auto', 'Premier Classics'
        ]
        
        for i in range(count):
            model = random.choice(sample_models)
            year = random.randint(1994, 2024)
            
            # Price based on model and year
            base_prices = {
                'F1': 15000000, 'F1 LM': 20000000, 'P1': 1200000, 'Senna': 1000000,
                '720S': 300000, '750S': 350000, 'Artura': 250000, '600LT': 250000,
                '570S': 180000, '650S': 200000, 'MP4-12C': 150000, 'Speedtail': 2200000
            }
            
            base_price = base_prices.get(model.split()[0], 200000)
            age_factor = max(0.3, 1 - (2024 - year) * 0.05)
            price = int(base_price * age_factor * random.uniform(0.8, 1.3))
            
            listing = {
                'brand': self.brand,
                'model': model,
                'year': year,
                'title': f"{year} {self.brand} {model}",
                'price': f"${price:,}",
                'mileage': f"{random.randint(100, 50000):,} mi",
                'location': random.choice(locations),
                'country_code': random.choice(['US', 'UK', 'CA', 'DE', 'JP']),
                'state': random.choice(['CA', 'FL', 'NY', 'TX', None, None]),
                'listing_url': f"https://www.classic.com/veh/{year}-{self.brand.lower()}-{model.lower().replace(' ', '-')}-{random.randint(100000, 999999)}",
                'engine': f"{random.choice(['3.8L V8 Twin Turbo', '4.0L V8 Twin Turbo', '6.1L V12'])}",
                'transmission': random.choice(['7-Speed DCT', '6-Speed Manual', '8-Speed DCT']),
                'sold': random.choice(['y', 'n']),
                'active_listing': random.choice(['y', 'n']),
                'seller_name': random.choice(sellers),
                'seller_url': f"https://www.classic.com/s/{random.choice(sellers).lower().replace(' ', '-')}-{random.randint(1000, 9999)}/",
                'seller_verified': random.choice(['y', 'n']),
                'image_url': f"https://images.classic.com/vehicles/{self.brand.lower()}-{random.randint(1, 100)}.jpg",
                'page_number': (i // 20) + 1,
                'listing_index': i % 20,
                'scraped_at': datetime.now().isoformat()
            }
            
            self.listings.append(listing)
        
        print(f"✅ Generated {count} synthetic {self.brand} listings")
        return self.listings

def main():
    """Main execution function"""
    print("🏎️  Enhanced Classic.com Data Extractor")
    print("=" * 60)
    
    # Important disclaimer
    print("\n⚠️  IMPORTANT LEGAL NOTICE:")
    print("This script is for educational purposes. Before scraping:")
    print("1. Check robots.txt compliance")
    print("2. Review the website's Terms of Service")
    print("3. Ensure you have permission for data collection")
    print("4. Use responsibly with appropriate delays")
    
    print("\n🔍 Starting enhanced parallel data extraction...")

    brand = input("Enter brand to scrape (default McLaren): ").strip() or "McLaren"
    scraper = ClassicComScraper(brand=brand)

    # Real scraping with parallelization
    listings = scraper.scrape_all_listings_parallel(
        max_pages=100,
        delay_range=(3, 6),
        max_workers=3
    )
    
    # Run data quality checks
    if listings:
        scraper.run_data_quality_checks()
        
        # Save to CSV
        filename = scraper.save_to_csv()
        if filename:
            print(f"\n🎉 Success! Enhanced {brand} data saved to: {filename}")
        else:
            print("\n❌ Error saving data to CSV")
    else:
        print("\n❌ No data extracted")

if __name__ == "__main__":
    main()