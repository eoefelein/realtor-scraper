"""
Azure Function for scraping and storing Realtor.com rental listings.

This function runs daily to:
1. Fetch rental listings from Realtor.com API for Austin, TX
2. Store active listings in Azure SQL Database
3. Retrieve detailed property information for new listings
4. Update property database with comprehensive property details

Environment Variables Required:
    REALTOR_SQL_DB_CONNECTION_STRING: Azure SQL connection string
    RAPIDAPI_KEY: RapidAPI key for Realtor.com API access
"""

import logging
import json
import time
import traceback
from datetime import date, datetime
from typing import Dict, List, Optional

import azure.functions as func
import numpy as np
import pandas as pd
import pyodbc
import requests
import urllib
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://realtor-data3.p.rapidapi.com"
API_HOST = "realtor-data3.p.rapidapi.com"
SEARCH_LOCATION = "Austin, TX"
PAGE_SIZE = 200
REQUEST_DELAY = 1  # seconds between API requests
MAX_DB_RETRIES = 2
RETRY_DELAY = 60  # seconds

# Property detail categories to extract
DETAIL_CATEGORIES = [
    "Appliances",
    "Heating and Cooling",
    "Interior Features",
    "Exterior and Lot Features",
    "Garage and Parking",
    "Rental Info",
]

app = func.FunctionApp()


def get_api_headers() -> Dict[str, str]:
    """Get API headers with credentials from environment."""
    import os
    
    api_key = os.environ.get('RAPIDAPI_KEY')
    if not api_key:
        raise ValueError("RAPIDAPI_KEY environment variable not set")
    
    return {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": API_HOST,
    }


def create_db_connection(connection_string: str, retry: bool = True) -> Optional[pyodbc.Connection]:
    """
    Create database connection with retry logic.
    
    Args:
        connection_string: Database connection string
        retry: Whether to retry on failure
    
    Returns:
        Database connection object or None if failed
    """
    try:
        conn = pyodbc.connect(connection_string)
        logger.info("✅ Database connection succeeded")
        return conn
    except Exception as e:
        if retry:
            logger.warning(f"⚠️ First connection attempt failed: {e}")
            logger.info(f"Waiting {RETRY_DELAY} seconds and retrying...")
            time.sleep(RETRY_DELAY)
            try:
                conn = pyodbc.connect(connection_string)
                logger.info("✅ Database connection succeeded on retry")
                return conn
            except Exception as retry_error:
                logger.exception("❌ Database connection failed after retry")
                return None
        else:
            logger.exception("❌ Database connection failed")
            return None


def create_sqlalchemy_engine(connection_string: str, max_retries: int = MAX_DB_RETRIES) -> Optional[create_engine]:
    """
    Create SQLAlchemy engine with retry logic.
    
    Args:
        connection_string: Database connection string
        max_retries: Maximum number of retry attempts
    
    Returns:
        SQLAlchemy engine or None if failed
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Creating SQLAlchemy engine (attempt {attempt}/{max_retries})...")
            params = urllib.parse.quote_plus(connection_string)
            engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
            
            # Test connection
            with engine.connect() as connection:
                logger.info("✅ SQLAlchemy engine created and tested")
            return engine
            
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"⚠️ Engine creation attempt {attempt} failed: {e}")
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.exception("❌ SQLAlchemy engine creation failed after all attempts")
                return None
    
    return None


def recursive_flatten(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """
    Recursively flatten nested dictionary structure.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(recursive_flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists of primitives directly, lists of dicts to JSON
            if all(isinstance(i, (str, int, float, bool, type(None))) for i in v):
                items.append((new_key, v))
            else:
                items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def fetch_listings_page(headers: Dict, page_num: int) -> Optional[Dict]:
    """
    Fetch a single page of listings from API.
    
    Args:
        headers: API request headers
        page_num: Page number to fetch
    
    Returns:
        JSON response or None if failed
    """
    url = f"{API_BASE_URL}/SearchRent"
    querystring = {
        "location": f"city:{SEARCH_LOCATION}",
        "pageSize": str(PAGE_SIZE),
        "page": str(page_num),
        "sort": "lowest_price",
        "propertyType": "townhome, single_family_home, multi_family",
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching page {page_num}: {e}")
        return None


def fetch_all_listings(headers: Dict) -> pd.DataFrame:
    """
    Fetch all available rental listings across all pages.
    
    Args:
        headers: API request headers
    
    Returns:
        DataFrame containing all listings
    """
    # Fetch first page to get total page count
    first_page = fetch_listings_page(headers, 1)
    if not first_page:
        logger.error("Failed to fetch first page")
        return pd.DataFrame()
    
    total_pages = first_page.get("totalPages", 1)
    logger.info(f"Total pages to fetch: {total_pages}")
    
    all_results = []
    
    for page_num in range(1, total_pages + 1):
        logger.info(f"Fetching page {page_num}/{total_pages}")
        
        response_json = fetch_listings_page(headers, page_num)
        if not response_json:
            logger.error(f"Failed to fetch page {page_num}, stopping pagination")
            break
        
        try:
            property_data = response_json.get("data", [])
            flattened_properties = [recursive_flatten(p) for p in property_data]
            property_df = pd.DataFrame(flattened_properties)
            all_results.append(property_df)
        except Exception as e:
            logger.exception(f"Error processing page {page_num}")
            continue
        
        time.sleep(REQUEST_DELAY)
    
    if not all_results:
        logger.error("No listings retrieved")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_results, ignore_index=True).drop_duplicates(subset=['property_id'])
    logger.info(f"Total unique properties fetched: {len(combined_df)}")
    
    return combined_df


def prepare_listings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare listings data for database insertion.
    
    Args:
        df: Raw listings DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Rename columns
    df_clean.rename(columns={
        "location.address.line": "location_address_line",
        "location.address.postal_code": "location_address_zipcode",
    }, inplace=True)
    
    # Fill missing list_date with today's date
    today_str = datetime.today().date().isoformat()
    df_clean["list_date"] = df_clean["list_date"].fillna(today_str)
    df_clean["last_observed"] = date.today().isoformat()
    
    logger.info(f"Prepared {len(df_clean)} listings for database insertion")
    return df_clean


def insert_active_listings(cursor: pyodbc.Cursor, df: pd.DataFrame) -> None:
    """
    Insert or update active listings in database using MERGE statement.
    
    Args:
        cursor: Database cursor
        df: DataFrame containing listings to insert
    """
    logger.info(f"Inserting {len(df)} active listings into database...")
    
    merge_query = """
        MERGE INTO active_listings AS target
        USING (SELECT ? AS property_id, ? AS listing_id, ? AS location_address_line, 
                      ? AS location_address_zipcode, ? AS list_date, ? AS last_observed) AS source
        ON target.property_id = source.property_id
        WHEN MATCHED THEN 
            UPDATE SET last_observed = source.last_observed
        WHEN NOT MATCHED THEN 
            INSERT (property_id, listing_id, location_address_line, location_address_zipcode, 
                    list_date, last_observed)
            VALUES (source.property_id, source.listing_id, source.location_address_line, 
                    source.location_address_zipcode, source.list_date, source.last_observed);
    """
    
    for _, row in df.iterrows():
        cursor.execute(
            merge_query,
            row['property_id'], row['listing_id'], row['location_address_line'],
            row['location_address_zipcode'], row['list_date'], row['last_observed']
        )


def fetch_property_details(headers: Dict, property_id: str, listing_id: str) -> Optional[Dict]:
    """
    Fetch detailed information for a specific property.
    
    Args:
        headers: API request headers
        property_id: Property ID
        listing_id: Listing ID
    
    Returns:
        Property details JSON or None if failed
    """
    url = f"{API_BASE_URL}/detail"
    querystring = {"propertyId": property_id, "listingId": listing_id}
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching details for property {property_id}: {e}")
        return None


def extract_property_details_text(details: List[Dict]) -> str:
    """
    Extract and concatenate property detail text from specific categories.
    
    Args:
        details: List of detail dictionaries
    
    Returns:
        Comma-separated string of detail text
    """
    all_text = []
    if details:
        for detail in details:
            if detail.get("category") in DETAIL_CATEGORIES:
                all_text.extend(detail.get("text", []))
    return ",".join(all_text)


def normalize_property_data(property_json: Dict) -> pd.DataFrame:
    """
    Normalize property JSON into DataFrame with standardized columns.
    
    Args:
        property_json: Raw property data JSON
    
    Returns:
        Normalized DataFrame with single row
    """
    df = pd.json_normalize(property_json.get("data", {}))
    cols = list(df.columns)
    
    # Extract and process details
    if "details" in cols and df.loc[0, "details"] is not None:
        df['details'] = extract_property_details_text(df.loc[0, "details"])
    
    # Handle missing mortgage fields
    if "monthly_fees" not in cols:
        df["monthly_fees"] = np.nan
    
    if "mortgage" in cols:
        df["mortgage.estimate.average_rate.rate"] = np.nan
        df["mortgage.property_tax_rate"] = np.nan
        df = df.rename(columns={"mortgage": "mortgage.estimate.monthly_payment"})
    
    # Rename school columns
    if "nearby_schools.schools" in cols:
        df = df.rename(columns={"nearby_schools.schools": "nearby_schools"})
    if "schools.schools" in cols:
        df = df.rename(columns={"schools.schools": "schools"})
    
    # Handle flood data
    if "local.flood" in cols:
        df = df.rename(columns={"local.flood": "local.flood.flood_factor_score"})
    
    for flood_col in ["local.flood.flood_trend_paragraph", "local.flood.fema_zone", 
                      "local.flood.flood_insurance_text"]:
        if flood_col not in cols:
            df[flood_col] = np.nan
    
    # Handle coordinates
    if "location.address.coordinate" in cols:
        coord = df["location.address.coordinate"].values[0]
        if coord is not None:
            df["location.address.coordinate.lat"] = coord[0]
            df["location.address.coordinate.lon"] = coord[1]
        else:
            df["location.address.coordinate.lat"] = np.nan
            df["location.address.coordinate.lon"] = np.nan
    
    # Process pet policy
    if "pet_policy.cats" in cols or "pet_policy.dogs" in cols:
        df["pet_policy"] = df.apply(
            lambda row: list(set([row.get("pet_policy.cats"), row.get("pet_policy.dogs")])),
            axis=1
        )
    
    # Select final columns (extensive property schema)
    selected_columns = [
        "property_id", "listing_id", "status", "href", "list_date",
        "last_price_change_date", "last_price_change_amount", "list_price",
        "mortgage.estimate.average_rate.rate", "mortgage.estimate.monthly_payment",
        "mortgage.property_tax_rate", "buyers", "tax_history", "monthly_fees",
        "one_time_fees", "units", "community_rental_floorplans",
        "description.beds", "description.beds_min", "description.beds_max",
        "description.baths", "description.baths_min", "description.baths_max",
        "description.baths_consolidated", "description.garage",
        "description.garage_min", "description.garage_max", "description.sqft",
        "description.sqft_min", "description.sqft_max", "description.lot_sqft",
        "description.units", "description.stories", "description.type",
        "description.styles", "description.heating", "description.cooling",
        "description.pool", "description.year_built", "description.name",
        "description.text", "details", "photo_count", "photos", "virtual_tours",
        "property_history", "last_sold_price", "last_sold_date",
        "location.address.line", "location.address.city", "location.address.state_code",
        "location.address.state", "location.address.postal_code",
        "location.address.coordinate.lat", "location.address.coordinate.lon",
        "location.county.fips_code", "location.street_view_url",
        "location.neighborhoods", "nearby_schools", "schools", "pet_policy",
        "local.noise.score", "local.noise.noise_categories",
        "local.flood.flood_factor_score", "local.flood.flood_trend_paragraph",
        "local.flood.fema_zone", "local.flood.flood_insurance_text",
        "estimates.current_values", "estimates.historical_values",
        "estimates.forecast_values", "flags.is_new_construction",
        "flags.is_pending", "flags.is_foreclosure", "flags.is_price_reduced",
        "flags.is_new_listing"
    ]
    
    # Only select columns that exist
    available_columns = [col for col in selected_columns if col in df.columns]
    return df[available_columns]


def process_new_properties(headers: Dict, new_properties_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch detailed data for new properties and compile into DataFrame.
    
    Args:
        headers: API request headers
        new_properties_df: DataFrame of new properties to process
    
    Returns:
        DataFrame with detailed property information
    """
    all_property_details = []
    
    for idx, row in new_properties_df.iterrows():
        property_id = row.get("property_id")
        listing_id = row.get("listing_id")
        
        if pd.isna(property_id) or pd.isna(listing_id):
            logger.warning(f"Skipping row {idx}: missing property_id or listing_id")
            continue
        
        logger.info(f"Fetching details for property {idx + 1}/{len(new_properties_df)}")
        
        property_json = fetch_property_details(headers, property_id, listing_id)
        if not property_json:
            continue
        
        try:
            normalized_df = normalize_property_data(property_json)
            all_property_details.append(normalized_df)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.exception(f"Error normalizing property {property_id}")
            continue
    
    if not all_property_details:
        logger.warning("No property details retrieved")
        return pd.DataFrame()
    
    final_df = pd.concat(all_property_details, ignore_index=True)
    
    # Clean column names (replace dots with underscores)
    final_df.columns = [col.replace(".", "_") for col in final_df.columns]
    
    logger.info(f"Successfully processed {len(final_df)} properties")
    return final_df


def convert_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dict/list columns to JSON strings for database storage.
    
    Args:
        df: DataFrame with mixed column types
    
    Returns:
        DataFrame with JSON columns converted to strings
    """
    json_columns = [
        "mortgage", "buyers", "one_time_fees", "units", "description_styles",
        "tax_history", "community_rental_floorplans", "photos", "virtual_tours",
        "property_history", "location_neighborhoods", "pet_policy",
        "nearby_schools", "schools", "local_noise_noise_categories",
        "local_flood_fema_zone", "local_flood_insurance_rates",
        "estimates_current_values", "estimates_historical_values",
        "estimates_forecast_values"
    ]
    
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
    
    logger.info("✅ JSON columns converted to strings")
    return df


def insert_property_data(engine, df: pd.DataFrame) -> bool:
    """
    Insert property data into database.
    
    Args:
        engine: SQLAlchemy engine
        df: DataFrame containing property data
    
    Returns:
        True if successful, False otherwise
    """
    try:
        df.to_sql(
            name="property_data",
            con=engine,
            if_exists="append",
            index=False,
            method='multi',
            chunksize=20
        )
        logger.info(f"✅ Successfully inserted {len(df)} property records")
        return True
    except Exception as e:
        logger.error("❌ Failed to insert property data")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error(traceback.format_exc())
        return False


@app.timer_trigger(
    schedule="0 0 14 * * *",
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=False
)
def realtor_scrape_trigger(myTimer: func.TimerRequest) -> None:
    """
    Main Azure Function triggered daily at 2:00 PM UTC.
    
    Orchestrates the complete workflow:
    1. Connects to database
    2. Fetches all active rental listings
    3. Updates active_listings table
    4. Identifies new properties
    5. Fetches detailed property information
    6. Inserts into property_data table
    """
    if myTimer.past_due:
        logger.info('⚠️ Timer trigger is past due')
    
    logger.info("🚀 Starting realtor scrape job")
    
    # Get environment variables
    import os
    connection_string = os.environ.get('REALTOR_SQL_DB_CONNECTION_STRING')
    if not connection_string:
        logger.error("❌ REALTOR_SQL_DB_CONNECTION_STRING not set")
        return
    
    # Create database connection
    conn = create_db_connection(connection_string)
    if not conn:
        logger.error("❌ Unable to establish database connection, aborting")
        return
    
    cursor = conn.cursor()
    
    try:
        # Get API headers
        headers = get_api_headers()
        
        # Fetch all listings
        logger.info("📋 Fetching rental listings...")
        listings_df = fetch_all_listings(headers)
        
        if listings_df.empty:
            logger.error("❌ No listings fetched, aborting")
            return
        
        # Prepare and insert active listings
        clean_listings = prepare_listings_data(listings_df)
        insert_active_listings(cursor, clean_listings)
        conn.commit()
        logger.info("✅ Active listings updated")
        
        # Create SQLAlchemy engine for bulk insert
        engine = create_sqlalchemy_engine(connection_string)
        if not engine:
            logger.error("❌ Unable to create SQLAlchemy engine, skipping property details")
            return
        
        # Identify new properties
        logger.info("🔍 Identifying new properties...")
        existing_properties_df = pd.read_sql_query(
            "SELECT * FROM property_data", 
            conn
        ).drop_duplicates(subset=['property_id'])
        
        existing_ids = set(existing_properties_df['property_id'].tolist())
        logger.info(f"📊 Found {len(existing_ids)} existing properties in database")
        
        new_properties_df = clean_listings[
            ~clean_listings["property_id"].isin(existing_ids)
        ].drop_duplicates(subset=["property_id"])
        
        logger.info(f"🆕 Found {len(new_properties_df)} new properties to process")
        
        if new_properties_df.empty:
            logger.info("✅ No new properties to process")
            return
        
        # Fetch and process property details
        logger.info("📝 Fetching property details...")
        property_details_df = process_new_properties(headers, new_properties_df)
        
        if property_details_df.empty:
            logger.warning("⚠️ No property details retrieved")
            return
        
        # Convert JSON columns and insert
        property_details_df = convert_json_columns(property_details_df)
        insert_property_data(engine, property_details_df)
        
        logger.info("✅ Realtor scrape job completed successfully")
        
    except Exception as e:
        logger.exception("❌ Unexpected error in main workflow")
        
    finally:
        cursor.close()
        conn.close()
        logger.info("✅ Database connection closed")