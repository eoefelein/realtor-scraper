Austin Rental Property Scraper
An Azure Function that scrapes rental listings from Realtor.com for Austin, TX and stores them in Azure SQL Database.

**What It Does**

Runs daily at 9 AM UTC

Fetches rental properties from Realtor.com API

Validates data quality (prices, locations, property details)

Stores listings in Azure SQL Database

Tracks new properties and updates existing ones

**Setup**
1. Install Dependencies
bashpip install -r requirements.txt
2. Configure Environment Variables
Create a .env file:
REALTOR_SQL_DB_CONNECTION_STRING=your_connection_string
RAPIDAPI_KEY=your_api_key
3. Run Locally
bashfunc start

**Database Schema**

*active_listings*

property_id - Unique property identifier

listing_id - MLS listing ID

location_address_line - Street address

location_address_zipcode - Zip code

list_date - Date property was listed

last_observed - Last time we saw this listing

*property_data*

Contains detailed property information including:

Pricing and financials

Beds, baths, square footage

Location coordinates

Property features

School information

Pet policies

**Tests**

Testing verifies:

Prices are reasonable ($500-$25,000/month)

Coordinates are within Austin bounds

Required fields are present

No duplicate properties

Dates are valid

Property specs (beds/baths/sqft) are reasonable

**Data Validation**

The scraper validates:

Prices: $500 - $25,000 per month

Location: Coordinates within Austin, TX bounds (30.0-30.6°N, -98.0 to -97.5°W)

Property specs: Reasonable bedroom, bathroom, and square footage ranges

Dates: Not in the future, not older than 2 years

Duplicates: Removes duplicate property IDs

**Project Structure**

realtor-scraper/

├── function_app.py                    # Main Azure Function

├── test_realtor_property_data.py      # Data validation tests

├── requirements.txt                   # Dependencies

├── .env.example                       # Environment variable template

├── .gitignore                         # Git ignore rules

└── README.md                          # This file

**Key Features**

Retry Logic: Automatically retries failed database connections

Rate Limiting: Waits 1 second between API requests

Error Handling: Continues processing even if individual records fail

Comprehensive Logging: Detailed logs for monitoring and debugging

Data Quality: Validates all data before insertion
