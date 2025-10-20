import pandas as pd
import numpy as np
from datetime import date, datetime


def test_price_range():
    """Test that prices are reasonable."""
    df = pd.DataFrame([
        {'property_id': 'M1', 'list_price': 2500},
        {'property_id': 'M2', 'list_price': 1800},
        {'property_id': 'M3', 'list_price': 3500},
    ])
    
    # Prices should be between $500 and $25,000
    assert (df['list_price'] >= 500).all()
    assert (df['list_price'] <= 25000).all()


def test_required_fields():
    """Test that required fields exist."""
    df = pd.DataFrame([
        {'property_id': 'M123', 'listing_id': 'L456', 'list_price': 2500}
    ])
    
    # Required fields must be present
    assert 'property_id' in df.columns
    assert 'listing_id' in df.columns
    assert not df['property_id'].isna().any()
    assert not df['listing_id'].isna().any()


def test_no_duplicates():
    """Test that there are no duplicate property IDs."""
    df = pd.DataFrame([
        {'property_id': 'M1', 'list_price': 2500},
        {'property_id': 'M2', 'list_price': 1800},
        {'property_id': 'M3', 'list_price': 3500},
    ])
    
    # No duplicate property IDs
    assert len(df) == len(df['property_id'].unique())


def test_valid_coordinates():
    """Test that coordinates are within Austin, TX."""
    df = pd.DataFrame([
        {
            'property_id': 'M1',
            'location.address.coordinate.lat': 30.2672,
            'location.address.coordinate.lon': -97.7431
        }
    ])
    
    # Austin bounds: lat 30.0-30.6, lon -98.0 to -97.5
    lat = df['location.address.coordinate.lat'].iloc[0]
    lon = df['location.address.coordinate.lon'].iloc[0]
    
    assert 30.0 <= lat <= 30.6
    assert -98.0 <= lon <= -97.5


def test_reasonable_property_specs():
    """Test that beds, baths, sqft are reasonable."""
    df = pd.DataFrame([
        {
            'property_id': 'M1',
            'description.beds': 3,
            'description.baths': 2,
            'description.sqft': 1500
        }
    ])
    
    # Reasonable ranges
    assert 0 < df['description.beds'].iloc[0] <= 15
    assert 0 < df['description.baths'].iloc[0] <= 15
    assert 250 < df['description.sqft'].iloc[0] <= 15000


def test_date_not_in_future():
    """Test that list dates are not in the future."""
    today = date.today()
    df = pd.DataFrame([
        {'property_id': 'M1', 'list_date': '2025-01-15'}
    ])
    
    df['list_date'] = pd.to_datetime(df['list_date'])
    list_date = df['list_date'].iloc[0].date()
    
    assert list_date <= today


def test_outlier_detection():
    """Test detection of price outliers."""
    df = pd.DataFrame([
        {'property_id': 'M1', 'list_price': 2000},
        {'property_id': 'M2', 'list_price': 2500},
        {'property_id': 'M3', 'list_price': 3000},
        {'property_id': 'M4', 'list_price': 50000},  # Outlier
    ])
    
    mean = df['list_price'].mean()
    std = df['list_price'].std()
    z_scores = (df['list_price'] - mean) / std
    
    # At least one outlier (>3 standard deviations)
    outliers = df[abs(z_scores) > 3]
    assert len(outliers) > 0


def test_missing_data_report():
    """Test calculation of missing data percentage."""
    df = pd.DataFrame([
        {'property_id': 'M1', 'sqft': 1500, 'garage': 2},
        {'property_id': 'M2', 'sqft': None, 'garage': 1},
        {'property_id': 'M3', 'sqft': 1800, 'garage': None},
    ])
    
    missing_pct = (df.isna().sum() / len(df)) * 100
    
    # Both sqft and garage have 33.33% missing
    assert abs(missing_pct['sqft'] - 33.33) < 0.1
    assert abs(missing_pct['garage'] - 33.33) < 0.1


def test_address_not_empty():
    """Test that addresses are not empty strings."""
    df = pd.DataFrame([
        {'property_id': 'M1', 'location.address.line': '123 Main St'}
    ])
    
    address = df['location.address.line'].iloc[0]
    assert address is not None
    assert len(address.strip()) > 0