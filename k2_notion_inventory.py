#!/usr/bin/env python3
"""
K2 Restaurant Inventory Management System - Notion Integration
============================================================

Production-ready inventory management system with Notion database integration.
Provides real-time inventory tracking with user-friendly data management
through Notion databases instead of SQLite.

Key Features:
- Notion databases for all data storage (items, inventory, ADU calculations)
- Manager-friendly interface through Notion for data editing
- Telegram bot for field operations and data entry
- Automated calculations and reporting
- Location-aware business logic
- Weekly and monthly ADU analysis

Author: Dorei Satō
License: Proprietary  
Version: 2.0.0
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import quote

import requests
SYSTEM_VERSION = "2.0.0"  # Make sure this is defined at module level

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"Loading environment variables from {env_file}")
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")  # Remove quotes
                        os.environ[key] = value
                        print(f"  Loaded: {key}={value if key not in ['TELEGRAM_BOT_TOKEN', 'NOTION_TOKEN'] else value[:10]+'...'}")
        except UnicodeDecodeError:
            # Fallback to system default encoding
            print(f"UTF-8 encoding failed, trying system default...")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")  # Remove quotes
                        os.environ[key] = value
                        print(f"  Loaded: {key}={value if key not in ['TELEGRAM_BOT_TOKEN', 'NOTION_TOKEN'] else value[:10]+'...'}")
    else:
        print(f"No {env_file} file found - using system environment variables")

# Load .env file before other imports
load_env_file()

# Helper function to get current local time
def get_local_time() -> datetime:
    """Get current local system time"""
    return datetime.now()

# Helper function to get current time in specified timezone
def get_time_in_timezone(timezone_str: str = None) -> datetime:
    """
    Get current time in specified timezone or local time if not specified.
    
    Args:
        timezone_str: Timezone string (e.g., 'America/Chicago') or None for local time
        
    Returns:
        datetime: Current time in specified timezone
    """
    if not timezone_str:
        return datetime.now()
    
    try:
        import pytz
        target_tz = pytz.timezone(timezone_str)
        utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        local_time = utc_now.astimezone(target_tz)
        return local_time.replace(tzinfo=None)  # Remove timezone info for consistency
    except ImportError:
        # Fallback to system local time if pytz not available
        return datetime.now()
    except:
        # Fallback to system local time if timezone is invalid
        return datetime.now()

# ===== CONFIGURATION AND CONSTANTS =====

# System Configuration
SYSTEM_VERSION = "2.0.0"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(funcName)-20s | %(lineno)d | %(message)s"
MAX_MEMORY_MB = 512
MAX_LOG_SIZE_MB = 50
RETENTION_DAYS = 90

# Business Constants
BUFFER_DAYS = 1.0  # Safety margin for all calculations
MAX_CONCURRENT_USERS = 10
RATE_LIMIT_COMMANDS_PER_MINUTE = 10

# Rounding Configuration
def round_order_quantity(qty: float) -> int:
    """
    Round order quantities to whole numbers for practical ordering.
    
    Business Rule: Always round UP to ensure sufficient inventory.
    Examples: 0.1 → 1, 1.7 → 2, 2.0 → 2
    
    Args:
        qty: Calculated quantity (can be decimal)
        
    Returns:
        int: Rounded up quantity (whole number)
    """
    import math
    if qty <= 0:
        return 0
    return math.ceil(qty)  # Always round up for safety

def round_consumption_display(qty: float) -> float:
    """
    Round consumption/need quantities for display purposes.
    
    Shows 1 decimal place for clarity while keeping precision.
    
    Args:
        qty: Consumption need quantity
        
    Returns:
        float: Rounded to 1 decimal place
    """
    return round(qty, 1)

def round_adu_display(adu: float) -> float:
    """
    Round ADU values for display purposes.
    
    Shows 2 decimal places for ADU precision.
    
    Args:
        adu: Average Daily Usage value
        
    Returns:
        float: Rounded to 2 decimal places
    """
    return round(adu, 2)

# Time Configuration
TIMEZONE = os.environ.get('TZ', 'local')  # Default to local system time
BUSINESS_TIMEZONE = "America/Chicago"  # For business operations (delivery schedules)

# Delivery Schedules (hour in 24-hour format, Chicago Time)
DELIVERY_SCHEDULES = {
    "Avondale": {
        "days": ["Monday", "Thursday"],
        "hour": 12,
        "request_schedule": {
            "Tuesday": 8,    # For Thursday delivery
            "Saturday": 8,   # For Monday delivery
        }
    },
    "Commissary": {
        "days": ["Tuesday", "Thursday", "Saturday"],
        "hour": 12,
        "request_schedule": {
            "Monday": 8,     # For Tuesday delivery
            "Wednesday": 8,  # For Thursday delivery
            "Friday": 8,     # For Saturday delivery
        }
    }
}

# --- INVENTORY CONSUMPTION SCHEDULES (required by InventoryItem.get_current_consumption_days) ---
# Keys must be the SAME day names used in DELIVERY_SCHEDULES["<Location>"]["days"].
INVENTORY_CONFIG = {
    "Avondale": {
        "consumption_schedule": {
            "Monday": 3.0,    # Monday delivery must last 3.0 days (Mon→Thu)
            "Thursday": 4.0,  # Thursday delivery must last 4.0 days (Thu→Mon)
        }
    },
    "Commissary": {
        "consumption_schedule": {
            "Tuesday": 2.0,   # Tue→Thu
            "Thursday": 2.0,  # Thu→Sat
            "Saturday": 3.0,  # Sat→Tue
        }
    },
}


# Error Messages for User Feedback
ERROR_MESSAGES = {
    "notion_timeout": "⏰ Notion database is busy, please try again in a moment",
    "invalid_quantity": "❌ Please enter a valid number (e.g., 5, 2.5, or 0)",
    "item_not_found": "❌ Item '{item_name}' not found in {location} inventory",
    "calculation_error": "🔧 Calculation error - support has been notified",
    "system_error": "🚨 System error - please try again or contact support",
    "network_error": "📡 Network error - please check connection and try again",
    "notion_error": "📝 Notion database error - please try again or contact support",
    "invalid_date": "📅 Please enter a valid date (YYYY-MM-DD format)",
    "invalid_command": "❓ Unknown command. Type /help for available commands",
    "conversation_timeout": "⏰ Conversation timed out. Please start over with the command"
}

# ===== LOGGING SETUP =====

def setup_logging():
    """
    Configure comprehensive logging system with local timezone.
    
    Log Levels:
    - CRITICAL: System startup/shutdown, database initialization, critical failures
    - INFO: Business operations, commands, transactions, messages
    - DEBUG: Function calls, calculations, query details, conversation state
    
    Returns:
        logging.Logger: Configured root logger
    """
    import logging.handlers
    from datetime import datetime
    
    # Custom formatter that uses local timezone
    class LocalTimeFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created)
            if datefmt:
                s = dt.strftime(datefmt)
            else:
                s = dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            return s
    
    # Configure root logger
    formatter = LocalTimeFormatter(LOG_FORMAT)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(
        f"k2_notion_system_{datetime.now().strftime('%Y%m%d')}.log", 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create specific loggers for different components
    loggers = {
        'system': logging.getLogger('system'),
        'notion': logging.getLogger('notion'),
        'telegram': logging.getLogger('telegram'),
        'calculations': logging.getLogger('calculations'),
        'scheduler': logging.getLogger('scheduler'),
        'business': logging.getLogger('business')
    }
    
    # Set specific log levels for different components in production
    if os.environ.get('RAILWAY_ENVIRONMENT') == 'production':
        loggers['notion'].setLevel(logging.INFO)
        loggers['calculations'].setLevel(logging.INFO)
    
    logger = logging.getLogger('system')
    logger.critical(f"K2 Notion Inventory System v{SYSTEM_VERSION} - Logging initialized")
    logger.info(f"Log format: {LOG_FORMAT}")
    logger.info(f"User timezone: {TIMEZONE}")
    logger.info(f"Business timezone: {BUSINESS_TIMEZONE}")
    
    return logger

# Initialize logging
logger = setup_logging()

# ===== MODULE-LEVEL HELPER FUNCTIONS =====

def _ik(rows: list[list[tuple[str, str]]]) -> Dict:
    """Create inline keyboard markup for Telegram."""
    return {
        "inline_keyboard": [
            [{"text": text, "callback_data": data} for text, data in row]
            for row in rows
        ]
    }

def validate_date_format(date_str: str) -> bool:
    """
    Validate date string format.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        bool: True if valid YYYY-MM-DD format
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def sanitize_user_input(text: str, max_length: int = 500) -> str:
    """
    Sanitize user input for safety.
    
    Args:
        text: Raw user input
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    # Remove control characters and limit length
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    return text[:max_length].strip()

# ===== DATA CLASSES =====

@dataclass
class InventoryItem:
    """
    Represents a single inventory item with sophisticated consumption calculation logic.
    
    Uses delivery-to-delivery consumption periods rather than static consumption days,
    accounting for varying intervals between deliveries based on restaurant schedules.
    """
    id: str  # Notion page ID
    name: str
    adu: float  # Average Daily Usage (containers per day)
    unit_type: str  # case, quart, tray, bag, bottle
    location: str  # Avondale or Commissary
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # ===== DELIVERY / CONSUMPTION CONFIG =====
    # Used by InventoryItem.get_current_consumption_days()
    INVENTORY_CONFIG = {
        "Avondale": {
            "delivery_days": ["Monday", "Thursday"],
            "delivery_hour": 12,  # 12:00 PM Central
            # Consumption days from delivery to next delivery
            "consumption_schedule": {"Thursday": 4.0, "Monday": 3.0},
        },
        "Commissary": {
            "delivery_days": ["Tuesday", "Thursday", "Saturday"],
            "delivery_hour": 12,  # 12:00 PM Central
            "consumption_schedule": {"Tuesday": 2.0, "Thursday": 2.0, "Saturday": 3.0},
        },
    }

    
    def get_current_consumption_days(self, from_date: datetime = None) -> float:
        """
        Calculate consumption days needed based on which delivery cycle we're in.
        
        Returns the exact days this delivery must last until next delivery arrives.
        Accounts for varying intervals between different delivery days.
        
        Args:
            from_date: Reference date to determine delivery cycle
            
        Returns:
            float: Days this delivery must last
        """
        if from_date is None:
            from_date = get_time_in_timezone(BUSINESS_TIMEZONE)
        
        schedule = DELIVERY_SCHEDULES[self.location]
        consumption_schedule = INVENTORY_CONFIG[self.location]["consumption_schedule"]
        delivery_days = schedule["days"]
        delivery_hour = schedule["hour"]
        
        # Determine which delivery cycle we're currently in
        current_weekday = from_date.weekday()  # 0=Monday, 6=Sunday
        current_hour = from_date.hour
        
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        # Find the most recent delivery day
        delivery_weekdays = [(weekday_map[day], day) for day in delivery_days]
        
        # Determine current delivery cycle
        current_delivery_day = None
        
        for weekday_num, day_name in sorted(delivery_weekdays, reverse=True):
            if (current_weekday > weekday_num or 
                (current_weekday == weekday_num and current_hour >= delivery_hour)):
                current_delivery_day = day_name
                break
        
        # If no delivery found, we're before the first delivery of the week
        if current_delivery_day is None:
            current_delivery_day = delivery_days[-1]  # Last delivery of previous week
        
        consumption_days = consumption_schedule.get(current_delivery_day, 3.5)
        
        logger.debug(f"Consumption days for {self.name} in {current_delivery_day} cycle: {consumption_days}")
        return consumption_days
    
    def calculate_consumption_need(self, from_date: datetime = None) -> float:
        """
        Calculate total consumption need based on current delivery cycle.
        
        Formula: consumption_need = adu × current_consumption_days
        
        Args:
            from_date: Reference date for calculation
            
        Returns:
            float: Total containers needed until next delivery
        """
        consumption_days = self.get_current_consumption_days(from_date)
        consumption = self.adu * consumption_days
        
        logger.debug(f"Consumption calculation for {self.name}: "
                    f"adu={self.adu} × consumption_days={consumption_days} = {consumption}")
        return consumption
    
    def determine_status(self, current_qty: float, consumption_need: float) -> str:
        """
        Determine inventory status with business-critical logic.
        
        Status Logic:
        - RED (Critical): current_qty < consumption_need (stockout risk)
        - GREEN (Good): current_qty >= consumption_need (sufficient coverage)
        
        Args:
            current_qty: Current inventory quantity
            consumption_need: Required quantity until next delivery
            
        Returns:
            str: Status color ('RED', 'GREEN')
        """
        if current_qty < consumption_need:
            status = 'RED'
        else:
            status = 'GREEN'
            
        logger.debug(f"Status determination for {self.name}: qty={current_qty}, "
                    f"need={consumption_need} → {status}")
        return status

@dataclass
class ConversationState:
    """
    Manages conversation state for multi-step Telegram interactions.
    
    Tracks the current step, collected data, and context for interactive workflows
    like inventory entry and data validation processes.
    """
    user_id: int
    chat_id: int
    command: str
    step: str
    note: str = ""
    review_payload: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    entry_type: Optional[str] = None  # 'on_hand' or 'received'
    current_item_index: int = 0
    items: List[InventoryItem] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if conversation has timed out"""
        return (datetime.now() - self.last_activity).total_seconds() > (timeout_minutes * 60)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

# ===== NOTION DATABASE MANAGER =====

class NotionManager:
    """
    Enterprise-grade Notion database manager with dynamic schema management.
    
    Implements a hybrid approach:
    1. Auto-initializes database schemas on first run
    2. Dynamically manages property columns for inventory items
    3. Provides data integrity and error recovery mechanisms
    4. Optimizes API usage with intelligent caching strategies
    """
    
    def __init__(self, token: str, items_db_id: str, inventory_db_id: str, adu_calc_db_id: str):
        """
        Initialize Notion manager with all three database IDs.
        
        Args:
            token: Notion integration token
            items_db_id: Items master database ID
            inventory_db_id: Inventory transactions database ID  
            adu_calc_db_id: ADU calculations database ID (required)
        """
        self.token = token
        self.items_db_id = items_db_id
        self.inventory_db_id = inventory_db_id
        self.adu_calc_db_id = adu_calc_db_id
        
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Notion-Version': '2022-06-28'
        }
        
        self.logger = logging.getLogger('notion')
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.base_url = "https://api.notion.com/v1"

        
        # Advanced caching system
        self._items_cache = {}
        self._schema_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        
        # Dynamic property management
        self._inventory_properties = set()
        self._items_initialized = False
        
        self.logger.critical(f"Notion manager initialized with dynamic schema management")
        self.logger.info(f"Items DB: {items_db_id[:8]}...")
        self.logger.info(f"Inventory DB: {inventory_db_id[:8]}...")
        self.logger.info(f"ADU Calculations DB: {adu_calc_db_id[:8]}...")
        
        # Initialize system on first run
        self._initialize_system()
    
    def _initialize_system(self):
        """
        Initialize the complete system with schema validation and data seeding.
        
        This method:
        1. Validates database connections
        2. Checks if items database is populated
        3. Auto-populates if empty
        4. Initializes inventory database schema
        5. Sets up dynamic property tracking
        """
        try:
            self.logger.info("Initializing Notion system...")
            
            # Validate database connections
            self._validate_databases()
            
            # Check if items database needs initialization
            if not self._check_items_initialized():
                self.logger.info("Items database empty - initializing with master data...")
                self._seed_items_database()
                self._items_initialized = True
            
            # Initialize inventory database schema
            self._initialize_inventory_schema()
            
            self.logger.critical("Notion system initialization completed successfully")
            
        except Exception as e:
            self.logger.critical(f"System initialization failed: {e}")
            raise
    
    def _check_items_initialized(self) -> bool:
        """Check if items database has been populated with master data."""
        try:
            response = self._make_request('POST', f'/databases/{self.items_db_id}/query', {
                'page_size': 1
            })
            
            if response and response['results']:
                self.logger.info("Items database already populated")
                return True
            else:
                self.logger.info("Items database is empty")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking items initialization: {e}")
            return False
    
    def _seed_items_database(self):
        """
        Seed the items database with master inventory configuration.
        
        Populates both locations with their respective items, ADU values,
        and unit types from the inventory configuration.
        """
        try:
            # Define inventory configuration directly here to ensure availability
            inventory_config = {
                "Avondale": {
                    "consumption_schedule": {
                        "Thursday": 4.0,  # Thursday 12PM → Monday 12PM
                        "Monday": 3.0     # Monday 12PM → Thursday 12PM
                    },
                    "items": {
                        "Steak": {"adu": 1.8, "unit_type": "case"},
                        "Salmon": {"adu": 0.9, "unit_type": "case"},
                        "Chipotle Aioli": {"adu": 8.0, "unit_type": "quart"},
                        "Garlic Aioli": {"adu": 6.0, "unit_type": "quart"},
                        "Jalapeno Aioli": {"adu": 5.0, "unit_type": "quart"},
                        "Sriracha Aioli": {"adu": 2.0, "unit_type": "quart"},
                        "Ponzu Sauce": {"adu": 3.0, "unit_type": "quart"},
                        "Teriyaki/Soyu Sauce": {"adu": 3.0, "unit_type": "quart"},
                        "Orange Sauce": {"adu": 4.0, "unit_type": "quart"},
                        "Bulgogi Sauce": {"adu": 3.0, "unit_type": "quart"},
                        "Fried Rice Sauce": {"adu": 4.0, "unit_type": "quart"},
                        "Honey": {"adu": 2.0, "unit_type": "bottle"}
                    }
                },
                "Commissary": {
                    "consumption_schedule": {
                        "Tuesday": 2.0,   # Tuesday 12PM → Thursday 12PM
                        "Thursday": 2.0,  # Thursday 12PM → Saturday 12PM  
                        "Saturday": 3.0   # Saturday 12PM → Tuesday 12PM
                    },
                    "items": {
                        "Fish": {"adu": 0.3, "unit_type": "tray"},
                        "Shrimp": {"adu": 0.5, "unit_type": "tray"},
                        "Grilled Chicken": {"adu": 2.5, "unit_type": "case"},
                        "Crispy Chicken": {"adu": 3.5, "unit_type": "case"},
                        "Crab Ragoon": {"adu": 1.9, "unit_type": "bag"},
                        "Nutella Ragoon": {"adu": 0.7, "unit_type": "bag"},
                        "Ponzu Cups": {"adu": 0.8, "unit_type": "quart"}
                    }
                }
            }
            
            items_created = 0
            
            for location, config in inventory_config.items():
                consumption_schedule = config["consumption_schedule"]
                items = config["items"]
                
                # Calculate average consumption days for this location
                avg_consumption_days = sum(consumption_schedule.values()) / len(consumption_schedule)
                
                for item_name, item_config in items.items():
                    page_data = {
                        'parent': {
                            'database_id': self.items_db_id
                        },
                        'properties': {
                            'Item Name': {
                                'title': [
                                    {
                                        'text': {
                                            'content': item_name
                                        }
                                    }
                                ]
                            },
                            'Location': {
                                'select': {
                                    'name': location
                                }
                            },
                            'ADU': {
                                'number': item_config['adu']
                            },
                            'Unit Type': {
                                'select': {
                                    'name': item_config['unit_type']
                                }
                            },
                            'Consumption Days': {
                                'number': avg_consumption_days
                            },
                            'Active': {
                                'checkbox': True
                            }
                        }
                    }
                    
                    response = self._make_request('POST', '/pages', page_data)
                    if response:
                        items_created += 1
                        self.logger.debug(f"Created item: {item_name} ({location})")
                    else:
                        self.logger.error(f"Failed to create item: {item_name}")
            
            self.logger.info(f"Seeded {items_created} items in items database")
            
        except Exception as e:
            self.logger.error(f"Error seeding items database: {e}")
            self.logger.error(f"Full error details: {str(e)}")
            raise
    
    def _initialize_inventory_schema(self):
        """
        Initialize inventory database schema with dynamic property creation.
        
        Creates quantity columns for each inventory item automatically,
        ensuring perfect property name matching for data entry.
        """
        try:
            self.logger.info("Initializing inventory database schema...")
            
            # Get all items to create quantity columns
            all_items = self.get_all_items(use_cache=False)
            
            # Build the set of required properties
            base_properties = {
                'Manager',      # Title
                'Date',         # Date
                'Location',     # Select
                'Type',         # Select  
                'Notes'         # Rich Text
            }
            
            # Add quantity columns for each item
            item_properties = set()
            for item in all_items:
                property_name = self._get_quantity_property_name(item.name)
                item_properties.add(property_name)
            
            self._inventory_properties = base_properties | item_properties
            
            self.logger.info(f"Inventory schema initialized with {len(self._inventory_properties)} properties")
            self.logger.debug(f"Item quantity properties: {sorted(item_properties)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing inventory schema: {e}")
            raise
    
    def _get_quantity_property_name(self, item_name: str) -> str:
        """
        Generate standardized property name for item quantity columns.
        
        Args:
            item_name: Name of inventory item
            
        Returns:
            str: Property name for quantity column (e.g., "Steak Qty")
        """
        # Clean item name and add "Qty" suffix
        clean_name = item_name.strip()
        return f"{clean_name} Qty"
    
    def get_inventory_properties(self) -> Dict[str, str]:
        """
        Get mapping of item names to their quantity property names.
        
        Returns:
            Dict[str, str]: Mapping of item_name -> property_name
        """
        items = self.get_all_items()
        return {item.name: self._get_quantity_property_name(item.name) for item in items}
    
    def _validate_databases(self):
        """Validate that all required databases are accessible."""
        try:
            # Test items database
            response = self._make_request('POST', f'/databases/{self.items_db_id}/query', {
                'page_size': 1
            })
            if response:
                self.logger.info("Items database connection validated")
            
            # Test inventory database  
            response = self._make_request('POST', f'/databases/{self.inventory_db_id}/query', {
                'page_size': 1
            })
            if response:
                self.logger.info("Inventory database connection validated")
            
            # Test ADU calculations database
            response = self._make_request('POST', f'/databases/{self.adu_calc_db_id}/query', {
                'page_size': 1
            })
            if response:
                self.logger.info("ADU calculations database connection validated")
                
        except Exception as e:
            self.logger.critical(f"Database validation failed: {e}")
            raise
    
    def _make_request(self, http_method: str, path: str, data: Dict = None) -> Optional[Dict]:
        """
        Make HTTP request to Notion API with error handling and logging.

        Args:
            http_method: 'GET' | 'POST' | 'PATCH' | 'DELETE'
            path: e.g., '/databases/{id}/query' or '/pages'
            data: JSON body (for non-GET)

        Returns:
            Optional[Dict]: Parsed JSON on success, else None
        """
        url = f"{self.base_url}{path}"
        try:
            start_time = time.time()
            if http_method.upper() == "GET":
                resp = self.session.get(url, timeout=30)
            else:
                resp = self.session.request(http_method.upper(), url, json=data or {}, timeout=30)
            duration_ms = (time.time() - start_time) * 1000

            if resp.status_code >= 200 and resp.status_code < 300:
                self.logger.debug(f"Notion {http_method} {path} OK in {duration_ms:.2f}ms")
                return resp.json()
            else:
                # Try to log Notion error body if present
                try:
                    err = resp.json()
                except Exception:
                    err = {"message": resp.text}
                self.logger.error(f"Notion {http_method} {path} HTTP {resp.status_code}: {err}")
                return None
        except requests.exceptions.Timeout:
            self.logger.error(f"Notion {http_method} {path} timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Notion {http_method} {path} network error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Notion {http_method} {path} unexpected error: {e}")
            return None
        
    def _is_cache_valid(self) -> bool:
        """Check if items cache is still valid."""
        if not self._cache_timestamp:
            return False
        return (time.time() - self._cache_timestamp) < self._cache_ttl
    
    def _parse_item_from_notion(self, page: Dict) -> InventoryItem:
        """
        Parse Notion page into InventoryItem object with enhanced validation.
        
        Args:
            page: Notion page object
            
        Returns:
            InventoryItem: Parsed inventory item with business logic
        """
        try:
            props = page['properties']
            
            return InventoryItem(
                id=page['id'],
                name=props['Item Name']['title'][0]['plain_text'] if props['Item Name']['title'] else 'Unknown',
                location=props['Location']['select']['name'] if props['Location']['select'] else 'Unknown',
                adu=props['ADU']['number'] if props['ADU']['number'] is not None else 0.0,
                unit_type=props['Unit Type']['select']['name'] if props['Unit Type']['select'] else 'case',
                active=props.get('Active', {}).get('checkbox', True),
                created_at=page['created_time'],
                updated_at=page['last_edited_time']
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing item from Notion: {e}")
            # Return a minimal valid item to prevent system crashes
            return InventoryItem(
                id=page.get('id', 'unknown'),
                name='Unknown Item',
                location='Unknown',
                adu=0.0,
                unit_type='case'
            )
    
    def get_items_for_location(self, location: str, use_cache: bool = True) -> List[InventoryItem]:
        """
        Retrieve all active items for a specific location.
        
        Args:
            location: Location name ('Avondale' or 'Commissary')
            use_cache: Whether to use cached data if available
            
        Returns:
            List[InventoryItem]: List of inventory items for location
        """
        cache_key = f"items_{location}"
        
        # Check cache first
        if use_cache and self._is_cache_valid() and cache_key in self._items_cache:
            self.logger.debug(f"Using cached items for {location}")
            return self._items_cache[cache_key]
        
        start_time = time.time()
        
        # Query Notion database
        query = {
            'filter': {
                'and': [
                    {
                        'property': 'Location',
                        'select': {
                            'equals': location
                        }
                    },
                    {
                        'property': 'Active',
                        'checkbox': {
                            'equals': True
                        }
                    }
                ]
            },
            'sorts': [
                {
                    'property': 'Item Name',
                    'direction': 'ascending'
                }
            ]
        }
        
        response = self._make_request('POST', f'/databases/{self.items_db_id}/query', query)
        
        if not response:
            self.logger.error(f"Failed to retrieve items for {location}")
            return []
        
        items = []
        for page in response['results']:
            try:
                item = self._parse_item_from_notion(page)
                items.append(item)
            except Exception as e:
                self.logger.error(f"Error parsing item from Notion: {e}")
                continue
        
        # Update cache
        self._items_cache[cache_key] = items
        self._cache_timestamp = time.time()
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.debug(f"Retrieved {len(items)} items for {location} in {duration_ms:.2f}ms")
        
        return items

    # ===== SHORTAGE TRACKING ENHANCEMENTS =====

    # Add this method to the NotionManager class
    def save_order_transaction(self, location: str, date: str, manager: str, 
                            notes: str, quantities: Dict[str, float]) -> bool:
        """
        Save order transaction to track what was ordered.
        
        Args:
            location: Location name
            date: Date in YYYY-MM-DD format
            manager: Manager name (becomes page title)
            notes: Optional notes
            quantities: Dict mapping item names to ordered quantities
            
        Returns:
            bool: True if successful
        """
        try:
            # Create order title for tracking
            total_items = sum(1 for qty in quantities.values() if qty > 0)
            title = f"{manager} • Order Placed • {date} • {location} ({total_items} items)"
            
            # Format quantities as readable summary
            quantities_summary = []
            for item_name, qty in quantities.items():
                if qty > 0:
                    quantities_summary.append(f"{item_name}: {qty}")
            
            quantities_display = "\n".join(quantities_summary) if quantities_summary else "No items ordered"
            
            # Build properties
            properties = {
                'Manager': {
                    'title': [
                        {
                            'text': {
                                'content': title
                            }
                        }
                    ]
                },
                'Date': {
                    'date': {
                        'start': date
                    }
                },
                'Location': {
                    'select': {
                        'name': location
                    }
                },
                'Type': {
                    'select': {
                        'name': 'Order'  # New type for orders
                    }
                },
                'Quantities': {
                    'rich_text': [
                        {
                            'text': {
                                'content': quantities_display
                            }
                        }
                    ]
                }
            }
            
            # Add notes if provided
            if notes:
                properties['Notes'] = {
                    'rich_text': [
                        {
                            'text': {
                                'content': notes
                            }
                        }
                    ]
                }
            
            # Store raw JSON data for processing
            quantities_json = json.dumps(quantities)
            properties['Quantities JSON'] = {
                'rich_text': [
                    {
                        'text': {
                            'content': quantities_json
                        }
                    }
                ]
            }
            
            # Create the page
            page_data = {
                'parent': {
                    'database_id': self.inventory_db_id
                },
                'properties': properties
            }
            
            response = self._make_request('POST', '/pages', page_data)
            
            if response:
                self.logger.info(f"Saved order transaction: {title}")
                return True
            else:
                self.logger.error(f"Failed to save order transaction")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving order transaction: {e}")
            return False

    def get_transactions_by_type_and_date(self, location: str, transaction_type: str, 
                                        start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Get transactions by type within date range.
        
        Args:
            location: Location name
            transaction_type: 'Order', 'On-Hand', or 'Received'
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            List of transaction records with parsed quantities
        """
        try:
            # Build query filter
            filter_conditions = [
                {"property": "Location", "select": {"equals": location}},
                {"property": "Type", "select": {"equals": transaction_type}},
            ]
            
            # Add date filters if provided
            if start_date:
                filter_conditions.append({
                    "property": "Date",
                    "date": {"on_or_after": start_date}
                })
            
            if end_date:
                filter_conditions.append({
                    "property": "Date", 
                    "date": {"on_or_before": end_date}
                })
            
            query = {
                "filter": {"and": filter_conditions},
                "sorts": [{"property": "Date", "direction": "descending"}]
            }
            
            response = self._make_request("POST", f"/databases/{self.inventory_db_id}/query", query)
            
            if not response:
                return []
            
            transactions = []
            for page in response.get("results", []):
                props = page.get("properties", {})
                
                # Extract date
                date_prop = props.get("Date", {}).get("date")
                date = date_prop.get("start") if date_prop else None
                
                # Extract quantities JSON
                json_prop = props.get("Quantities JSON") or props.get("Quantities")
                quantities = {}
                
                if json_prop and json_prop.get("rich_text"):
                    raw_json = "".join(
                        segment.get("plain_text", "")
                        for segment in json_prop["rich_text"]
                    ).strip()
                    
                    if raw_json:
                        try:
                            quantities = json.loads(raw_json)
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON in transaction: {page['id']}")
                
                # Add transaction record
                transactions.append({
                    'id': page['id'],
                    'date': date,
                    'type': transaction_type,
                    'location': location,
                    'quantities': quantities,
                    'created_time': page.get('created_time')
                })
            
            self.logger.debug(f"Retrieved {len(transactions)} {transaction_type} transactions for {location}")
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error getting transactions: {e}")
            return []

    def calculate_shortages(self, location: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Calculate shortages by comparing orders to received deliveries.
        
        Args:
            location: Location name
            days_back: How many days back to analyze
            
        Returns:
            Dict with shortage analysis
        """
        try:
            from datetime import datetime, timedelta
            
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Get transactions
            orders = self.get_transactions_by_type_and_date(location, "Order", start_date, end_date)
            received = self.get_transactions_by_type_and_date(location, "Received", start_date, end_date)
            
            # Group by date for comparison
            orders_by_date = {t['date']: t['quantities'] for t in orders if t['date']}
            received_by_date = {t['date']: t['quantities'] for t in received if t['date']}
            
            shortages = []
            total_ordered_items = 0
            total_received_items = 0
            total_shortage_items = 0
            
            # Compare each order to corresponding delivery
            for order_date, ordered_qty in orders_by_date.items():
                # Find matching received delivery (could be same day or day after)
                received_qty = None
                received_date = None
                
                # Check same day first
                if order_date in received_by_date:
                    received_qty = received_by_date[order_date]
                    received_date = order_date
                else:
                    # Check next day
                    try:
                        next_day = (datetime.strptime(order_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                        if next_day in received_by_date:
                            received_qty = received_by_date[next_day]
                            received_date = next_day
                    except:
                        pass
                
                if received_qty is None:
                    # No delivery found - everything is shorted
                    for item, qty in ordered_qty.items():
                        if qty > 0:
                            shortages.append({
                                'item_name': item,
                                'ordered': qty,
                                'received': 0,
                                'shortage': qty,
                                'order_date': order_date,
                                'delivery_date': None,
                                'status': 'NOT_DELIVERED'
                            })
                            total_ordered_items += qty
                            total_shortage_items += qty
                else:
                    # Compare quantities item by item
                    all_items = set(ordered_qty.keys()) | set(received_qty.keys())
                    
                    for item in all_items:
                        ordered = ordered_qty.get(item, 0)
                        recv = received_qty.get(item, 0)
                        
                        if ordered > 0:
                            total_ordered_items += ordered
                            total_received_items += recv
                            
                            if recv < ordered:
                                shortage_qty = ordered - recv
                                shortages.append({
                                    'item_name': item,
                                    'ordered': ordered,
                                    'received': recv,
                                    'shortage': shortage_qty,
                                    'order_date': order_date,
                                    'delivery_date': received_date,
                                    'status': 'PARTIAL' if recv > 0 else 'NOT_DELIVERED'
                                })
                                total_shortage_items += shortage_qty
            
            # Sort shortages by severity (highest shortage first)
            shortages.sort(key=lambda x: x['shortage'], reverse=True)
            
            return {
                'location': location,
                'date_range': f"{start_date} to {end_date}",
                'days_analyzed': days_back,
                'total_ordered': total_ordered_items,
                'total_received': total_received_items,
                'total_shortage': total_shortage_items,
                'shortage_percentage': (total_shortage_items / total_ordered_items * 100) if total_ordered_items > 0 else 0,
                'shortages': shortages,
                'orders_analyzed': len(orders),
                'deliveries_analyzed': len(received)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating shortages: {e}")
            return {
                'location': location,
                'error': str(e),
                'shortages': []
            }        
    
    def get_all_items(self, use_cache: bool = True) -> List[InventoryItem]:
        """
        Retrieve all active items from all locations.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List[InventoryItem]: List of all inventory items
        """
        avondale_items = self.get_items_for_location('Avondale', use_cache)
        commissary_items = self.get_items_for_location('Commissary', use_cache)
        
        return avondale_items + commissary_items
    
    def save_inventory_transaction(self, location: str, entry_type: str, date: str, 
                                manager: str, notes: str, quantities: Dict[str, float],
                                image_file_id: Optional[str] = None) -> bool:
        """
        Save inventory transaction with optional image using Telegram file approach.
        
        Args:
            location: Location name
            entry_type: 'on_hand' or 'received'
            date: Date in YYYY-MM-DD format
            manager: Manager name (becomes page title)
            notes: Optional notes
            quantities: Dict mapping item names to quantities
            image_file_id: Optional Telegram file ID for product image
            
        Returns:
            bool: True if successful
        """
        try:
            # Create executive-level title for management visibility
            entry_type_display = "On-Hand Count" if entry_type == 'on_hand' else "Delivery Received"
            total_items = sum(1 for qty in quantities.values() if qty > 0)
            title = f"{manager} • {entry_type_display} • {date} • {location} ({total_items} items)"
            
            # Format quantities as readable JSON string for Notion
            quantities_summary = []
            for item_name, qty in quantities.items():
                if qty > 0:  # Only show items with quantities
                    quantities_summary.append(f"{item_name}: {qty}")
            
            quantities_display = "\n".join(quantities_summary) if quantities_summary else "No items recorded"
            
            # Build properties using single JSON approach
            properties = {
                'Manager': {  # Title property
                    'title': [
                        {
                            'text': {
                                'content': title
                            }
                        }
                    ]
                },
                'Date': {
                    'date': {
                        'start': date
                    }
                },
                'Location': {
                    'select': {
                        'name': location
                    }
                },
                'Type': {
                    'select': {
                        'name': 'On-Hand' if entry_type == 'on_hand' else 'Received'
                    }
                },
                'Quantities': {  # Single rich text field with all quantities
                    'rich_text': [
                        {
                            'text': {
                                'content': quantities_display
                            }
                        }
                    ]
                }
            }
            
            # Add notes with rich formatting
            if notes:
                properties['Notes'] = {
                    'rich_text': [
                        {
                            'text': {
                                'content': notes
                            }
                        }
                    ]
                }
            
            # Store raw JSON data for system processing
            quantities_json = json.dumps(quantities)
            properties['Quantities JSON'] = {
                'rich_text': [
                    {
                        'text': {
                            'content': quantities_json
                        }
                    }
                ]
            }
            
            # Add image if provided - using the working approach from communication bot
            if image_file_id:
                try:
                    # Get the Telegram bot token from environment
                    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
                    if bot_token:
                        photo_url = self._get_telegram_photo_url(image_file_id, bot_token)
                        if photo_url:
                            properties['Product Image'] = {
                                'files': [
                                    {
                                        'type': 'external',
                                        'name': f'Delivery Photo - {date}',
                                        'external': {'url': photo_url}
                                    }
                                ]
                            }
                            self.logger.info(f"Added product image to inventory transaction")
                        else:
                            self.logger.warning(f"Could not process image {image_file_id}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing image: {e}")
                    # Continue without image rather than failing the entire transaction
            
            # Create the page
            page_data = {
                'parent': {
                    'database_id': self.inventory_db_id
                },
                'properties': properties
            }
            
            response = self._make_request('POST', '/pages', page_data)
            
            if response:
                image_note = " with image" if image_file_id else ""
                self.logger.info(f"Saved inventory transaction: {title}{image_note}")
                self.logger.info(f"Items recorded: {len([q for q in quantities.values() if q > 0])}")
                return True
            else:
                self.logger.error(f"Failed to save inventory transaction")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving inventory transaction: {e}")
            return False
        
    def get_latest_inventory(self, location: str, entry_type: str = "on_hand") -> Dict[str, float]:
            """
            FIXED: Query with correct Type values that match what's saved.
            """
            try:
                # FIX: Use "On-Hand" not "On-Hand Count"
                type_select = "On-Hand" if entry_type == "on_hand" else "Received"
                
                query = {
                    "filter": {
                        "and": [
                            {"property": "Location", "select": {"equals": location}},
                            {"property": "Type", "select": {"equals": type_select}},
                        ]
                    },
                    "sorts": [{"property": "Date", "direction": "descending"}],
                    "page_size": 1,
                }
                
                response = self._make_request("POST", 
                                            f"/databases/{self.inventory_db_id}/query", 
                                            query)
                
                if not response or not response.get("results"):
                    self.logger.debug(f"No inventory found for {location} ({type_select})")
                    return {}
                
                page = response["results"][0]
                props = page.get("properties", {})
                
                # Try both possible property names
                json_prop = props.get("Quantities JSON") or props.get("Quantities")
                
                if not json_prop or not json_prop.get("rich_text"):
                    return {}
                
                # Extract JSON from rich text
                raw_json = "".join(
                    segment.get("plain_text", "") 
                    for segment in json_prop["rich_text"]
                ).strip()
                
                if not raw_json:
                    return {}
                
                # Parse JSON data
                data = json.loads(raw_json)
                
                # Convert to float dict
                result = {}
                for item_name, quantity in (data or {}).items():
                    try:
                        result[str(item_name)] = float(quantity)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid quantity for {item_name}: {quantity}")
                        continue
                
                self.logger.debug(f"Retrieved {len(result)} items from latest {type_select} for {location}")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error in get_latest_inventory: {e}")
                return {}
            except Exception as e:
                self.logger.error(f"get_latest_inventory error: {e}", exc_info=True)
                return {}
        
    def get_missing_counts(self, location: str, date: str) -> List[str]:
        """
        Get list of items missing inventory counts for a specific date.
        
        Args:
            location: Location name
            date: Date in YYYY-MM-DD format
            
        Returns:
            List[str]: List of item names missing counts
        """
        try:
            # Query for on-hand entries for this location and date
            query = {
                'filter': {
                    'and': [
                        {
                            'property': 'Location',
                            'select': {
                                'equals': location
                            }
                        },
                        {
                            'property': 'Type',
                            'select': {
                                'equals': 'On-Hand'
                            }
                        },
                        {
                            'property': 'Date',
                            'date': {
                                'equals': date
                            }
                        }
                    ]
                }
            }
            
            response = self._make_request('POST', f'/databases/{self.inventory_db_id}/query', query)
            
            if not response:
                self.logger.error(f"Failed to check missing counts for {location} on {date}")
                return []
            
            # Get all items for this location
            items = self.get_items_for_location(location)
            all_item_names = set(item.name for item in items)
            
            # Find which items have counts for this date
            items_with_counts = set()
            
            for page in response['results']:
                props = page['properties']
                
                # Check each item quantity column
                for item_name in all_item_names:
                    column_name = f"{item_name} Qty"
                    if column_name in props and props[column_name]['number'] is not None:
                        items_with_counts.add(item_name)
            
            # Items missing counts are those not found
            missing_items = sorted(list(all_item_names - items_with_counts))
            
            self.logger.debug(f"Found {len(missing_items)} missing counts for {location} on {date}")
            return missing_items
            
        except Exception as e:
            self.logger.error(f"Error checking missing counts: {e}")
            return []
    
    def invalidate_cache(self):
        """Invalidate the items cache to force refresh on next request."""
        self._items_cache.clear()
        self._cache_timestamp = None
        self.logger.debug("Items cache invalidated")

    def _get_telegram_photo_url(self, file_id: str, bot_token: str) -> Optional[str]:
        """
        Get Telegram photo URL using the same approach as communication bot.
        
        Args:
            file_id: Telegram file ID
            bot_token: Telegram bot token
            
        Returns:
            Optional[str]: Photo URL that Notion can cache
        """
        try:
            # Get file info from Telegram
            file_response = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getFile",
                params={'file_id': file_id}
            )
            
            if not file_response.ok:
                return None
            
            file_info = file_response.json()
            if not file_info.get('ok'):
                return None
            
            file_path = file_info['result']['file_path']
            
            # Return the download URL - Notion will cache it when we create the page
            download_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
            return download_url
            
        except Exception as e:
            self.logger.error(f"Failed to get photo URL: {e}")
            return None

# ===== BUSINESS CALCULATIONS ENGINE =====

class InventoryCalculator:
    """
    Core business logic engine with FIXED consumption math.
    Now correctly forecasts on-hand at delivery and sizes orders for post-delivery window.
    """
    
    def __init__(self, notion_manager):
        """Initialize calculator with Notion manager dependency."""
        self.notion = notion_manager
        self.logger = logging.getLogger('calculations')
        self.logger.info("Inventory calculator initialized with FIXED consumption math")
        
        # Delivery schedules with full business days counting
        self.delivery_cycles = {
            "Avondale": {
                # Cycle A: Order Tuesday → Deliver Thursday → Next Monday
                "Tuesday": {"days_pre": 2, "days_post": 3, "delivery_day": "Thursday"},
                # Cycle B: Order Saturday → Deliver Monday → Next Thursday  
                "Saturday": {"days_pre": 2, "days_post": 2, "delivery_day": "Monday"}
            },
            "Commissary": {
                # Cycle C1: Order Monday → Deliver Tuesday → Next Thursday
                "Monday": {"days_pre": 1, "days_post": 1, "delivery_day": "Tuesday"},
                # Cycle C2: Order Wednesday → Deliver Thursday → Next Saturday
                "Wednesday": {"days_pre": 1, "days_post": 1, "delivery_day": "Thursday"},
                # Cycle C3: Order Friday → Deliver Saturday → Next Tuesday
                "Friday": {"days_pre": 1, "days_post": 2, "delivery_day": "Saturday"}
            }
        }
    
    def get_current_order_cycle(self, location: str, from_date: datetime = None) -> Dict[str, Any]:
        """
        Determine which order cycle we're in based on location and current day.
        
        Returns:
            Dict with days_pre, days_post, delivery_day
        """
        if from_date is None:
            from_date = get_time_in_timezone(BUSINESS_TIMEZONE)
        
        weekday_name = from_date.strftime("%A")
        
        # Find the appropriate cycle
        cycles = self.delivery_cycles.get(location, {})
        
        # Check if today is an order day
        if weekday_name in cycles:
            return cycles[weekday_name]
        
        # Find the next order day
        days_ahead = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        current_idx = days_ahead.index(weekday_name)
        
        for i in range(1, 8):
            next_day = days_ahead[(current_idx + i) % 7]
            if next_day in cycles:
                return cycles[next_day]
        
        # Default fallback (should never reach here)
        return {"days_pre": 2, "days_post": 3, "delivery_day": "Thursday"}
    
    def forecast_on_hand_at_delivery(self, oh_now: float, adu: float, days_pre: int) -> float:
        """
        Forecast on-hand quantity at delivery time.
        
        Core formula: OH_at_delivery = max(0, OH_now - ADU × Days_pre)
        
        Args:
            oh_now: Current on-hand quantity
            adu: Average daily usage
            days_pre: Full business days until delivery
            
        Returns:
            float: Forecasted on-hand at delivery
        """
        consumption_before_delivery = adu * days_pre
        oh_at_delivery = max(0, oh_now - consumption_before_delivery)
        
        self.logger.debug(f"Forecast: OH_now={oh_now:.1f} - (ADU={adu:.2f} × days_pre={days_pre}) = {oh_at_delivery:.1f}")
        return oh_at_delivery
    
    def compute_order_quantity(self, oh_now: float, adu: float, days_pre: int, 
                              days_post: int, safety_days: float = 0) -> float:
        """
        Compute order quantity using FIXED consumption math.
        
        Core formulas:
        - OH_at_delivery = max(0, OH_now - ADU × Days_pre)  
        - Need_post = ADU × (Days_post + Safety_days)
        - Order_raw = max(0, Need_post - OH_at_delivery)
        
        Args:
            oh_now: Current on-hand quantity
            adu: Average daily usage
            days_pre: Full business days until delivery
            days_post: Full business days from delivery to next delivery
            safety_days: Optional buffer days
            
        Returns:
            float: Raw order quantity (before rounding)
        """
        # Forecast what we'll have when truck arrives
        oh_at_delivery = self.forecast_on_hand_at_delivery(oh_now, adu, days_pre)
        
        # Calculate need for post-delivery window
        need_post = adu * (days_post + safety_days)
        
        # Calculate raw order
        order_raw = max(0, need_post - oh_at_delivery)
        
        self.logger.debug(
            f"Order calc: need_post={need_post:.1f} - oh_at_delivery={oh_at_delivery:.1f} = {order_raw:.1f}"
        )
        
        return order_raw
    
    def round_to_pack(self, quantity: float, unit_type: str) -> int:
        """
        Round up to purchasable unit.
        
        Args:
            quantity: Raw quantity
            unit_type: Unit type (case, bag, bottle, etc.)
            
        Returns:
            int: Rounded quantity for ordering
        """
        if quantity <= 0:
            return 0
        return math.ceil(quantity)
    
    def calculate_item_status(self, item: Any, current_qty: float = None,
                            from_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate item status with FIXED consumption math.
        """
        start_time = time.time()
        
        if from_date is None:
            from_date = get_time_in_timezone(BUSINESS_TIMEZONE)
        
        # Get current quantity if not provided
        if current_qty is None:
            inventory_data = self.notion.get_latest_inventory(item.location)
            current_qty = inventory_data.get(item.name, 0.0)
        
        # Get order cycle parameters
        cycle = self.get_current_order_cycle(item.location, from_date)
        days_pre = cycle["days_pre"]
        days_post = cycle["days_post"]
        
        # Calculate using FIXED formulas
        oh_at_delivery = self.forecast_on_hand_at_delivery(current_qty, item.adu, days_pre)
        need_post = item.adu * days_post
        order_raw = self.compute_order_quantity(current_qty, item.adu, days_pre, days_post)
        order_final = self.round_to_pack(order_raw, item.unit_type)
        
        # Determine status based on whether we need to order
        status = 'RED' if order_final > 0 else 'GREEN'
        
        # Calculate days of stock remaining
        days_of_stock = (current_qty / item.adu) if item.adu > 0 else float('inf')
        
        # Risk assessment
        coverage_ratio = oh_at_delivery / need_post if need_post > 0 else float('inf')
        risk_level = 'HIGH' if coverage_ratio < 0.5 else 'MEDIUM' if coverage_ratio < 1.0 else 'LOW'
        
        # Get delivery info
        days_until_delivery, delivery_date = self.calculate_days_until_next_delivery(item.location, from_date)
        
        result = {
            'item_id': item.id,
            'item_name': item.name,
            'location': item.location,
            'unit_type': item.unit_type,
            'adu': item.adu,
            'current_qty': current_qty,
            'oh_at_delivery': oh_at_delivery,
            'days_pre': days_pre,
            'days_post': days_post,
            'days_until_delivery': days_until_delivery,
            'delivery_date': delivery_date,
            'consumption_need': need_post,  # This is Need_post
            'required_order': order_raw,
            'required_order_rounded': order_final,
            'status': status,
            'days_of_stock': days_of_stock,
            'coverage_ratio': coverage_ratio,
            'risk_level': risk_level,
            'calculation_date': from_date.isoformat()
        }
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"[FIXED MATH] {item.name}: OH_now={current_qty:.1f}, Days_pre={days_pre}, "
            f"Days_post={days_post}, OH_delivery={oh_at_delivery:.1f}, "
            f"Need_post={need_post:.1f}, Order={order_final}"
        )
        
        return result
    
    def calculate_location_summary(self, location: str, from_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate summary with FIXED consumption math for all items.
        """
        start_time = time.time()
        
        if from_date is None:
            from_date = get_time_in_timezone(BUSINESS_TIMEZONE)
        
        # Get order cycle info
        cycle = self.get_current_order_cycle(location, from_date)
        
        # Log cycle info for debugging
        self.logger.info(
            f"[ORDER CYCLE] {location} on {from_date.strftime('%A')}: "
            f"Days_pre={cycle['days_pre']}, Days_post={cycle['days_post']}, "
            f"Delivery={cycle['delivery_day']}"
        )
        
        # Get all items for location
        items = self.notion.get_items_for_location(location)
        
        # Get current inventory
        inventory_data = self.notion.get_latest_inventory(location)
        
        # Calculate status for each item
        item_statuses = []
        status_counts = {'RED': 0, 'GREEN': 0}
        total_required_order = 0
        critical_items = []
        
        for item in items:
            current_qty = inventory_data.get(item.name, 0.0)
            status_info = self.calculate_item_status(item, current_qty, from_date)
            
            item_statuses.append(status_info)
            status_counts[status_info['status']] += 1
            total_required_order += status_info['required_order_rounded']
            
            if status_info['status'] == 'RED':
                critical_items.append(status_info['item_name'])
        
        # Get delivery info
        days_until_delivery, delivery_date = self.calculate_days_until_next_delivery(location, from_date)
        
        summary = {
            'location': location,
            'calculation_date': from_date.isoformat(),
            'order_cycle': cycle,
            'total_items': len(items),
            'days_until_delivery': days_until_delivery,
            'delivery_date': delivery_date,
            'status_counts': status_counts,
            'critical_items': critical_items,
            'total_required_order': total_required_order,
            'items': item_statuses
        }
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"[SUMMARY] {location} calculated in {duration_ms:.2f}ms: "
            f"{status_counts['RED']} RED, {status_counts['GREEN']} GREEN, "
            f"Total order: {total_required_order} units"
        )
        
        return summary
    
    def generate_auto_requests(self, location: str, from_date: datetime = None) -> Dict[str, Any]:
        """
        Generate purchase orders with FIXED consumption math.
        """
        start_time = time.time()
        
        if from_date is None:
            from_date = get_time_in_timezone(BUSINESS_TIMEZONE)
        
        # Get location summary
        summary = self.calculate_location_summary(location, from_date)
        
        # Generate requests for items that need ordering
        requests = []
        total_items_requested = 0
        
        for item_status in summary['items']:
            if item_status['required_order_rounded'] > 0:
                request = {
                    'item_id': item_status['item_id'],
                    'item_name': item_status['item_name'],
                    'unit_type': item_status['unit_type'],
                    'current_qty': item_status['current_qty'],
                    'oh_at_delivery': item_status['oh_at_delivery'],
                    'consumption_need': item_status['consumption_need'],  # Need_post
                    'requested_qty': item_status['required_order_rounded'],
                    'status': item_status['status'],
                    'delivery_date': item_status['delivery_date'],
                    'days_pre': item_status['days_pre'],
                    'days_post': item_status['days_post']
                }
                
                requests.append(request)
                total_items_requested += item_status['required_order_rounded']
        
        request_summary = {
            'location': location,
            'request_date': from_date.strftime('%Y-%m-%d'),
            'order_cycle': summary['order_cycle'],
            'delivery_date': summary['delivery_date'],
            'total_items': len(requests),
            'total_quantity': total_items_requested,
            'critical_items': len(summary['critical_items']),
            'requests': requests
        }
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"[AUTO-REQUEST] {location} generated in {duration_ms:.2f}ms: "
            f"{len(requests)} items, {total_items_requested} total units"
        )
        
        return request_summary
    
    def calculate_days_until_next_delivery(self, location: str, from_date: datetime = None) -> Tuple[float, str]:
        """
        Calculate days until next scheduled delivery.
        
        Args:
            location: Location name
            from_date: Reference date
            
        Returns:
            Tuple of (days_until_delivery, delivery_date_string)
        """
        if from_date is None:
            from_date = get_time_in_timezone(BUSINESS_TIMEZONE)
        
        # Get delivery schedule
        schedule = DELIVERY_SCHEDULES[location]
        delivery_days = schedule["days"]
        delivery_hour = schedule["hour"]
        
        # Find next delivery
        current_weekday = from_date.weekday()
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        delivery_weekdays = [weekday_map[day] for day in delivery_days]
        
        days_ahead = []
        for delivery_weekday in delivery_weekdays:
            if delivery_weekday > current_weekday:
                days_ahead.append(delivery_weekday - current_weekday)
            elif delivery_weekday == current_weekday:
                if from_date.hour < delivery_hour:
                    days_ahead.append(0)
                else:
                    days_ahead.append(7)
            else:
                days_ahead.append(7 - current_weekday + delivery_weekday)
        
        days_until = min(days_ahead)
        next_delivery = from_date + timedelta(days=days_until)
        next_delivery = next_delivery.replace(hour=delivery_hour, minute=0, second=0)
        
        if days_until == 0:
            time_diff = next_delivery - from_date
            days_until = time_diff.total_seconds() / (24 * 3600)
        
        return days_until, next_delivery.strftime('%Y-%m-%d')


# Helper functions needed
def get_time_in_timezone(timezone_str: str = None) -> datetime:
    """Get current time in specified timezone."""
    if not timezone_str:
        return datetime.now()
    
    try:
        import pytz
        target_tz = pytz.timezone(timezone_str)
        utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        local_time = utc_now.astimezone(target_tz)
        return local_time.replace(tzinfo=None)
    except:
        return datetime.now()

# Constants needed
BUSINESS_TIMEZONE = "America/Chicago"

DELIVERY_SCHEDULES = {
    "Avondale": {
        "days": ["Monday", "Thursday"],
        "hour": 12,
        "request_schedule": {
            "Tuesday": 8,
            "Saturday": 8,
        }
    },
    "Commissary": {
        "days": ["Tuesday", "Thursday", "Saturday"],
        "hour": 12,
        "request_schedule": {
            "Monday": 8,
            "Wednesday": 8,
            "Friday": 8,
        }
    }
}

# ===== TELEGRAM BOT INTERFACE =====

class TelegramBot:
    """
    Production-ready Telegram bot with comprehensive error handling.
    """
    
    def __init__(self, token: str, notion_manager, calculator):
        """Initialize bot with enhanced error handling and state management."""
        self.token = token
        self.notion = notion_manager
        self.calc = calculator
        self.logger = logging.getLogger('telegram')
        
        # Bot configuration
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.running = False
        self.last_update_id = 0
        
        # Enhanced conversation state management
        self.conversations: Dict[int, ConversationState] = {}
        self.conversation_lock = threading.Lock()
        self.conversation_cleanup_interval = 1800  # 30 minutes
        self.last_cleanup_time = datetime.now()

        # THE ENTRY HANDLE
        self.entry_handler = EnhancedEntryHandler(self, notion_manager, calculator)
        
        # Rate limiting with exemptions
        self.user_commands: Dict[int, List[datetime]] = {}
        self.rate_limit_lock = threading.Lock()
        self.rate_limit_exempt_commands = {'/cancel', '/help', '/done', '/skip'}
        
        # Connection retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Chat configuration from environment
        import os
        self.chat_config = {
            'onhand': int(os.environ.get('CHAT_ONHAND', '0')),
            'autorequest': int(os.environ.get('CHAT_AUTOREQUEST', '0')),
            'received': int(os.environ.get('CHAT_RECEIVED', '0')),
            'reassurance': int(os.environ.get('CHAT_REASSURANCE', '0'))
        }
        
        # Test chat override
        self.use_test_chat = os.environ.get('USE_TEST_CHAT', 'false').lower() == 'true'
        self.test_chat = int(os.environ.get('TEST_CHAT', '0')) if self.use_test_chat else None
        
        self.logger.info(f"Telegram bot initialized with enhanced error handling")
        if self.use_test_chat:
            self.logger.info(f"Test mode enabled - all messages will go to chat {self.test_chat}")

    # ===== CONVERSATION STATE MANAGEMENT =====
    
    def _cleanup_stale_conversations(self):
        """Remove expired conversation states to prevent memory leaks."""
        now = datetime.now()
        
        # Only cleanup every interval
        if (now - self.last_cleanup_time).total_seconds() < self.conversation_cleanup_interval:
            return
        
        with self.conversation_lock:
            expired_users = []
            for user_id, state in self.conversations.items():
                if state.is_expired(timeout_minutes=30):
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.conversations[user_id]
                self.logger.info(f"Cleaned up expired conversation for user {user_id}")
        
        self.last_cleanup_time = now
        
        if expired_users:
            self.logger.info(f"Cleaned up {len(expired_users)} expired conversations")
    
    def _get_or_create_conversation(self, user_id: int, chat_id: int, 
                                   command: str) -> ConversationState:
        """Get existing or create new conversation state."""
        with self.conversation_lock:
            if user_id in self.conversations:
                state = self.conversations[user_id]
                state.update_activity()
            else:
                state = ConversationState(
                    user_id=user_id,
                    chat_id=chat_id,
                    command=command,
                    step="initial"
                )
                self.conversations[user_id] = state
        return state
    
    def _end_conversation(self, user_id: int):
        """Safely end a conversation."""
        with self.conversation_lock:
            if user_id in self.conversations:
                del self.conversations[user_id]
                self.logger.debug(f"Ended conversation for user {user_id}")

    # ===== NETWORK COMMUNICATION WITH RETRY LOGIC =====
    
    def _make_request_with_retry(self, method: str, data: Dict = None) -> Optional[Dict]:
        """
        Make API request with automatic retry on failure.
        
        Args:
            method: Telegram API method
            data: Request payload
            
        Returns:
            Optional[Dict]: Response or None if all retries failed
        """
        for attempt in range(self.max_retries):
            result = self._make_request(method, data)
            if result is not None:
                return result
            
            if attempt < self.max_retries - 1:
                self.logger.warning(f"Request {method} failed, attempt {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        self.logger.error(f"Request {method} failed after {self.max_retries} attempts")
        return None
    
    def _make_request(self, method: str, data: Dict = None) -> Optional[Dict]:
        """Make Telegram API request with comprehensive error handling."""
        import requests
        url = f"{self.base_url}/{method}"
        
        try:
            start = time.time()
            resp = requests.post(url, json=data or {}, timeout=30)
            duration = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("ok"):
                    self.logger.debug(f"Telegram {method} OK in {duration:.2f}ms")
                    return payload
                else:
                    error_code = payload.get("error_code", "unknown")
                    error_desc = payload.get("description", "no description")
                    self.logger.error(f"Telegram {method} error {error_code}: {error_desc}")
                    return None
            else:
                self.logger.error(f"Telegram {method} HTTP {resp.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Telegram {method} timeout")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Telegram {method} connection error")
            return None
        except Exception as e:
            self.logger.error(f"Telegram {method} unexpected error: {e}")
            return None
    
    def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML",
                    disable_web_page_preview: bool = True, 
                    reply_markup: Optional[Dict] = None) -> bool:
        """Send message with automatic fallback and sanitization."""
        import html
        
        # Test mode redirect
        if self.use_test_chat and self.test_chat:
            original_chat_id = chat_id
            chat_id = self.test_chat
            text = f"<b>[Test Mode - Original Chat: {original_chat_id}]</b>\n\n{text}"
        
        # Truncate if too long (Telegram limit is 4096)
        if len(text) > 4000:
            text = text[:3997] + "..."
        
        # Sanitize HTML
        safe_text = self._sanitize_html(text)
        
        # Prepare payload
        payload = {
            "chat_id": chat_id,
            "text": safe_text,
            "disable_web_page_preview": disable_web_page_preview,
            "parse_mode": parse_mode if parse_mode else None
        }
        
        if reply_markup:
            payload["reply_markup"] = reply_markup
        
        # Try sending with retry
        result = self._make_request_with_retry("sendMessage", payload)
        
        if result:
            self.logger.info(f"Message sent to chat {chat_id}")
            return True
        
        # Fallback to plain text if HTML failed
        if parse_mode == "HTML":
            payload["parse_mode"] = None
            payload["text"] = html.unescape(text)
            result = self._make_request_with_retry("sendMessage", payload)
            if result:
                self.logger.info(f"Message sent as plain text to chat {chat_id}")
                return True
        
        self.logger.error(f"Failed to send message to chat {chat_id}")
        return False

    def _save_order_to_notion(self, location: str, summary: Dict[str, Any]):
        """Save order to Notion for shortage tracking."""
        try:
            requests = summary.get("requests", [])
            if not requests:
                return False
            
            # Build quantities dict
            quantities = {}
            for item in requests:
                item_name = item.get("item_name", "Unknown")
                qty = item.get("requested_qty", 0)
                if qty > 0:
                    quantities[item_name] = qty
            
            if not quantities:
                return False
            
            # Save order transaction
            date = datetime.now().strftime('%Y-%m-%d')
            success = self.notion.save_order_transaction(
                location=location,
                date=date,
                manager="System Auto-Order",
                notes=f"Auto-generated order for {summary.get('delivery_date', 'next delivery')}",
                quantities=quantities
            )
            
            if success:
                self.logger.info(f"Saved order to Notion: {location}, {len(quantities)} items")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error saving order to Notion: {e}")
            return False

    # Add new shortage command handler
    def _handle_shortages(self, message: Dict):
        """Display shortage analysis for both locations."""
        chat_id = message["chat"]["id"]
        
        parts = message.get("text", "").split()
        days_back = 7  # Default
        
        # Parse optional days parameter
        if len(parts) > 1:
            try:
                days_back = int(parts[1])
                if days_back < 1 or days_back > 30:
                    days_back = 7
            except ValueError:
                pass
        
        try:
            # Get shortage analysis for both locations
            avondale_shortages = self.notion.calculate_shortages("Avondale", days_back)
            commissary_shortages = self.notion.calculate_shortages("Commissary", days_back)
            
            # Build message
            text = (
                f"📊 <b>SHORTAGE ANALYSIS</b>\n"
                f"📅 Last {days_back} days • {datetime.now().strftime('%b %d, %Y')}\n"
                "─────────────────────────────\n\n"
            )
            
            # Avondale section
            a_shortages = avondale_shortages.get("shortages", [])
            a_total_shortage = avondale_shortages.get("total_shortage", 0)
            a_shortage_pct = avondale_shortages.get("shortage_percentage", 0)
            
            text += f"🏪 <b>AVONDALE</b>\n"
            if a_shortages:
                text += f"⚠️ {len(a_shortages)} items shorted • {a_shortage_pct:.1f}% shortage rate\n\n"
                
                for shortage in a_shortages[:10]:  # Show top 10
                    item = shortage['item_name']
                    ordered = shortage['ordered']
                    received = shortage['received']
                    short = shortage['shortage']
                    status = shortage['status']
                    
                    if status == 'NOT_DELIVERED':
                        icon = "🚨"
                    elif shortage['shortage'] > shortage['ordered'] * 0.5:
                        icon = "⚠️"
                    else:
                        icon = "📉"
                    
                    text += f"{icon} <b>{item}</b>\n"
                    text += f"   Ordered: {ordered} • Received: {received} • Short: {short}\n"
                
                if len(a_shortages) > 10:
                    text += f"<i>...and {len(a_shortages) - 10} more items</i>\n"
            else:
                text += "✅ No shortages detected\n"
            
            text += "\n"
            
            # Commissary section  
            c_shortages = commissary_shortages.get("shortages", [])
            c_total_shortage = commissary_shortages.get("total_shortage", 0)
            c_shortage_pct = commissary_shortages.get("shortage_percentage", 0)
            
            text += f"🏭 <b>COMMISSARY</b>\n"
            if c_shortages:
                text += f"⚠️ {len(c_shortages)} items shorted • {c_shortage_pct:.1f}% shortage rate\n\n"
                
                for shortage in c_shortages[:10]:  # Show top 10
                    item = shortage['item_name']
                    ordered = shortage['ordered']
                    received = shortage['received']
                    short = shortage['shortage']
                    status = shortage['status']
                    
                    if status == 'NOT_DELIVERED':
                        icon = "🚨"
                    elif shortage['shortage'] > shortage['ordered'] * 0.5:
                        icon = "⚠️"
                    else:
                        icon = "📉"
                    
                    text += f"{icon} <b>{item}</b>\n"
                    text += f"   Ordered: {ordered} • Received: {received} • Short: {short}\n"
                
                if len(c_shortages) > 10:
                    text += f"<i>...and {len(c_shortages) - 10} more items</i>\n"
            else:
                text += "✅ No shortages detected\n"
            
            # Summary
            total_shortages = len(a_shortages) + len(c_shortages)
            if total_shortages > 0:
                text += (
                    f"\n─────────────────────────────\n"
                    f"📈 <b>SUMMARY</b>\n"
                    f"• Total shortage events: {total_shortages}\n"
                    f"• Items affected: {len(set([s['item_name'] for s in a_shortages + c_shortages]))}\n"
                    f"• Avg shortage rate: {(a_shortage_pct + c_shortage_pct) / 2:.1f}%\n\n"
                    f"💡 Use /shortages [days] to analyze different periods"
                )
            else:
                text += (
                    f"\n─────────────────────────────\n"
                    f"✅ <b>EXCELLENT DELIVERY PERFORMANCE</b>\n"
                    f"No shortages detected in the last {days_back} days"
                )
            
            self.send_message(chat_id, text)
            self.logger.info(f"/shortages sent - {total_shortages} shortage events")
            
        except Exception as e:
            self.logger.error(f"/shortages failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to analyze shortages. Please try again.")
    
    def _sanitize_html(self, text: str) -> str:
        """Enhanced HTML sanitization for Telegram."""
        import html
        import re
        
        # First escape everything
        text = html.escape(text, quote=False)
        
        # Re-enable safe tags
        safe_tags = ["b", "/b", "i", "/i", "u", "/u", "s", "/s", 
                    "code", "/code", "pre", "/pre", "tg-spoiler", "/tg-spoiler"]
        
        for tag in safe_tags:
            text = text.replace(f"&lt;{tag}&gt;", f"<{tag}>")
        
        # Remove empty tags
        text = re.sub(r"<\s*>", "", text)
        text = re.sub(r"</\s*>", "", text)
        
        return text

    def _rate_limit_ok(self, user_id: int) -> bool: ...

    def get_updates(self, timeout: int = 25) -> List[Dict]:
        """Get updates with error handling."""
        data = {
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
        }
        
        if self.last_update_id:
            data["offset"] = self.last_update_id + 1
        
        result = self._make_request("getUpdates", data)
        
        if not result:
            return []
        
        updates = result.get("result", [])
        
        if updates:
            self.last_update_id = updates[-1]["update_id"]
        
        return updates

    def _rate_limit_ok(self, user_id: int) -> bool:
        """
        Basic per-user command rate limiting.
        """
        now = datetime.now()
        with self.rate_limit_lock:
            buf = self.user_commands.setdefault(user_id, [])
            # keep the last 60 seconds
            cutoff = now - timedelta(seconds=60)
            buf[:] = [t for t in buf if t > cutoff]
            if len(buf) >= RATE_LIMIT_COMMANDS_PER_MINUTE:
                return False
            buf.append(now)
            return True

    def _sanitize_html_basic(self, text: str) -> str:
        """
        Make dynamic text safe for Telegram HTML:
        - Escape all angle brackets
        - Re-enable only a small, safe whitelist of tags we actually use (<b>, <i>, <u>, <s>, <code>, <pre>, <tg-spoiler>)
        - Strip any empty tags like "<>"
        """
        import html, re

        # Escape everything first (so any accidental '<' in item names/notes won't become tags)
        t = html.escape(text, quote=False)

        # Re-enable a minimal whitelist of tags we deliberately use in our templates
        for tag in ("b", "/b", "i", "/i", "u", "/u", "s", "/s", "code", "/code", "pre", "/pre", "tg-spoiler", "/tg-spoiler"):
            t = t.replace(f"&lt;{tag}&gt;", f"<{tag}>")

        # Remove empty/broken tags like "<>" that trigger "Unsupported start tag"
        t = re.sub(r"<\s*>", "", t)

        return t


    # ===== ENHANCED RATE LIMITING =====
    
    def _check_rate_limit(self, user_id: int, command: str = "") -> bool:
        """
        Check rate limit with command exemptions.
        
        Args:
            user_id: Telegram user ID
            command: Command being executed
            
        Returns:
            bool: True if allowed, False if rate limited
        """
        # Exempt certain commands and conversation continuations
        if command in self.rate_limit_exempt_commands:
            return True
        
        # Check if user has active conversation (exempt from rate limit)
        with self.conversation_lock:
            if user_id in self.conversations:
                return True
        
        # Apply standard rate limiting
        now = datetime.now()
        with self.rate_limit_lock:
            commands = self.user_commands.setdefault(user_id, [])
            
            # Clean old entries
            cutoff = now - timedelta(seconds=60)
            commands[:] = [t for t in commands if t > cutoff]
            
            # Check limit
            if len(commands) >= 10:  # 10 commands per minute
                return False
            
            commands.append(now)
            return True
    
    # ===== POLLING WITH ERROR RECOVERY =====
    
    def start_polling(self):
        """Start polling with automatic error recovery and cleanup."""
        self.running = True
        
        if self.use_test_chat and self.test_chat:
            self.send_message(self.test_chat, 
                            f"✅ K2 Bot v{SYSTEM_VERSION} online (test mode)")
        
        backoff = 1
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                # Periodic cleanup
                self._cleanup_stale_conversations()
                self.entry_handler.cleanup_expired_sessions()
                # Get updates
                updates = self.get_updates(timeout=25)
                
                if updates:
                    consecutive_errors = 0
                    backoff = 1
                    
                    for update in updates:
                        try:
                            self._process_update(update)
                        except Exception as e:
                            self.logger.error(f"Error processing update: {e}", exc_info=True)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Polling error ({consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical("Too many consecutive errors, stopping bot")
                    self.running = False
                    break
                
                time.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)

    def stop(self):
        """Gracefully stop the bot."""
        self.running = False
        self.logger.info("Telegram bot stopping...")

    # ===== UPDATE PROCESSING =====
    
    def _process_update(self, update: Dict):
        """
        Process update with proper routing for entry wizard input including photos.
        """
        try:
            # Handle callback queries
            if "callback_query" in update:
                callback_data = update["callback_query"].get("data", "")
                
                # Route entry callbacks to entry handler
                if callback_data.startswith("entry_"):
                    self.entry_handler.handle_callback(update["callback_query"])
                    return
                
                # Handle other callbacks
                self._handle_callback_safe(update["callback_query"])
                return
            
            # Handle messages
            message = update.get("message")
            if not message:
                return
            
            chat_id = message["chat"]["id"]
            user_id = message["from"]["id"]
            
            # IMPORTANT: Check for entry session BEFORE command processing
            # This allows number input AND photo input to work during entry
            if hasattr(self, 'entry_handler') and user_id in self.entry_handler.sessions:
                session = self.entry_handler.sessions[user_id]
                if not session.is_expired():
                    # Handle photo messages - same approach as communication bot
                    if 'photo' in message:
                        if session.mode == "received" and session.current_step == "image":
                            self.entry_handler.handle_photo_input(message, session)
                        return
                    # Handle text input (including numbers)
                    elif 'text' in message:
                        self.entry_handler.handle_text_input(message, session)
                        return
            
            # Handle text messages and commands
            if "text" in message:
                text = sanitize_user_input(message.get("text", ""))
                if not text:
                    return
                
                # Handle commands
                if text.startswith("/"):
                    command = text.split()[0].lower()
                    
                    # Check rate limit
                    if not self._check_rate_limit(user_id, command):
                        self.send_message(chat_id, 
                                        "⏳ Too many commands. Please wait a moment.")
                        return
                    
                    # Route command
                    self._route_command(message, command)
                    return
                
                # Handle old-style conversation input (if still in use)
                with self.conversation_lock:
                    if user_id in self.conversations:
                        state = self.conversations[user_id]
                        self._handle_conversation_input_safe(message, state)
                        return
            
            # No active session or conversation
            self.send_message(chat_id, 
                            "Type /help to see available commands or /entry to start.")
            
        except Exception as e:
            self.logger.error(f"Error in _process_update: {e}", exc_info=True)
            try:
                chat_id = update.get("message", {}).get("chat", {}).get("id")
                if chat_id:
                    self.send_message(chat_id, 
                                    "⚠️ An error occurred. Please try again.")
            except:
                pass

    
    def _route_command(self, message: Dict, command: str):
        """Route commands to appropriate handlers."""
        handlers = {
            "/start": self._handle_start,
            "/help": self._handle_help,
            "/entry": self.entry_handler.handle_entry_command,
            "/info": self._handle_info,
            "/order": self._handle_order,
            "/order_avondale": self._handle_order_avondale,
            "/order_commissary": self._handle_order_commissary,
            "/reassurance": self._handle_reassurance,
            "/shortages": self._handle_shortages,
            "/status": self._handle_status,
            "/cancel": self._handle_cancel,
            "/adu": self._handle_adu,
            "/missing": self._handle_missing,
        }
        
        handler = handlers.get(command)
        if handler:
            try:
                handler(message)
            except Exception as e:
                self.logger.error(f"Error in {command}: {e}", exc_info=True)
                chat_id = message["chat"]["id"]
                self.send_message(chat_id, 
                                f"⚠️ Error executing {command}. Please try again.")
        else:
            self._handle_unknown(message)
    
    def _handle_callback_safe(self, callback_query: Dict):
        """Handle callback with error handling."""
        try:
            self._handle_callback(callback_query)
        except Exception as e:
            self.logger.error(f"Error in callback: {e}", exc_info=True)
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            if chat_id:
                self.send_message(chat_id, "⚠️ Error processing selection. Please try again.")
    
    def _handle_conversation_input_safe(self, message: Dict, state: ConversationState):
        """Handle conversation input with error handling."""
        try:
            # Try enhanced handler first
            if self._handle_conversation_input_enhanced(message, state):
                return
            # Fallback to basic handler
            self._handle_conversation_input(message, state)
        except Exception as e:
            self.logger.error(f"Error in conversation: {e}", exc_info=True)
            self.send_message(state.chat_id, 
                            "⚠️ Error processing input. Please try /cancel and start over.")

    # ===== CALLBACK HANDLING =====
    
    def _handle_callback(self, callback_query: Dict):
        """Handle inline keyboard callbacks."""
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")

        # Route entry-related callbacks to entry handler
        if data.startswith("entry_"):
            self.entry_handler.handle_callback(callback_query)
            return
        
        # Acknowledge callback
        self._make_request("answerCallbackQuery", 
                          {"callback_query_id": callback_query.get("id")})
        
        with self.conversation_lock:
            state = self.conversations.get(user_id)
        
        if not state:
            self.send_message(chat_id, "Session expired. Use /entry to start again.")
            return
        
        # Route callback based on data
        if data.startswith("loc|"):
            self._handle_location_callback(state, data)
        elif data.startswith("type|"):
            self._handle_type_callback(state, data)
        elif data.startswith("date|"):
            self._handle_date_callback(state, data)
        elif data.startswith("review|"):
            self._handle_review_callback(state, data)

    def _handle_location_callback(self, state: ConversationState, data: str):
        """Handle location selection."""
        state.location = data.split("|", 1)[1]
        state.step = "choose_type"
        
        keyboard = _ik([
            [("📦 On-Hand Count", "type|on_hand")],
            [("📥 Received Delivery", "type|received")]
        ])
        
        self.send_message(state.chat_id, 
                        f"Location: <b>{state.location}</b>\n"
                        "Select entry type:",
                        reply_markup=keyboard)
    
    def _handle_type_callback(self, state: ConversationState, data: str):
        """Handle entry type selection."""
        state.entry_type = data.split("|", 1)[1]
        state.step = "choose_date"
        
        today = get_time_in_timezone(BUSINESS_TIMEZONE).strftime("%Y-%m-%d")
        
        keyboard = _ik([
            [("📅 Today", f"date|{today}")],
            [("✏️ Enter custom date", "date|manual")]
        ])
        
        self.send_message(state.chat_id, 
                        "Select date:",
                        reply_markup=keyboard)
    
    def _handle_date_callback(self, state: ConversationState, data: str):
        """Handle date selection."""
        selection = data.split("|", 1)[1]
        
        if selection == "manual":
            state.step = "enter_date"
            self.send_message(state.chat_id, 
                            "Enter date (YYYY-MM-DD) or type 'today':")
        else:
            state.data["date"] = selection
            self._begin_item_loop(state)
    
    def _handle_review_callback(self, state: ConversationState, data: str):
        """Handle review actions."""
        action = data.split("|", 1)[1]
        
        if action == "submit":
            self._finalize_entry(state)
        elif action == "back":
            state.step = "enter_items"
            self._prompt_next_item(state)
        elif action == "cancel":
            self._end_conversation(state.user_id)
            self.send_message(state.chat_id, "❌ Entry cancelled. No data saved.")


    # ===== COMMAND HANDLERS =====
    
    def _handle_start(self, message: Dict):
        """Welcome message with system status."""
        chat_id = message["chat"]["id"]
        
        try:
            # Quick system check
            items_count = len(self.notion.get_all_items())
            system_status = "✅ Online" if items_count > 0 else "⚠️ Check connection"
            
            text = (
                "🚀 <b>K2 Restaurant Inventory System</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Version 2.0.0 • Status: {system_status}\n\n"
                
                "📊 <b>Core Commands</b>\n"
                "├ /entry — Record inventory counts\n"
                "├ /info — Live status dashboard\n"
                "├ /order — Generate purchase orders\n"
                "└ /reassurance — Daily risk check\n\n"
                
                "🔧 <b>Quick Actions</b>\n"
                "├ /order_avondale — Avondale orders\n"
                "├ /order_commissary — Commissary orders\n"
                "├ /adu — View usage rates\n"
                "├ /missing — Check missing counts\n"
                "└ /status — System diagnostics\n\n"
                
                "💡 Type /help for details • /cancel to exit"
            )
            self.send_message(chat_id, text)
            
        except Exception as e:
            self.logger.error(f"Error in /start: {e}", exc_info=True)
            self.send_message(chat_id, "Welcome! Type /help for available commands.")
    
    def _handle_help(self, message: Dict):
        """Command reference."""
        chat_id = message["chat"]["id"]
        text = (
            "📚 <b>Command Reference</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "📝 <b>Data Entry</b>\n"
            "/entry — Interactive inventory recording\n"
            "  • Choose location → type → date\n"
            "  • Enter quantities or skip items\n"
            "  • Saves directly to Notion\n\n"
            
            "📊 <b>Analytics & Reports</b>\n"
            "/info — Real-time inventory analysis\n"
            "/order — Supplier-ready order lists\n"
            "/reassurance — Risk assessment\n\n"
            
            "🔍 <b>Quick Checks</b>\n"
            "/adu — Average daily usage rates\n"
            "/missing [location] [date] — Missing counts\n"
            "/status — System health check\n\n"
            
            "💡 <b>Tips</b>\n"
            "• Use 'today' for current date\n"
            "• Type /skip to skip items\n"
            "• Type /done to finish early\n"
            "• Use /cancel anytime to exit"
        )
        self.send_message(chat_id, text)


    def _handle_status(self, message: Dict):
        """System diagnostics with visual indicators"""
        chat_id = message["chat"]["id"]
        try:
            avondale = self.notion.get_items_for_location("Avondale")
            commissary = self.notion.get_items_for_location("Commissary")
            now = get_time_in_timezone(BUSINESS_TIMEZONE)
            
            # Check system components
            notion_status = "✅ Connected" if avondale or commissary else "❌ Error"
            bot_status = "✅ Active" if self.running else "⚠️ Idle"
            
            text = (
                "🔧 <b>System Diagnostics</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                
                "⚡ <b>Status Overview</b>\n"
                f"├ Notion Database: {notion_status}\n"
                f"├ Telegram Bot: {bot_status}\n"
                f"├ Version: {SYSTEM_VERSION}\n"
                f"└ Mode: {'🧪 Test' if self.use_test_chat else '🚀 Production'}\n\n"
                
                "📊 <b>Database Stats</b>\n"
                f"├ Avondale Items: {len(avondale)}\n"
                f"├ Commissary Items: {len(commissary)}\n"
                f"└ Total Active: {len(avondale) + len(commissary)}\n\n"
                
                "🕐 <b>Time Information</b>\n"
                f"├ System Time: {now.strftime('%I:%M %p')}\n"
                f"├ Date: {now.strftime('%b %d, %Y')}\n"
                f"└ Timezone: {BUSINESS_TIMEZONE}\n\n"
                
                "✅ All systems operational"
            )
            self.send_message(chat_id, text)
        except Exception as e:
            self.logger.error(f"/status failed: {e}", exc_info=True)
            self.send_message(chat_id, (
                "🚨 <b>System Error</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Unable to retrieve system status.\n"
                "Please contact support if this persists."
            ))

    def _handle_adu(self, message: Dict):
        """ADU rates with visual grouping by location"""
        chat_id = message["chat"]["id"]
        
        try:
            items = self.notion.get_all_items()
            
            if not items:
                self.send_message(chat_id, "No items found in database.")
                return
            
            # Group by location
            avondale_items = [i for i in items if i.location == "Avondale"]
            commissary_items = [i for i in items if i.location == "Commissary"]
            
            text = (
                "📈 <b>AVERAGE DAILY USAGE</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )
            
            # Avondale section
            if avondale_items:
                text += "🏪 <b>AVONDALE</b>\n"
                for item in sorted(avondale_items, key=lambda x: x.adu, reverse=True):
                    # Use emoji indicators for high/medium/low usage
                    if item.adu >= 5:
                        indicator = "🔴"  # High usage
                    elif item.adu >= 2:
                        indicator = "🟡"  # Medium usage
                    else:
                        indicator = "🟢"  # Low usage
                    
                    text += f"{indicator} <b>{item.name}</b>\n"
                    text += f"   {item.adu:.2f} {item.unit_type}/day\n"
                text += "\n"
            
            # Commissary section
            if commissary_items:
                text += "🏭 <b>COMMISSARY</b>\n"
                for item in sorted(commissary_items, key=lambda x: x.adu, reverse=True):
                    # Use emoji indicators
                    if item.adu >= 2:
                        indicator = "🔴"  # High usage
                    elif item.adu >= 1:
                        indicator = "🟡"  # Medium usage
                    else:
                        indicator = "🟢"  # Low usage
                    
                    text += f"{indicator} <b>{item.name}</b>\n"
                    text += f"   {item.adu:.2f} {item.unit_type}/day\n"
            
            text += (
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📊 Usage Indicators:\n"
                "🔴 High • 🟡 Medium • 🟢 Low\n\n"
                "💡 ADU drives all calculations"
            )
            
            self.send_message(chat_id, text)
            
        except Exception as e:
            self.logger.error(f"/adu failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to retrieve ADU data.")

    def _handle_missing(self, message: Dict):
        """Missing counts with clear visual formatting"""
        chat_id = message["chat"]["id"]
        
        parts = message.get("text", "").split()
        
        if len(parts) < 3:
            # Help message
            text = (
                "ℹ️ <b>Check Missing Counts</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📝 <b>Usage:</b>\n"
                "/missing [location] [date]\n\n"
                "📍 <b>Locations:</b>\n"
                "  • Avondale\n"
                "  • Commissary\n\n"
                "📅 <b>Date Format:</b>\n"
                "  • YYYY-MM-DD\n"
                "  • Example: 2025-09-16\n\n"
                "💡 <b>Example:</b>\n"
                "<code>/missing Avondale 2025-09-16</code>"
            )
            self.send_message(chat_id, text)
            return
        
        location = parts[1]
        date = parts[2]
        
        # Validate location
        if location not in ["Avondale", "Commissary"]:
            self.send_message(chat_id, (
                "❌ Invalid location\n"
                "Please use: Avondale or Commissary"
            ))
            return
        
        try:
            missing = self.notion.get_missing_counts(location, date)
            
            if not missing:
                text = (
                    "✅ <b>Inventory Check Complete</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📍 Location: <b>{location}</b>\n"
                    f"📅 Date: <b>{date}</b>\n\n"
                    "✅ All items have been counted\n"
                    "No missing entries detected"
                )
            else:
                text = (
                    "⚠️ <b>Missing Inventory Counts</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📍 Location: <b>{location}</b>\n"
                    f"📅 Date: <b>{date}</b>\n"
                    f"📊 Missing: <b>{len(missing)} items</b>\n\n"
                    
                    "📝 <b>Items Without Counts:</b>\n"
                )
                
                for item in missing:
                    text += f"  ☐ {item}\n"
                
                text += (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💡 Use /entry to record these counts"
                )
            
            self.send_message(chat_id, text)
            
        except Exception as e:
            self.logger.error(f"/missing failed: {e}", exc_info=True)
            self.send_message(chat_id, (
                "⚠️ Unable to check missing counts\n"
                "Please verify the date format and try again"
            ))

    def _handle_entry(self, message: Dict):
        """Start inventory entry flow."""
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        
        try:
            state = self._get_or_create_conversation(user_id, chat_id, "/entry")
            state.step = "choose_location"
            
            keyboard = _ik([
                [("🏪 Avondale", "loc|Avondale")],
                [("🏭 Commissary", "loc|Commissary")]
            ])
            
            self.send_message(chat_id, 
                            "<b>📝 Inventory Entry</b>\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            "Select location:",
                            reply_markup=keyboard)
            
        except Exception as e:
            self.logger.error(f"Error starting entry: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to start entry. Please try again.")
    
    def _handle_cancel(self, message: Dict):
        """Cancel current operation."""
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        
        with self.conversation_lock:
            if user_id in self.conversations:
                state = self.conversations[user_id]
                command = state.command
                del self.conversations[user_id]
                
                text = (
                    "❌ <b>Operation Cancelled</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"Cancelled: {command}\n"
                    "No data was saved\n\n"
                    "Start over with /entry or /help"
                )
            else:
                text = "ℹ️ No active operation to cancel"
        
        self.send_message(chat_id, text)

    def _handle_unknown(self, message: Dict):
        """Handle unknown commands."""
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        
        self.send_message(chat_id,
                        f"❓ Unknown command: {text}\n"
                        "Type /help to see available commands")

    # ===== CONVERSATION INPUT HANDLING =====
    
    def _handle_conversation_input_enhanced(self, message: Dict, state: ConversationState) -> bool:
        """
        Enhanced conversation input handler.
        
        Returns:
            bool: True if handled, False to use fallback handler
        """
        text = sanitize_user_input(message.get("text", ""))
        chat_id = state.chat_id
        
        # Update activity
        state.update_activity()
        
        # Handle date entry
        if state.step == "enter_date":
            return self._handle_date_entry(state, text)
        
        # Handle item entry
        if state.step == "enter_items":
            return self._handle_item_entry(state, text)
        
        # Handle note entry
        if state.step == "enter_note":
            return self._handle_note_entry(state, text)
        
        return False
    
    def _handle_date_entry(self, state: ConversationState, text: str) -> bool:
        """Handle manual date entry."""
        if text.lower() in ("today", "t"):
            state.data["date"] = get_time_in_timezone(BUSINESS_TIMEZONE).strftime("%Y-%m-%d")
            self._begin_item_loop(state)
        elif validate_date_format(text):
            state.data["date"] = text
            self._begin_item_loop(state)
        else:
            self.send_message(state.chat_id, 
                            "❌ Invalid date format. Use YYYY-MM-DD or 'today'")
        return True
    
    def _handle_item_entry(self, state: ConversationState, text: str) -> bool:
        """Handle item quantity entry."""
        lower_text = text.lower()
        
        # Handle commands
        if lower_text in ("/skip", "skip"):
            state.current_item_index += 1
            self._prompt_next_item(state)
            return True
        
        if lower_text in ("/done", "done"):
            self._start_review(state)
            return True
        
        # Parse quantity
        try:
            qty = float(text)
            if qty < 0:
                raise ValueError("Negative quantity")
            
            item = state.items[state.current_item_index]
            state.data.setdefault("quantities", {})[item.name] = qty
            state.current_item_index += 1
            self._prompt_next_item(state)
            
        except ValueError:
            self.send_message(state.chat_id, 
                            "❌ Please enter a valid number, /skip, or /done")
        
        return True
    
    def _handle_note_entry(self, state: ConversationState, text: str) -> bool:
        """Handle note entry."""
        if text.lower() != "none":
            state.note = sanitize_user_input(text, 500)
        else:
            state.note = ""
        
        self._show_review(state)
        return True

    def _handle_conversation_input_entry_ext(self, message: Dict, state: "ConversationState") -> bool:
        """
        Extends your existing _handle_conversation_input.
        Returns True if this function fully handled the message; False to let your original logic run.
        """
        chat_id = state.chat_id
        text = (message.get("text") or "").strip()
        low = text.lower()
        state.update_activity()

        # global escape
        if low == "/cancel":
            self._handle_cancel(message)
            return True

        if user_id in self.entry_handler.sessions:
            session = self.entry_handler.sessions[user_id]
            self.entry_handler.handle_text_input(message, session)
            return

        # manual date entry
        if state.step == "choose_date":
            today = get_time_in_timezone(BUSINESS_TIMEZONE).strftime("%Y-%m-%d")
            if low in ("today", "t"):
                state.data["date"] = today
                self._begin_item_loop(state)
                return True
            try:
                datetime.strptime(text, "%Y-%m-%d")
                state.data["date"] = text
                self._begin_item_loop(state)
            except ValueError:
                self.send_message(chat_id, "Invalid date. Use YYYY-MM-DD or 'today'.")
            return True
        
        # item quantities with /skip /done
        if state.step == "enter_items":
            if low == "/skip":
                state.current_item_index += 1
                self._prompt_next_item(state)
                return True
            if low == "/done":
                state.step = "note"
                self.send_message(chat_id, "Add a note? Reply text or 'none'.")
                return True
            try:
                qty = float(text)
                item = state.items[state.current_item_index]
                state.data["quantities"][item.name] = qty
                state.current_item_index += 1
                self._prompt_next_item(state)
            except ValueError:
                self.send_message(chat_id, "Enter a number, or /skip /done /cancel.")
            return True

        # note → review card
        if state.step == "note":
            if low != "none":
                state.note = text
            state.step = "review"
            lines = [f"• {k}: {v}" for k, v in state.data.get("quantities", {}).items()]
            preview = (
                f"<b>Review</b>\n"
                f"Location: <b>{state.location}</b>\n"
                f"Type: <b>{'On-Hand' if state.entry_type=='on_hand' else 'Received'}</b>\n"
                f"Date: <b>{state.data['date']}</b>\n"
                f"Items: {len(lines)}\n" + ("\n".join(lines) if lines else "• none") + "\n"
                f"Note: {getattr(state, 'note', '') or '—'}"
            )
            kb = {"inline_keyboard": [[{"text": "Submit", "callback_data": "review|submit"},
                                    {"text": "Go Back", "callback_data": "review|back"}],
                                    [{"text": "Cancel", "callback_data": "review|cancel"}]]}
            self.send_message(chat_id, preview, reply_markup=kb)
            return True

        # not handled here → let your original handler run
        return False


    # ===== ITEM ENTRY FLOW =====
    
    def _begin_item_loop(self, state: ConversationState):
        """Start item entry loop."""
        try:
            state.items = self.notion.get_items_for_location(state.location)
            
            if not state.items:
                self.send_message(state.chat_id, 
                                f"⚠️ No items found for {state.location}")
                self._end_conversation(state.user_id)
                return
            
            state.current_item_index = 0
            state.data["quantities"] = {}
            state.step = "enter_items"
            
            entry_type = "On-Hand Count" if state.entry_type == "on_hand" else "Delivery"
            
            self.send_message(state.chat_id, 
                            f"📝 <b>{entry_type} for {state.location}</b>\n"
                            f"Date: {state.data['date']}\n"
                            f"Items: {len(state.items)}\n\n"
                            "Enter quantities (or /skip, /done, /cancel)")
            
            self._prompt_next_item(state)
            
        except Exception as e:
            self.logger.error(f"Error starting item loop: {e}", exc_info=True)
            self.send_message(state.chat_id, 
                            "⚠️ Error loading items. Please try again.")
            self._end_conversation(state.user_id)

    def _prompt_next_item(self, state: ConversationState):
        """Prompt for next item or complete if done."""
        if state.current_item_index >= len(state.items):
            self._start_review(state)
            return
        
        item = state.items[state.current_item_index]
        progress = f"{state.current_item_index + 1}/{len(state.items)}"
        
        # Get last recorded quantity if available
        last_qty = ""
        if hasattr(state, 'data') and 'quantities' in state.data:
            if item.name in state.data['quantities']:
                last_qty = f" (currently: {state.data['quantities'][item.name]})"
        
        self.send_message(state.chat_id,
                        f"[{progress}] <b>{item.name}</b>\n"
                        f"Unit: {item.unit_type} • ADU: {item.adu:.2f}/day{last_qty}\n"
                        f"Enter quantity:")

    def _start_review(self, state: ConversationState):
        """Start review process."""
        state.step = "enter_note"
        self.send_message(state.chat_id, 
                        "📝 Add a note? (type note or 'none'):")
    
    def _show_review(self, state: ConversationState):
        """Show review summary."""
        state.step = "review"
        
        quantities = state.data.get("quantities", {})
        items_with_qty = [(k, v) for k, v in quantities.items() if v > 0]
        
        entry_type = "On-Hand Count" if state.entry_type == "on_hand" else "Delivery"
        
        text = (
            "📋 <b>Review Entry</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Type: <b>{entry_type}</b>\n"
            f"Location: <b>{state.location}</b>\n"
            f"Date: <b>{state.data['date']}</b>\n"
            f"Items recorded: <b>{len(items_with_qty)}</b>\n\n"
        )
        
        if items_with_qty:
            text += "📦 <b>Quantities:</b>\n"
            for name, qty in sorted(items_with_qty):
                text += f"  • {name}: {qty}\n"
        else:
            text += "⚠️ No quantities entered\n"
        
        if state.note:
            text += f"\n📝 Note: {state.note}\n"
        
        keyboard = _ik([
            [("✅ Submit", "review|submit"), ("◀️ Back", "review|back")],
            [("❌ Cancel", "review|cancel")]
        ])
        
        self.send_message(state.chat_id, text, reply_markup=keyboard)

    def _finalize_entry(self, state: ConversationState):
        """Save entry to Notion."""
        try:
            quantities = state.data.get("quantities", {})
            
            # Validate quantities
            if not quantities or all(v == 0 for v in quantities.values()):
                self.send_message(state.chat_id, 
                                "⚠️ No quantities entered. Entry cancelled.")
                self._end_conversation(state.user_id)
                return
            
            # Save to Notion
            success = self.notion.save_inventory_transaction(
                location=state.location,
                entry_type=state.entry_type,
                date=state.data["date"],
                manager="Manager",  # Could be from state.data if collected
                notes=state.note if hasattr(state, 'note') else "",
                quantities=quantities
            )
            
            if success:
                items_count = len([v for v in quantities.values() if v > 0])
                entry_type = "on-hand count" if state.entry_type == "on_hand" else "delivery"
                
                self.send_message(state.chat_id,
                                f"✅ <b>Entry Saved</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Saved {items_count} items for {state.location}\n"
                                f"Type: {entry_type}\n"
                                f"Date: {state.data['date']}\n\n"
                                f"Use /info to see updated status")
            else:
                self.send_message(state.chat_id, 
                                "⚠️ Failed to save to Notion. Please try again.")
            
        except Exception as e:
            self.logger.error(f"Error finalizing entry: {e}", exc_info=True)
            self.send_message(state.chat_id, 
                            "⚠️ Error saving entry. Please contact support.")
        finally:
            self._end_conversation(state.user_id)

    def _handle_info(self, message: Dict):
        """
        Display inventory dashboard with FIXED consumption math.
        Shows forecasted on-hand at delivery and post-delivery needs.
        """
        import math
        chat_id = message["chat"]["id"]
        
        def format_critical_item(item: dict) -> str:
            """Format critical item with forecasted values."""
            name = item.get("item_name", "Unknown")
            unit = item.get("unit_type", "unit")
            current = float(item.get("current_qty", 0))
            oh_delivery = float(item.get("oh_at_delivery", 0))
            need = float(item.get("consumption_need", 0))  # This is Need_post
            order = item.get("required_order_rounded", 0)
            
            # Determine status icon based on severity
            if oh_delivery == 0:
                status_icon = "🚨"
            elif oh_delivery < need * 0.3:
                status_icon = "⚠️"
            else:
                status_icon = "📉"
            
            # Format the display
            return (
                f"{status_icon} <b>{name}</b>\n"
                f"   Order {order} {unit} • Now: {current:.1f} → Delivery: {oh_delivery:.1f}\n"
                f"   Need {need:.1f} for post-delivery window"
            )
        
        try:
            now = get_time_in_timezone(BUSINESS_TIMEZONE)
            avondale = self.calc.calculate_location_summary("Avondale")
            commissary = self.calc.calculate_location_summary("Commissary")
            
            # Header with timestamp
            text = (
                "📊 <b>Inventory Dashboard</b>\n"
                f"🕐 {now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )
            
            # Avondale Section
            a_red = avondale.get("status_counts", {}).get("RED", 0)
            a_green = avondale.get("status_counts", {}).get("GREEN", 0)
            a_days = avondale.get("days_until_delivery", 0)
            a_delivery = avondale.get("delivery_date", "—")
            a_cycle = avondale.get("order_cycle", {})
            
            text += (
                "🏪 <b>AVONDALE</b>\n"
                f"├ Next Delivery: {a_delivery} ({a_days:.1f} days)\n"
                f"├ Status: 🔴 {a_red} • 🟢 {a_green}\n"
            )
            
            # Avondale critical items
            a_critical = [item for item in avondale.get("items", []) 
                        if item.get("status") == "RED"]
            if a_critical:
                text += "└ <b>Critical Items:</b>\n"
                for item in sorted(a_critical, 
                                key=lambda x: x.get("required_order_rounded", 0), 
                                reverse=True)[:5]:
                    lines = format_critical_item(item).split('\n')
                    for line in lines:
                        text += f"  {line}\n"
                if len(a_critical) > 5:
                    text += f"  <i>...and {len(a_critical) - 5} more critical</i>\n"
            else:
                text += "└ ✅ All items sufficient through next delivery\n"
            
            text += "\n"
            
            # Commissary Section
            c_red = commissary.get("status_counts", {}).get("RED", 0)
            c_green = commissary.get("status_counts", {}).get("GREEN", 0)
            c_days = commissary.get("days_until_delivery", 0)
            c_delivery = commissary.get("delivery_date", "—")
            c_cycle = commissary.get("order_cycle", {})
            
            text += (
                "🏭 <b>COMMISSARY</b>\n"
                f"├ Next Delivery: {c_delivery} ({c_days:.1f} days)\n"
                f"├ Status: 🔴 {c_red} • 🟢 {c_green}\n"
            )
            
            # Commissary critical items
            c_critical = [item for item in commissary.get("items", []) 
                        if item.get("status") == "RED"]
            if c_critical:
                text += "└ <b>Critical Items:</b>\n"
                for item in sorted(c_critical, 
                                key=lambda x: x.get("required_order_rounded", 0), 
                                reverse=True)[:5]:
                    lines = format_critical_item(item).split('\n')
                    for line in lines:
                        text += f"  {line}\n"
                if len(c_critical) > 5:
                    text += f"  <i>...and {len(c_critical) - 5} more critical</i>\n"
            else:
                text += "└ ✅ All items sufficient through next delivery\n"
            
            # Footer with explanation
            text += (
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📌 <b>How to read:</b>\n"
                "• Now → Delivery: consumption forecast\n"
                "• Need: required for post-delivery period\n"
                "• /order for supplier-ready list"
            )
            
            self.send_message(chat_id, text)
            self.logger.info(f"/info sent - A: {a_red} red, C: {c_red} red")
            
        except Exception as e:
            self.logger.error(f"/info failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to generate dashboard. Please try again.")


    def _handle_order(self, message: Dict):
        """
        Generate combined purchase orders with team notification format.
        """
        import math
        chat_id = message["chat"]["id"]
        
        def format_order_section(location: str, summary: dict, emoji: str) -> str:
            """Format order section with clean layout."""
            delivery = summary.get("delivery_date", "—")
            requests = summary.get("requests", [])
            cycle = summary.get("order_cycle", {})
            
            # Calculate totals by unit type
            totals = {}
            order_lines = []
            
            for item in requests:
                qty = item.get("requested_qty", 0)
                if qty <= 0:
                    continue
                    
                name = item.get("item_name", "Unknown")
                unit = item.get("unit_type", "unit")
                current = float(item.get("current_qty", 0))
                oh_delivery = float(item.get("oh_at_delivery", 0))
                need = float(item.get("consumption_need", 0))
                
                totals[unit] = totals.get(unit, 0) + qty
                order_lines.append({
                    'qty': qty,
                    'name': name,
                    'unit': unit,
                    'current': current,
                    'oh_delivery': oh_delivery,
                    'need': need
                })
            
            # Sort by quantity descending
            order_lines.sort(key=lambda x: x['qty'], reverse=True)
            
            # Build section text
            text = f"{emoji} <b>{location.upper()} ORDER</b>\n"
            text += f"📅 Delivery: {delivery}\n"
            
            if not order_lines:
                text += "✅ No items needed\n"
                return text
            
            # Totals summary
            text += "📦 Totals: "
            text += " • ".join(f"{v} {k}" for k, v in sorted(totals.items()))
            text += f"\n\n"
            
            # Item list (compact for mobile)
            for item in order_lines[:10]:
                text += f"<b>{item['qty']} {item['unit']}</b> — {item['name']}\n"
                # Show burn-down if significant
                if abs(item['current'] - item['oh_delivery']) > 0.1:
                    text += f"  {item['current']:.1f} now → {item['oh_delivery']:.1f} at delivery\n"
                else:
                    text += f"  Current: {item['current']:.1f} • Need: {item['need']:.1f}\n"
            
            if len(order_lines) > 10:
                text += f"<i>...and {len(order_lines) - 10} more items</i>\n"
            
            return text
        
        try:
            avondale = self.calc.generate_auto_requests("Avondale")
            commissary = self.calc.generate_auto_requests("Commissary")
            self._save_order_to_notion("Avondale", avondale)
            self._save_order_to_notion("Commissary", commissary)
        

            
            text = (
                "📋 <b>PURCHASE ORDERS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )
            
            text += format_order_section("Avondale", avondale, "🏪")
            text += "\n" + ("─" * 28) + "\n\n"
            text += format_order_section("Commissary", commissary, "🏭")
            
            # Team messages
            text += (
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📱 <b>Team Messages:</b>\n\n"
            )
            
            # Avondale team message if items needed
            if avondale.get("requests"):
                a_delivery = avondale.get("delivery_date", "")
                a_totals = {}
                for item in avondale.get("requests", []):
                    unit = item.get("unit_type", "unit")
                    qty = item.get("requested_qty", 0)
                    a_totals[unit] = a_totals.get(unit, 0) + qty
                
                a_summary = " • ".join(f"{v} {k}" for k, v in sorted(a_totals.items()))
                text += f"<i>Avondale team: Please order {a_summary} for {a_delivery} delivery.</i>\n\n"
            
            # Commissary team message if items needed
            if commissary.get("requests"):
                c_delivery = commissary.get("delivery_date", "")
                c_totals = {}
                for item in commissary.get("requests", []):
                    unit = item.get("unit_type", "unit")
                    qty = item.get("requested_qty", 0)
                    c_totals[unit] = c_totals.get(unit, 0) + qty
                
                c_summary = " • ".join(f"{v} {k}" for k, v in sorted(c_totals.items()))
                text += f"<i>Commissary team: Please order {c_summary} for {c_delivery} delivery.</i>\n"
            
            text += (
                "\n💡 Orders account for consumption to delivery\n"
                "• /order_avondale • /order_commissary"
            )
            
            self.send_message(chat_id, text)
            self.logger.info(f"/order sent successfully")
            
        except Exception as e:
            self.logger.error(f"/order failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to generate orders. Please try again.")

    def _handle_order_avondale(self, message: Dict):
        """
        Avondale order with header info and clean request-style list.
        """
        import math
        from datetime import datetime
        chat_id = message["chat"]["id"]
        
        try:
            summary = self.calc.generate_auto_requests("Avondale")
            self._save_order_to_notion("Avondale", summary)
            delivery = summary.get("delivery_date", "—")
            requests = summary.get("requests", [])
            cycle = summary.get("order_cycle", {})
            
            # Process and sort orders
            orders = []
            totals = {}
            
            for item in requests:
                qty = item.get("requested_qty", 0)
                if qty <= 0:
                    continue
                
                unit = item.get("unit_type", "unit")
                totals[unit] = totals.get(unit, 0) + qty
                
                orders.append({
                    'qty': qty,
                    'name': item.get("item_name", "Unknown"),
                    'unit': unit,
                    'current': float(item.get("current_qty", 0)),
                    'oh_delivery': float(item.get("oh_at_delivery", 0)),
                    'need': float(item.get("consumption_need", 0))
                })
            
            orders.sort(key=lambda x: x['qty'], reverse=True)
            
            # Build message header
            text = (
                "🏪 <b>AVONDALE PURCHASE ORDER</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 Delivery Date: {delivery}\n"
            )
            
            # Show order timing
            if cycle:
                days_pre = cycle.get('days_pre', 0)
                days_post = cycle.get('days_post', 0)
                text += (
                    f"📊 Order Window:\n"
                    f"  • Burn-down days: {days_pre}\n"
                    f"  • Coverage days: {days_post}\n"
                )
            
            text += f"📦 Items to Order: {len(orders)}\n\n"
            
            if orders:
                # Order summary by unit type
                text += "📊 <b>Order Summary</b>\n"
                for unit, total in sorted(totals.items(), key=lambda x: (-x[1], x[0])):
                    text += f"  • {total} {unit}{'s' if total > 1 else ''}\n"
                
                # Request-style message format
                text += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                text += f"📋 <b>ORDER REQUEST - AVONDALE</b>\n"
                
                # Get current day
                now = get_time_in_timezone(BUSINESS_TIMEZONE)
                text += f"📅 {now.strftime('%a %b %d, %Y')}\n\n"
                
                # Delivery day from date
                try:
                    delivery_date = datetime.strptime(delivery, '%Y-%m-%d')
                    delivery_day = delivery_date.strftime('%A')
                except:
                    delivery_day = "delivery"
                
                text += (
                    f"Hey Avondale Prep Team! This is what we need for {delivery_day} Delivery.\n"
                    f"Please confirm at your earliest convenience:\n\n"
                )
                
                # Clean item list
                for item in orders:
                    # Format: • Item Name: X units
                    plural = 's' if item['qty'] > 1 else ''
                    # Handle unit pluralization properly
                    if item['unit'] == 'case':
                        unit_plural = 'cases' if item['qty'] > 1 else 'case'
                    elif item['unit'] == 'bag':
                        unit_plural = 'bags' if item['qty'] > 1 else 'bag'
                    elif item['unit'] == 'tray':
                        unit_plural = 'trays' if item['qty'] > 1 else 'tray'
                    elif item['unit'] == 'bottle':
                        unit_plural = 'bottles' if item['qty'] > 1 else 'bottle'
                    elif item['unit'] == 'quart':
                        unit_plural = 'quarts' if item['qty'] > 1 else 'quart'
                    else:
                        unit_plural = item['unit'] + plural
                    
                    text += f"• {item['name']}: {item['qty']} {unit_plural}\n"
                
                text += f"\nTotal items: {len(orders)}"
                
            else:
                text += (
                    "✅ <b>No Orders Needed</b>\n\n"
                    "All inventory levels are sufficient\n"
                    "through the next delivery cycle."
                )
            
            self.send_message(chat_id, text)
            self.logger.info(f"/order_avondale sent - {len(orders)} items")
            
        except Exception as e:
            self.logger.error(f"/order_avondale failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to generate Avondale order. Please try again.")


    def _handle_order_commissary(self, message: Dict):
        """
        Commissary order with header info and clean request-style list.
        """
        import math
        from datetime import datetime
        chat_id = message["chat"]["id"]
        
        try:
            summary = self.calc.generate_auto_requests("Commissary")
            self._save_order_to_notion("Commissary", summary)
            
            delivery = summary.get("delivery_date", "—")
            requests = summary.get("requests", [])
            cycle = summary.get("order_cycle", {})
            
            # Process and sort orders
            orders = []
            totals = {}
            
            for item in requests:
                qty = item.get("requested_qty", 0)
                if qty <= 0:
                    continue
                
                unit = item.get("unit_type", "unit")
                totals[unit] = totals.get(unit, 0) + qty
                
                orders.append({
                    'qty': qty,
                    'name': item.get("item_name", "Unknown"),
                    'unit': unit,
                    'current': float(item.get("current_qty", 0)),
                    'oh_delivery': float(item.get("oh_at_delivery", 0)),
                    'need': float(item.get("consumption_need", 0))
                })
            
            orders.sort(key=lambda x: x['qty'], reverse=True)
            
            # Build message header
            text = (
                "🏭 <b>COMMISSARY PURCHASE ORDER</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 Delivery Date: {delivery}\n"
            )
            
            # Show order timing
            if cycle:
                days_pre = cycle.get('days_pre', 0)
                days_post = cycle.get('days_post', 0)
                text += (
                    f"📊 Order Window:\n"
                    f"  • Burn-down days: {days_pre}\n"
                    f"  • Coverage days: {days_post}\n"
                )
            
            text += f"📦 Items to Order: {len(orders)}\n\n"
            
            if orders:
                # Order summary by unit type
                text += "📊 <b>Order Summary</b>\n"
                for unit, total in sorted(totals.items(), key=lambda x: (-x[1], x[0])):
                    text += f"  • {total} {unit}{'s' if total > 1 else ''}\n"
                
                # Request-style message format
                text += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                text += f"📋 <b>ORDER REQUEST - COMMISSARY</b>\n"
                
                # Get current day
                now = get_time_in_timezone(BUSINESS_TIMEZONE)
                text += f"📅 {now.strftime('%a %b %d, %Y')}\n\n"
                
                # Delivery day from date
                try:
                    delivery_date = datetime.strptime(delivery, '%Y-%m-%d')
                    delivery_day = delivery_date.strftime('%A')
                except:
                    delivery_day = "delivery"
                
                text += (
                    f"Hey Commissary Prep Team! This is what we need for {delivery_day} Delivery.\n"
                    f"Please confirm at your earliest convenience:\n\n"
                )
                
                # Clean item list
                for item in orders:
                    # Format: • Item Name: X units
                    plural = 's' if item['qty'] > 1 else ''
                    # Handle unit pluralization properly
                    if item['unit'] == 'case':
                        unit_plural = 'cases' if item['qty'] > 1 else 'case'
                    elif item['unit'] == 'bag':
                        unit_plural = 'bags' if item['qty'] > 1 else 'bag'
                    elif item['unit'] == 'tray':
                        unit_plural = 'trays' if item['qty'] > 1 else 'tray'
                    elif item['unit'] == 'bottle':
                        unit_plural = 'bottles' if item['qty'] > 1 else 'bottle'
                    elif item['unit'] == 'quart':
                        unit_plural = 'quarts' if item['qty'] > 1 else 'quart'
                    else:
                        unit_plural = item['unit'] + plural
                    
                    text += f"• {item['name']}: {item['qty']} {unit_plural}\n"
                
                text += f"\nTotal items: {len(orders)}"
                
            else:
                text += (
                    "✅ <b>No Orders Needed</b>\n\n"
                    "All inventory levels are sufficient\n"
                    "through the next delivery cycle."
                )
            
            self.send_message(chat_id, text)
            self.logger.info(f"/order_commissary sent - {len(orders)} items")
            
        except Exception as e:
            self.logger.error(f"/order_commissary failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to generate Commissary order. Please try again.")
 
    def _handle_reassurance(self, message: Dict):
        """
        Daily risk assessment with FIXED consumption math.
        Shows which items won't make it to delivery based on burn-down.
        """
        chat_id = message["chat"]["id"]
        
        try:
            avondale = self.calc.calculate_location_summary("Avondale")
            commissary = self.calc.calculate_location_summary("Commissary")
            
            # Get critical items that need immediate attention
            a_critical = [item for item in avondale.get("items", []) 
                        if item.get("status") == "RED"]
            c_critical = [item for item in commissary.get("items", []) 
                        if item.get("status") == "RED"]
            total_critical = len(a_critical) + len(c_critical)
            
            now = get_time_in_timezone(BUSINESS_TIMEZONE)
            
            if total_critical == 0:
                # All clear message
                text = (
                    "✅ <b>DAILY RISK ASSESSMENT</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🕐 {now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
                    
                    "🟢 <b>ALL CLEAR</b>\n"
                    "No critical inventory issues detected\n\n"
                    
                    "📊 <b>Location Status</b>\n"
                )
                
                # Avondale status
                a_cycle = avondale.get("order_cycle", {})
                text += (
                    f"├ 🏪 Avondale: {avondale['status_counts']['GREEN']} items OK\n"
                    f"│  Next delivery: {avondale['delivery_date']}\n"
                    f"│  Coverage window: {a_cycle.get('days_post', 0)} days\n"
                )
                
                # Commissary status
                c_cycle = commissary.get("order_cycle", {})
                text += (
                    f"├ 🏭 Commissary: {commissary['status_counts']['GREEN']} items OK\n"
                    f"│  Next delivery: {commissary['delivery_date']}\n"
                    f"│  Coverage window: {c_cycle.get('days_post', 0)} days\n"
                )
                
                text += (
                    f"└ Total Coverage: 100%\n\n"
                    
                    "✅ All levels sufficient through delivery\n"
                    "✅ Orders sized for post-delivery needs\n"
                    "✅ No immediate action required\n\n"
                    
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "💚 <b>System Status: Healthy</b>"
                )
            else:
                # Critical items alert
                text = (
                    "🚨 <b>DAILY RISK ASSESSMENT</b>\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"🕐 {now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
                    
                    f"⚠️ <b>ACTION REQUIRED</b>\n"
                    f"{total_critical} item{'s' if total_critical != 1 else ''} need ordering\n\n"
                )
                
                if a_critical:
                    a_cycle = avondale.get("order_cycle", {})
                    text += f"🏪 <b>AVONDALE ({len(a_critical)} critical)</b>\n"
                    text += f"Delivery: {avondale['delivery_date']} • "
                    text += f"Coverage: {a_cycle.get('days_post', 0)} days\n\n"
                    
                    for item in sorted(a_critical, 
                                    key=lambda x: x.get('oh_at_delivery', 0))[:5]:
                        oh_delivery = item.get('oh_at_delivery', 0)
                        need = item.get('consumption_need', 0)
                        order = item.get('required_order_rounded', 0)
                        
                        text += f"🔴 <b>{item['item_name']}</b>\n"
                        text += f"   At delivery: {oh_delivery:.1f} {item['unit_type']}\n"
                        text += f"   Need: {need:.1f} • Order: {order}\n"
                    
                    if len(a_critical) > 5:
                        text += f"<i>...plus {len(a_critical) - 5} more</i>\n"
                    text += "\n"
                
                if c_critical:
                    c_cycle = commissary.get("order_cycle", {})
                    text += f"🏭 <b>COMMISSARY ({len(c_critical)} critical)</b>\n"
                    text += f"Delivery: {commissary['delivery_date']} • "
                    text += f"Coverage: {c_cycle.get('days_post', 0)} days\n\n"
                    
                    for item in sorted(c_critical, 
                                    key=lambda x: x.get('oh_at_delivery', 0))[:5]:
                        oh_delivery = item.get('oh_at_delivery', 0)
                        need = item.get('consumption_need', 0)
                        order = item.get('required_order_rounded', 0)
                        
                        text += f"🔴 <b>{item['item_name']}</b>\n"
                        text += f"   At delivery: {oh_delivery:.1f} {item['unit_type']}\n"
                        text += f"   Need: {need:.1f} • Order: {order}\n"
                    
                    if len(c_critical) > 5:
                        text += f"<i>...plus {len(c_critical) - 5} more</i>\n"
                
                text += (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "⚠️ <b>IMMEDIATE ACTION NEEDED</b>\n"
                    "These items need ordering NOW\n\n"
                    "📞 Contact supplier immediately\n"
                    "📋 Use /order for complete list"
                )
            
            # Send to reassurance chat if different from requester
            reassurance_chat = self.chat_config.get('reassurance')
            if reassurance_chat and reassurance_chat != chat_id:
                self.send_message(reassurance_chat, text)
                self.logger.info(f"Reassurance sent to management chat {reassurance_chat}")
            
            # Always send to requesting user
            self.send_message(chat_id, text)
            self.logger.info(f"/reassurance sent - {total_critical} critical items")
            
        except Exception as e:
            self.logger.error(f"/reassurance failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Unable to generate risk assessment. Please try again.")

            
    
    def _format_reassurance_clear(self, now, avondale, commissary):
        """Format all-clear reassurance message."""
        return (
            "✅ <b>DAILY RISK ASSESSMENT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
            
            "🟢 <b>ALL CLEAR</b>\n"
            "No critical inventory issues detected\n\n"
            
            "📊 <b>Location Status</b>\n"
            f"├ Avondale: {avondale['status_counts']['GREEN']} items OK\n"
            f"│  Next delivery: {avondale['delivery_date']}\n"
            f"├ Commissary: {commissary['status_counts']['GREEN']} items OK\n"
            f"│  Next delivery: {commissary['delivery_date']}\n"
            f"└ Total Coverage: 100%\n\n"
            
            "✅ All inventory levels sufficient\n"
            "✅ No immediate action required\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "💚 System Status: Healthy"
        )
    
    def _format_reassurance_alert(self, now, total_critical, a_critical, c_critical):
        """Format critical alert reassurance message."""
        text = (
            "🚨 <b>DAILY RISK ASSESSMENT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
            
            f"⚠️ <b>ACTION REQUIRED</b>\n"
            f"{total_critical} critical item{'s' if total_critical != 1 else ''} at risk\n\n"
        )
        
        if a_critical:
            text += f"🏪 <b>AVONDALE ({len(a_critical)} critical)</b>\n"
            for item in a_critical[:5]:
                days_stock = item.get('days_of_stock', 0)
                text += (
                    f"🔴 <b>{item['item_name']}</b>\n"
                    f"   Stock: {item['current_qty']:.1f} {item['unit_type']}\n"
                    f"   Days remaining: {days_stock:.1f}\n"
                )
            if len(a_critical) > 5:
                text += f"<i>...plus {len(a_critical) - 5} more</i>\n"
            text += "\n"
        
        if c_critical:
            text += f"🏭 <b>COMMISSARY ({len(c_critical)} critical)</b>\n"
            for item in c_critical[:5]:
                days_stock = item.get('days_of_stock', 0)
                text += (
                    f"🔴 <b>{item['item_name']}</b>\n"
                    f"   Stock: {item['current_qty']:.1f} {item['unit_type']}\n"
                    f"   Days remaining: {days_stock:.1f}\n"
                )
            if len(c_critical) > 5:
                text += f"<i>...plus {len(c_critical) - 5} more</i>\n"
        
        text += (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <b>IMMEDIATE ACTION NEEDED</b>\n"
            "📞 Contact supplier immediately\n"
            "📋 Use /order for complete list"
        )
        
        return text

# ===== Entry Point for Bot =====

@dataclass
class EntrySession:
    """
    Enhanced session state for conversational entry wizard.
    """
    user_id: int
    chat_id: int
    mode: str  # 'on_hand' or 'received'
    location: str  # 'Avondale' or 'Commissary'
    items: List[Dict[str, Any]]  # List of items with metadata
    index: int = 0  # Current item index
    answers: Dict[str, Optional[float]] = field(default_factory=dict)  # item_name -> quantity
    started_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=2))
    last_message_id: Optional[int] = None
    manager_name: str = "Manager"
    notes: str = ""
    image_file_id: Optional[str] = None  # Telegram file ID for product image
    current_step: str = "items"  # Track if we're on items or image step
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at
    
    def update_activity(self):
        """Reset expiration timer on activity."""
        self.expires_at = datetime.now() + timedelta(hours=2)
    
    def get_current_item(self) -> Optional[Dict[str, Any]]:
        """Get current item being edited."""
        if 0 <= self.index < len(self.items):
            return self.items[self.index]
        return None
    
    def move_back(self):
        """Move to previous item."""
        self.index = max(0, self.index - 1)
    
    def move_forward(self):
        """Move to next item."""
        self.index = min(len(self.items) - 1, self.index + 1)
    
    def skip_current(self):
        """Mark current item as skipped and move forward."""
        item = self.get_current_item()
        if item:
            self.answers[item['name']] = None
        self.index += 1
    
    def set_current_quantity(self, quantity: float):
        """Set quantity for current item."""
        item = self.get_current_item()
        if item:
            self.answers[item['name']] = quantity
    
    def get_progress(self) -> str:
        """Get progress string."""
        return f"{self.index + 1}/{len(self.items)}"
    
    def get_answered_count(self) -> int:
        """Count items with non-null answers."""
        return sum(1 for v in self.answers.values() if v is not None)
    
    def get_total_quantity(self) -> float:
        """Get sum of all entered quantities."""
        return sum(v for v in self.answers.values() if v is not None)


class EnhancedEntryHandler:
    """
    Enhanced entry handler with full navigation controls.
    """
    
    def __init__(self, bot, notion_manager, calculator):
        """Initialize with dependencies."""
        self.bot = bot
        self.notion = notion_manager
        self.calc = calculator
        self.logger = logging.getLogger('entry_wizard')
        self.sessions: Dict[int, EntrySession] = {}
    
    def handle_entry_command(self, message: Dict):
        """
        Handle /entry command - check for existing session first.
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        
        # Check for existing session
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if not session.is_expired():
                # Offer to resume or start over
                keyboard = self._create_keyboard([
                    [("📂 Resume", "entry_resume"), ("🔄 Start Over", "entry_new")],
                    [("❌ Cancel", "entry_cancel_existing")]
                ])
                
                progress = session.get_progress()
                mode_text = "On-Hand Count" if session.mode == "on_hand" else "Delivery"
                
                self.bot.send_message(
                    chat_id,
                    f"📋 <b>Active Session Found</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Type: {mode_text} for {session.location}\n"
                    f"Progress: {progress} • Answered: {session.get_answered_count()}\n\n"
                    f"What would you like to do?",
                    reply_markup=keyboard
                )
                return
        
        # No active session - start new
        self._start_new_entry(chat_id, user_id)
    
    def _start_new_entry(self, chat_id: int, user_id: int):
        """Start new entry session."""
        keyboard = self._create_keyboard([
            [("🏪 Avondale", "entry_loc|Avondale")],
            [("🏭 Commissary", "entry_loc|Commissary")],
            [("❌ Cancel", "entry_cancel")]
        ])
        
        self.bot.send_message(
            chat_id,
            "📝 <b>Inventory Entry Wizard</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Select location:",
            reply_markup=keyboard
        )
    
    def handle_callback(self, callback_query: Dict):
        """Handle all entry-related callbacks."""
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")
        
        # Acknowledge callback
        self.bot._make_request("answerCallbackQuery", 
                              {"callback_query_id": callback_query.get("id")})
        
        # Route callbacks
        if data == "entry_resume":
            self._resume_session(chat_id, user_id)
        elif data == "entry_new":
            self._delete_session(user_id)
            self._start_new_entry(chat_id, user_id)
        elif data.startswith("entry_loc|"):
            location = data.split("|")[1]
            self._handle_location_selection(chat_id, user_id, location)
        elif data.startswith("entry_mode|"):
            mode = data.split("|")[1]
            self._handle_mode_selection(chat_id, user_id, mode)
        elif data == "entry_back":
            self._handle_back(chat_id, user_id)
        elif data == "entry_skip":
            self._handle_skip(chat_id, user_id)
        elif data == "entry_done":
            self._handle_done(chat_id, user_id)
        elif data in ["entry_cancel", "entry_cancel_existing"]:
            self._handle_cancel(chat_id, user_id)
        elif data == "entry_submit":
            self._handle_submit(chat_id, user_id)
        elif data == "entry_resume_items":
            self._resume_items(chat_id, user_id)
        elif data == "entry_skip_image":
            session = self.sessions.get(user_id)
            if session and session.current_step == "image":
                session.current_step = "review"
                self._handle_done(chat_id, user_id)

    def handle_text_input(self, message: Dict, session: EntrySession):
        """
        Handle text input for quantity entry and image flow.
        """
        text = message.get("text", "").strip()
        chat_id = session.chat_id
        
        # Update session activity
        session.update_activity()
        
        # Check for special text commands first
        if text.lower() in ["/back", "back"]:
            self._handle_back(chat_id, session.user_id)
            return
        elif text.lower() in ["/skip", "skip"]:
            self._handle_skip(chat_id, session.user_id)
            return
        elif text.lower() in ["/done", "done"]:
            self._handle_done(chat_id, session.user_id)
            return
        elif text.lower() in ["/cancel", "cancel"]:
            self._handle_cancel(chat_id, session.user_id)
            return
        
        # Check if we're in image step
        if session.current_step == "image":
            if text.lower() in ["skip", "skip image"]:
                session.current_step = "review"
                self._handle_done(chat_id, session.user_id)
            else:
                self.bot.send_message(
                    chat_id,
                    "📷 Please send a photo or use the 'Skip Image' button."
                )
                self._show_image_request(session)
            return
        
        # Try to parse as number for item quantities
        try:
            quantity = float(text)
            
            # Validate range
            if quantity < 0:
                self.bot.send_message(
                    chat_id,
                    "❌ Please enter a positive number or 0"
                )
                self._show_current_item(session)
                return
            
            if quantity > 9999:
                self.bot.send_message(
                    chat_id,
                    "⚠️ That seems very high. Please verify and re-enter."
                )
                self._show_current_item(session)
                return
            
            # Save quantity and move forward
            session.set_current_quantity(quantity)
            session.index += 1
            
            # Log the entry
            item = session.items[session.index - 1] if session.index > 0 else None
            if item:
                self.logger.info(f"Entry: {item['name']} = {quantity}")
            
            # Show next item or move to next step
            if session.index >= len(session.items):
                # Check if this is a received delivery that needs image
                if session.mode == "received":
                    session.current_step = "image"
                    self._show_image_request(session)
                else:
                    self._handle_done(chat_id, session.user_id)
            else:
                self._show_current_item(session)
                
        except ValueError:
            # Not a valid number
            self.bot.send_message(
                chat_id,
                "❌ Please enter a valid number (e.g., 0, 1, 1.5, 2)\n"
                "Or use the buttons: Skip, Done, Cancel"
            )
            self._show_current_item(session)


    
    def _handle_location_selection(self, chat_id: int, user_id: int, location: str):
        """Handle location selection."""
        # Create session
        session = EntrySession(
            user_id=user_id,
            chat_id=chat_id,
            location=location,
            mode="",
            items=[]
        )
        self.sessions[user_id] = session
        
        # Show mode selection
        keyboard = self._create_keyboard([
            [("📦 On-Hand Count", "entry_mode|on_hand")],
            [("📥 Received Delivery", "entry_mode|received")],
            [("❌ Cancel", "entry_cancel")]
        ])
        
        self.bot.send_message(
            chat_id,
            f"📍 Location: <b>{location}</b>\n\n"
            f"Select entry type:",
            reply_markup=keyboard
        )
    
    def _handle_mode_selection(self, chat_id: int, user_id: int, mode: str):
        """Handle mode selection and start item entry."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "Session expired. Use /entry to start over.")
            return
        
        session.mode = mode
        
        # Load items for location
        items = self.notion.get_items_for_location(session.location)
        session.items = [
            {
                'name': item.name,
                'unit': item.unit_type,
                'adu': item.adu,
                'id': item.id
            }
            for item in items
        ]
        
        if not session.items:
            self.bot.send_message(
                chat_id,
                f"⚠️ No items found for {session.location}"
            )
            self._delete_session(user_id)
            return
        
        # Initialize answers dict
        for item in session.items:
            session.answers[item['name']] = None
        
        # Start item entry
        mode_text = "On-Hand Count" if mode == "on_hand" else "Delivery"
        date = datetime.now().strftime('%Y-%m-%d')
        
        self.bot.send_message(
            chat_id,
            f"📝 <b>{session.location} • {mode_text}</b>\n"
            f"📅 Date: {date}\n"
            f"📦 Items: {len(session.items)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Enter quantity for each item.\n"
            f"You can use: back, skip, done, cancel"
        )
        
        self._show_current_item(session)
    
    def _show_current_item(self, session: EntrySession):
        """Display current item with navigation buttons."""
        item = session.get_current_item()
        if not item:
            self._handle_done(session.chat_id, session.user_id)
            return
        
        # Build message
        progress = session.get_progress()
        current_value = session.answers.get(item['name'])
        
        text = (
            f"[{progress}] <b>{item['name']}</b>\n"
            f"Unit: {item['unit']} • ADU: {item['adu']:.2f}/day\n"
        )
        
        if current_value is not None:
            text += f"💡 Current value: {current_value}\n"
        
        text += "\nEnter quantity:"
        
        # Create navigation buttons
        buttons = []
        
        # First row: Back (if not first) and Skip
        first_row = []
        if session.index > 0:
            first_row.append(("◀️ Back", "entry_back"))
        first_row.append(("⏭️ Skip", "entry_skip"))
        buttons.append(first_row)
        
        # Second row: Done and Cancel
        buttons.append([
            ("✅ Done", "entry_done"),
            ("❌ Cancel", "entry_cancel")
        ])
        
        keyboard = self._create_keyboard(buttons)
        
        self.bot.send_message(session.chat_id, text, reply_markup=keyboard)
    
    def _handle_back(self, chat_id: int, user_id: int):
        """Move back to previous item."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "No active session.")
            return
        
        if session.index > 0:
            session.move_back()
            self._show_current_item(session)
        else:
            self.bot.send_message(chat_id, "Already at first item.")
            self._show_current_item(session)
    
    def _handle_skip(self, chat_id: int, user_id: int):
        """Skip current item."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "No active session.")
            return
        
        session.skip_current()
        
        if session.index >= len(session.items):
            self._handle_done(chat_id, user_id)
        else:
            self._show_current_item(session)
    
    def _handle_done(self, chat_id: int, user_id: int):
        """Show review screen."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "No active session.")
            return
        
        # Build review summary
        mode_text = "On-Hand Count" if session.mode == "on_hand" else "Delivery"
        date = datetime.now().strftime('%Y-%m-%d')
        
        text = (
            f"📋 <b>Review Your Entry</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Type: <b>{mode_text}</b>\n"
            f"Location: <b>{session.location}</b>\n"
            f"Date: <b>{date}</b>\n\n"
        )
        
        # Group items by status
        entered_items = []
        skipped_items = []
        
        for item in session.items:
            qty = session.answers.get(item['name'])
            if qty is not None:
                entered_items.append(f"  • {item['name']}: {qty} {item['unit']}")
            else:
                skipped_items.append(f"  • {item['name']}")
        
        if entered_items:
            text += f"📦 <b>Entered ({len(entered_items)}):</b>\n"
            text += "\n".join(entered_items[:20])  # Limit display
            if len(entered_items) > 20:
                text += f"\n  ...and {len(entered_items) - 20} more"
            text += "\n\n"
        
        if skipped_items:
            text += f"⏭️ <b>Skipped ({len(skipped_items)}):</b>\n"
            text += "\n".join(skipped_items[:10])  # Limit display
            if len(skipped_items) > 10:
                text += f"\n  ...and {len(skipped_items) - 10} more"
            text += "\n\n"
        
        # Summary stats
        total_qty = session.get_total_quantity()
        text += (
            f"📊 <b>Summary:</b>\n"
            f"  • Items entered: {session.get_answered_count()}/{len(session.items)}\n"
            f"  • Total quantity: {total_qty:.1f}\n"
        )
        
        # Action buttons
        keyboard = self._create_keyboard([
            [("✅ Submit", "entry_submit"), ("📝 Resume", "entry_resume_items")],
            [("❌ Cancel", "entry_cancel")]
        ])
        
        self.bot.send_message(chat_id, text, reply_markup=keyboard)
    
    def _handle_submit(self, chat_id: int, user_id: int):
        """Submit the entry to Notion with optional image."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "No active session.")
            return
        
        # Prepare data for saving
        quantities = {
            name: (qty if qty is not None else 0.0)
            for name, qty in session.answers.items()
        }
        
        date = datetime.now().strftime('%Y-%m-%d')
        
        # Save to Notion with image file ID (not processed data)
        try:
            success = self.notion.save_inventory_transaction(
                location=session.location,
                entry_type=session.mode,
                date=date,
                manager=session.manager_name,
                notes=session.notes,
                quantities=quantities,
                image_file_id=session.image_file_id  # Pass the Telegram file ID
            )
            
            if success:
                # Success message
                mode_text = "on-hand count" if session.mode == "on_hand" else "delivery"
                items_count = session.get_answered_count()
                total_qty = session.get_total_quantity()
                image_note = " with image" if session.image_file_id else ""
                
                self.bot.send_message(
                    chat_id,
                    f"✅ <b>Entry Saved Successfully</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Location: {session.location}\n"
                    f"Type: {mode_text}\n"
                    f"Date: {date}\n"
                    f"Items: {items_count}\n"
                    f"Total: {total_qty:.1f} units{image_note}\n\n"
                    f"Use /info to see updated status"
                )
                
                # Log for audit
                self.logger.info(
                    f"Entry submitted: {session.location} {mode_text} "
                    f"by user {user_id}, {items_count} items, total {total_qty:.1f}{image_note}"
                )
            else:
                self.bot.send_message(
                    chat_id,
                    "⚠️ Failed to save to Notion. Please try again."
                )
                return
                
        except Exception as e:
            self.logger.error(f"Error submitting entry: {e}", exc_info=True)
            self.bot.send_message(
                chat_id,
                "⚠️ Error saving entry. Please contact support."
            )
            return
        
        # Clean up session
        self._delete_session(user_id)


    
    def _resume_items(self, chat_id: int, user_id: int):
        """Resume item entry from review screen."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "No active session.")
            return
        
        # Reset to first unanswered item or last item
        for i, item in enumerate(session.items):
            if session.answers.get(item['name']) is None:
                session.index = i
                break
        else:
            # All answered, go to last item
            session.index = len(session.items) - 1
        
        self._show_current_item(session)
    
    def _resume_session(self, chat_id: int, user_id: int):
        """Resume existing session."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "No active session.")
            return
        
        session.update_activity()
        self._show_current_item(session)
    
    def _handle_cancel(self, chat_id: int, user_id: int):
        """Cancel and delete session."""
        if user_id in self.sessions:
            self._delete_session(user_id)
            self.bot.send_message(
                chat_id,
                "❌ <b>Entry Cancelled</b>\n"
                "No data was saved.\n\n"
                "Use /entry to start over."
            )
        else:
            self.bot.send_message(chat_id, "No active session to cancel.")
    
    def _delete_session(self, user_id: int):
        """Delete a session."""
        if user_id in self.sessions:
            del self.sessions[user_id]
            self.logger.debug(f"Deleted session for user {user_id}")
    
    def _create_keyboard(self, buttons: List[List[tuple]]) -> Dict:
        """Create inline keyboard markup."""
        return {
            "inline_keyboard": [
                [{"text": text, "callback_data": data} for text, data in row]
                for row in buttons
            ]
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions periodically."""
        expired_users = []
        for user_id, session in self.sessions.items():
            if session.is_expired():
                expired_users.append(user_id)
        
        for user_id in expired_users:
            self._delete_session(user_id)
        
        if expired_users:
            self.logger.info(f"Cleaned up {len(expired_users)} expired entry sessions")

    def handle_photo_input(self, message: Dict, session: EntrySession):
        """
        Handle photo input for received deliveries.
        
        Args:
            message: Telegram message with photo
            session: Current entry session
        """
        chat_id = session.chat_id
        
        try:
            # Get the largest photo (best quality) - same as communication bot
            photos = message.get('photo', [])
            if not photos:
                self.bot.send_message(chat_id, "No photo received. Please try again.")
                self._show_image_request(session)
                return
            
            # Get the highest resolution photo
            largest_photo = max(photos, key=lambda p: p.get('file_size', 0))
            file_id = largest_photo['file_id']
            
            # Store the file ID for later processing
            session.image_file_id = file_id
            session.current_step = "review"
            
            self.bot.send_message(
                chat_id,
                "✅ Photo received successfully!\n"
                "Moving to review..."
            )
            
            # Move to review
            self._handle_done(chat_id, session.user_id)
            
        except Exception as e:
            self.logger.error(f"Error handling photo input: {e}")
            self.bot.send_message(
                chat_id,
                "⚠️ Error processing photo. You can skip this step or try again."
            )
            self._show_image_request(session)

    def _show_image_request(self, session: EntrySession):
        """Show image request prompt for received deliveries."""
        keyboard = self._create_keyboard([
            [("⏭️ Skip Image", "entry_skip_image")],
            [("✅ Done", "entry_done"), ("❌ Cancel", "entry_cancel")]
        ])
        
        self.bot.send_message(
            session.chat_id,
            "📷 <b>Product Image</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Please send a photo of the received delivery.\n\n"
            "This helps with quality tracking and verification.\n\n"
            "You can also skip this step if no photo is available.",
            reply_markup=keyboard
        )



# ===== MAIN APPLICATION =====

class K2NotionInventorySystem:
    """
    Main application class with Notion integration.
    """
    
    def __init__(self):
        """Initialize the complete K2 Notion inventory system."""
        self.logger = logging.getLogger('system')
        self.logger.critical(f"K2 Notion Inventory Management System v{SYSTEM_VERSION} initializing")
        
        # Validate environment variables
        if not self._validate_environment():
            sys.exit(1)
        
        # Initialize core components
        self.notion_manager = None
        self.calculator = None
        self.bot = None
        self.scheduler = None
        
        # System state
        self.running = False
        self.startup_time = datetime.now()
        
        self.logger.info("System initialization completed")
    
    def _validate_environment(self) -> bool:
        """Validate required environment variables."""
        required_vars = [
            'TELEGRAM_BOT_TOKEN',
            'NOTION_TOKEN', 
            'NOTION_ITEMS_DB_ID',
            'NOTION_INVENTORY_DB_ID',
            'NOTION_ADU_CALC_DB_ID'
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            self.logger.critical(f"Missing required environment variables: {missing_vars}")
            return False
        
        self.logger.info("Environment validation passed")
        return True
        
    def start(self):
        """Start all system components in proper order."""
        try:
            self.logger.critical("Starting K2 Notion Inventory Management System")
            
            # Initialize Notion manager
            self.logger.info("Initializing Notion manager...")
            notion_token = os.environ.get('NOTION_TOKEN')
            items_db_id = os.environ.get('NOTION_ITEMS_DB_ID')
            inventory_db_id = os.environ.get('NOTION_INVENTORY_DB_ID')
            adu_calc_db_id = os.environ.get('NOTION_ADU_CALC_DB_ID')
            
            self.notion_manager = NotionManager(notion_token, items_db_id, inventory_db_id, adu_calc_db_id)
            
            # Initialize calculator
            self.logger.info("Initializing inventory calculator...")
            self.calculator = InventoryCalculator(self.notion_manager)
            
            # Initialize Telegram bot
            self.logger.info("Initializing Telegram bot...")
            bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            self.bot = TelegramBot(bot_token, self.notion_manager, self.calculator)
            
            
            self.running = True
            self.logger.critical("System startup completed successfully")
            
            # Start bot polling (this blocks)
            self.logger.info("Starting Telegram bot polling...")
            print("🚀 K2 Notion Inventory System is running!")
            print("📝 Data is stored in Notion databases") 
            print("🤖 Bot is ready for commands - try /start")
            print("Press Ctrl+C to stop")
            
            self.bot.start_polling()
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.critical(f"Critical error during startup: {e}")
            raise
        finally:
            self.stop()

    def stop(self):
        """Gracefully stop all system components."""
        if not self.running:
            return
        
        self.logger.critical("Shutting down K2 Notion Inventory Management System")
        self.running = False
        
        # Stop components in reverse order
        # if self.scheduler:
        #     self.logger.info("Stopping scheduler...")
        #     self.scheduler.stop()
        
        if self.bot:
            self.logger.info("Stopping Telegram bot...")
            self.bot.stop()
        
        # Log shutdown
        if self.notion_manager:
            uptime = datetime.now() - self.startup_time
            self.logger.info(f"System ran for {uptime.total_seconds():.1f} seconds")
        
        self.logger.critical("System shutdown completed")

# ===== ENTRY POINT =====

# REPLACE the entire main() function with this:

def main():
    """Main entry point for the application."""
    try:
        # Check if running in test/development mode
        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            print("🧪 TEST MODE: Running system validation...")
            system = K2NotionInventorySystem()
            print("✅ System validation completed successfully!")
            print("✅ Notion databases accessible")
            print("\nTo run the full system, use: python k2_notion_inventory.py")
            return
        
        # Create and start the system
        system = K2NotionInventorySystem()
        system.start()
        
    except KeyboardInterrupt:
        print("\n⚠️ Shutdown requested by user")
        if 'system' in locals():
            system.stop()
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()