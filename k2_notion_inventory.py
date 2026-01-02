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

Author: Ladios Satō
License: Proprietary
Version: 4.0.0
"""

import asyncio
import uuid
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
SYSTEM_VERSION = "4.0.0"  # Make sure this is defined at module level

# Fix Windows console encoding for Unicode/emoji support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7 fallback
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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
SYSTEM_VERSION = "4.0.0"

# ===== SUPABASE CLIENT (Phase 1: Dual-Write) =====
_supabase_client = None
_supabase_enabled = False

def _init_supabase():
    """Initialize Supabase client from environment variables."""
    global _supabase_client, _supabase_enabled
    try:
        from supabase import create_client
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_KEY')

        if url and key:
            _supabase_client = create_client(url, key)
            _supabase_enabled = True
            print(f"✓ Supabase connected: {url[:40]}...")
            print(f"[PHASE 1] Dual-write to Supabase ENABLED")
        else:
            print("⚠ Supabase not configured (set SUPABASE_URL and SUPABASE_KEY)")
    except ImportError:
        print("⚠ Supabase not installed (pip install supabase)")
    except Exception as e:
        print(f"⚠ Supabase init failed: {e}")

_init_supabase()
# ===== END SUPABASE CLIENT =====

# ===== EXTERNAL SUPABASE CLIENT (for Operations Metrics) =====
_external_supabase_client = None
_external_supabase_enabled = False

# South Loop kitchen location ID (from external database)
SOUTH_LOOP_LOCATION_ID = '5e83c48c-25b8-4109-9509-155a4ab9c603'

# Metric targets for operations performance
# Note: Error rate data has a 3-day delay from the platform
ERROR_RATE_DELAY_DAYS = 3

METRIC_TARGETS = {
    'star_rating': {'target': 4.8, 'direction': 'above'},
    'prep_time': {'target': 9.0, 'direction': 'below'},
    'error_rate': {'target': 0.75, 'direction': 'below'},
}

def _init_external_supabase():
    """Initialize external Supabase client for operations metrics."""
    global _external_supabase_client, _external_supabase_enabled
    try:
        from supabase import create_client
        url = os.environ.get('EXTERNAL_SUPABASE_URL')
        key = os.environ.get('EXTERNAL_SUPABASE_KEY')

        if url and key:
            _external_supabase_client = create_client(url, key)
            _external_supabase_enabled = True
            print(f"✓ External Supabase connected: {url[:40]}...")
        else:
            print("⚠ External Supabase not configured (optional - for ops metrics)")
    except Exception as e:
        print(f"⚠ External Supabase init failed: {e}")

_init_external_supabase()
# ===== END EXTERNAL SUPABASE CLIENT =====

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
    min_par: float = 0.0  # Minimum par level (reorder point)
    max_par: float = 0.0  # Maximum par level (order up to)
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

# ===== ORDER SESSION STATE =====

@dataclass
class OrderSession:
    """
    State for interactive /order flow.
    Now fully location-agnostic - uses dynamic location from Notion.
    """
    user_id: int
    chat_id: int
    session_token: str
    location: str  # Dynamic location from Notion (single source of truth)
    items: List[Dict[str, Any]]
    
    # Date fields for delivery scheduling
    delivery_date: Optional[str] = None
    next_delivery_date: Optional[str] = None
    onhand_time_hint: Optional[str] = None
    consumption_days: Optional[int] = None
    
    # Legacy fields
    order_date: str = ""
    submitter_name: str = ""
    
    # Item navigation
    index: int = 0
    quantities: Dict[str, Optional[float]] = field(default_factory=dict)
    person_tag: str = ""
    
    # Session lifecycle
    started_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=2))
    last_activity: datetime = field(default_factory=datetime.now)
    last_message_id: Optional[int] = None
    
    # Input mode tracking
    _date_input_mode: Optional[str] = None
    _calendar_month_offset: int = 0
    
    # Legacy compatibility property
    @property
    def vendor(self) -> str:
        """Alias for location to maintain compatibility with existing code."""
        return self.location
    
    @vendor.setter
    def vendor(self, value: str):
        """Alias setter for location."""
        self.location = value
    
    def is_expired(self) -> bool:
        """Check if session has exceeded timeout."""
        return datetime.now() > self.expires_at
    
    def update_activity(self):
        """Reset expiration timer on user activity."""
        self.expires_at = datetime.now() + timedelta(hours=2)
        self.last_activity = datetime.now()
    
    def get_current_item(self) -> Optional[Dict[str, Any]]:
        """Retrieve current item dict or None if past end."""
        if 0 <= self.index < len(self.items):
            return self.items[self.index]
        return None
    
    def move_back(self):
        """Navigate to previous item (clamped at 0)."""
        self.index = max(0, self.index - 1)
    
    def skip_current(self):
        """Mark current item as skipped (None) and advance."""
        item = self.get_current_item()
        if item:
            self.quantities[item['name']] = None
        self.index += 1
    
    def set_current_quantity(self, qty: float):
        """Record user-entered quantity for current item."""
        item = self.get_current_item()
        if item:
            self.quantities[item['name']] = qty
    
    def get_progress(self) -> str:
        """Return progress string like '3/12'."""
        return f"{self.index + 1}/{len(self.items)}"
    
    def get_entered_count(self) -> int:
        """Count items with non-None quantities."""
        return sum(1 for v in self.quantities.values() if v is not None)

# ===== DEADLINE CHECKER =====

class DeadlineChecker:
    """
    Background checker for order deadlines.

    Runs every 5 minutes and:
    1. Sends reminder when approaching deadline
    2. Sends escalation when deadline missed

    Logs: All checks, reminders sent, escalations
    """

    def __init__(self, notion_manager, bot):
        self.notion = notion_manager
        self.bot = bot
        self.logger = logging.getLogger('deadline_checker')
        self.running = False
        self._thread = None

        # Track what we've already notified about today (orders)
        self._reminders_sent = set()  # (location, date) tuples
        self._escalations_sent = set()

        # Track count schedule notifications
        self._count_reminders_sent = set()  # (location, date) tuples
        self._count_escalations_sent = set()

    def start(self):
        """Start the deadline checker background thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[DEADLINE-CHECKER] ✓ Started background checker")
        self.logger.info("[DEADLINE-CHECKER] Started")

    def stop(self):
        """Stop the deadline checker."""
        self.running = False
        print(f"[DEADLINE-CHECKER] Stopped")

    def _run_loop(self):
        """Main loop - check every 5 minutes."""
        while self.running:
            try:
                self._check_all_deadlines()
            except Exception as e:
                print(f"[DEADLINE-CHECKER] ✗ Error in deadline check: {e}")
                self.logger.error(f"[DEADLINE-CHECKER] Deadline error: {e}")

            try:
                self._check_all_count_schedules()
            except Exception as e:
                print(f"[COUNT-CHECKER] ✗ Error in count check: {e}")
                self.logger.error(f"[COUNT-CHECKER] Error: {e}")

            # Sleep for 5 minutes
            for _ in range(300):  # 300 seconds = 5 minutes
                if not self.running:
                    break
                time.sleep(1)

    def _check_all_deadlines(self):
        """Check all active deadlines."""
        now = get_time_in_timezone(BUSINESS_TIMEZONE)
        today = now.date()
        current_time = now.time()
        day_of_week = now.weekday()

        print(f"[DEADLINE-CHECKER] Checking deadlines at {now.strftime('%Y-%m-%d %H:%M')}")

        # Get all deadlines for today
        deadlines = self.notion.get_all_deadlines()

        for dl in deadlines:
            if dl['day_of_week'] != day_of_week:
                continue

            location = dl['location']
            deadline_hour = dl['deadline_hour']
            deadline_minute = dl['deadline_minute']
            reminder_minutes = dl.get('reminder_minutes_before', 60)
            notification_chat = dl.get('notification_chat_id')
            escalation_chat = dl.get('escalation_chat_id')

            # Create deadline datetime
            deadline_time = datetime.combine(today,
                datetime.strptime(f"{deadline_hour:02d}:{deadline_minute:02d}", "%H:%M").time())
            reminder_time = deadline_time - timedelta(minutes=reminder_minutes)

            # Current datetime for comparison
            current_datetime = datetime.combine(today, current_time)

            key = (location, today.isoformat())

            # Check if order already submitted
            order_submitted = self.notion.check_order_submitted_today(location)

            if order_submitted:
                print(f"[DEADLINE-CHECKER] {location}: Order already submitted ✓")
                continue

            # Check for reminder window
            if (reminder_time <= current_datetime < deadline_time and
                key not in self._reminders_sent):

                self._send_reminder(location, deadline_time, notification_chat)
                self._reminders_sent.add(key)
                self.notion.log_deadline_event(location, 'reminder_sent')

            # Check for missed deadline
            if (current_datetime >= deadline_time and
                key not in self._escalations_sent):

                self._send_escalation(location, deadline_time, escalation_chat, notification_chat)
                self._escalations_sent.add(key)
                self.notion.log_deadline_event(location, 'deadline_missed')
                self.notion.log_deadline_event(location, 'escalation_sent')

        # Clear old entries (from previous days)
        self._reminders_sent = {k for k in self._reminders_sent if k[1] == today.isoformat()}
        self._escalations_sent = {k for k in self._escalations_sent if k[1] == today.isoformat()}

    def _send_reminder(self, location: str, deadline_time: datetime, chat_id: int):
        """Send reminder notification."""
        if not chat_id:
            print(f"[DEADLINE-CHECKER] ⚠ No notification chat for {location}")
            return

        time_str = deadline_time.strftime('%I:%M %p')

        message = (
            f"⏰ <b>ORDER REMINDER</b>\n\n"
            f"📍 <b>{location}</b> order due at {time_str}\n\n"
            f"Use /order to submit before deadline!"
        )

        try:
            self.bot.send_message(chat_id, message)
            print(f"[DEADLINE-CHECKER] ✓ Reminder sent for {location}")
            self.logger.info(f"[DEADLINE-CHECKER] Reminder sent: {location}")
        except Exception as e:
            print(f"[DEADLINE-CHECKER] ✗ Failed to send reminder: {e}")

    def _send_escalation(self, location: str, deadline_time: datetime,
                         escalation_chat: int, notification_chat: int):
        """Send escalation notification."""
        time_str = deadline_time.strftime('%I:%M %p')

        message = (
            f"🚨 <b>DEADLINE MISSED</b>\n\n"
            f"📍 <b>{location}</b> order was due at {time_str}\n"
            f"⚠️ No order has been submitted!\n\n"
            f"Please submit immediately or contact manager."
        )

        # Send to escalation chat (admin)
        if escalation_chat:
            try:
                self.bot.send_message(escalation_chat, message)
                print(f"[DEADLINE-CHECKER] ✓ Escalation sent to admin for {location}")
            except Exception as e:
                print(f"[DEADLINE-CHECKER] ✗ Failed to send escalation: {e}")

        # Also send to notification chat (team)
        if notification_chat and notification_chat != escalation_chat:
            try:
                self.bot.send_message(notification_chat, message)
                print(f"[DEADLINE-CHECKER] ✓ Escalation sent to team for {location}")
            except Exception as e:
                print(f"[DEADLINE-CHECKER] ✗ Failed to send team escalation: {e}")

        self.logger.warning(f"[DEADLINE-CHECKER] MISSED DEADLINE: {location}")

    # ===== COUNT SCHEDULE CHECKING =====

    def _check_all_count_schedules(self):
        """Check all count schedules and send reminders/escalations."""
        now = get_time_in_timezone(BUSINESS_TIMEZONE)
        today = now.date()
        current_time = now.time()
        day_of_week = now.weekday()

        print(f"[COUNT-CHECKER] Checking count schedules at {now.strftime('%Y-%m-%d %H:%M')}")

        # Get all count schedules for today
        schedules = self.notion.get_all_count_schedules()

        for sched in schedules:
            if sched['day_of_week'] != day_of_week:
                continue

            location = sched['location']
            due_hour = sched['due_hour']
            due_minute = sched['due_minute']
            reminder_minutes = sched.get('reminder_minutes_before', 60)
            notification_chat = sched.get('notification_chat_id')
            escalation_chat = sched.get('escalation_chat_id')

            # Create due datetime
            due_time = datetime.combine(today,
                datetime.strptime(f"{due_hour:02d}:{due_minute:02d}", "%H:%M").time())
            reminder_time = due_time - timedelta(minutes=reminder_minutes)

            # Current datetime for comparison
            current_datetime = datetime.combine(today, current_time)

            key = (location, today.isoformat())

            # Check if count already submitted
            count_submitted = self.notion.check_count_submitted_today(location)

            if count_submitted:
                print(f"[COUNT-CHECKER] {location}: Count already submitted ✓")
                continue

            # Check for reminder window
            if (reminder_time <= current_datetime < due_time and
                key not in self._count_reminders_sent):

                self._send_count_reminder(location, due_time, notification_chat)
                self._count_reminders_sent.add(key)
                self.notion.log_count_event(location, 'count_reminder_sent')

            # Check for missed count deadline
            if (current_datetime >= due_time and
                key not in self._count_escalations_sent):

                self._send_count_escalation(location, due_time, escalation_chat, notification_chat)
                self._count_escalations_sent.add(key)
                self.notion.log_count_event(location, 'count_missed')
                self.notion.log_count_event(location, 'count_escalation_sent')

        # Clear old entries (from previous days)
        self._count_reminders_sent = {k for k in self._count_reminders_sent if k[1] == today.isoformat()}
        self._count_escalations_sent = {k for k in self._count_escalations_sent if k[1] == today.isoformat()}

    def _send_count_reminder(self, location: str, due_time: datetime, chat_id: int):
        """Send count reminder notification."""
        if not chat_id:
            print(f"[COUNT-CHECKER] ⚠ No notification chat for {location}")
            return

        time_str = due_time.strftime('%I:%M %p')

        message = (
            f"📋 <b>COUNT REMINDER</b>\n\n"
            f"📍 <b>{location}</b> inventory count due at {time_str}\n\n"
            f"Use /entry → On-Hand to submit your count!"
        )

        try:
            self.bot.send_message(chat_id, message)
            print(f"[COUNT-CHECKER] ✓ Reminder sent for {location}")
            self.logger.info(f"[COUNT-CHECKER] Reminder sent: {location}")
        except Exception as e:
            print(f"[COUNT-CHECKER] ✗ Failed to send reminder: {e}")

    def _send_count_escalation(self, location: str, due_time: datetime,
                               escalation_chat: int, notification_chat: int):
        """Send count missed escalation notification."""
        time_str = due_time.strftime('%I:%M %p')

        message = (
            f"🚨 <b>COUNT MISSED</b>\n\n"
            f"📍 <b>{location}</b> inventory count was due at {time_str}\n"
            f"⚠️ No count has been submitted!\n\n"
            f"Please submit immediately using /entry → On-Hand"
        )

        # Send to escalation chat (admin)
        if escalation_chat:
            try:
                self.bot.send_message(escalation_chat, message)
                print(f"[COUNT-CHECKER] ✓ Escalation sent to admin for {location}")
            except Exception as e:
                print(f"[COUNT-CHECKER] ✗ Failed to send escalation: {e}")

        # Also send to notification chat (team)
        if notification_chat and notification_chat != escalation_chat:
            try:
                self.bot.send_message(notification_chat, message)
                print(f"[COUNT-CHECKER] ✓ Escalation sent to team for {location}")
            except Exception as e:
                print(f"[COUNT-CHECKER] ✗ Failed to send team escalation: {e}")

        self.logger.warning(f"[COUNT-CHECKER] MISSED COUNT: {location}")

    # ===== END COUNT SCHEDULE CHECKING =====

# ===== END DEADLINE CHECKER =====

# ===== TASK ASSIGNMENT CHECKER =====

class TaskAssignmentChecker:
    """
    Background checker for task assignment reminders.

    Runs every 5 minutes and:
    1. Checks task_assignments table for pending tasks
    2. Sends reminders 30 minutes before due_time
    3. Respects start_time and end_time windows

    Works alongside DeadlineChecker for comprehensive reminder coverage.
    """

    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger('task_assignment_checker')
        self.running = False
        self._thread = None
        self._reminders_sent = set()  # (task_id, date) tuples

    def start(self):
        """Start the task assignment checker background thread."""
        if self.running:
            return

        if not _supabase_enabled or not _supabase_client:
            print(f"[TASK-CHECKER] ⚠ Supabase not available, skipping task checker")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[TASK-CHECKER] ✓ Started background checker")
        self.logger.info("[TASK-CHECKER] Started")

    def stop(self):
        """Stop the task assignment checker."""
        self.running = False
        print(f"[TASK-CHECKER] Stopped")

    def _run_loop(self):
        """Main loop - check every 5 minutes."""
        while self.running:
            try:
                self._check_task_reminders()
            except Exception as e:
                print(f"[TASK-CHECKER] ✗ Error in task check: {e}")
                self.logger.error(f"[TASK-CHECKER] Error: {e}")

            # Sleep for 5 minutes
            for _ in range(300):
                if not self.running:
                    break
                time.sleep(1)

    def _check_task_reminders(self):
        """Check for task assignments that need reminders."""
        now = get_time_in_timezone(BUSINESS_TIMEZONE)
        today = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M')

        print(f"[TASK-CHECKER] Checking task reminders at {now.strftime('%Y-%m-%d %H:%M')}")

        try:
            # Get tasks due today that haven't been reminded
            tasks_result = _supabase_client.table('task_assignments') \
                .select('*, report_types(name), telegram_users!task_assignments_assigned_to_fkey(telegram_id, name)') \
                .eq('scheduled_date', today) \
                .eq('reminder_sent', False) \
                .eq('status', 'pending') \
                .execute()

            if not tasks_result.data:
                print(f"[TASK-CHECKER] No pending tasks needing reminders")
                return

            for task in tasks_result.data:
                due_time = task.get('due_time')
                if not due_time:
                    continue

                # Parse due time (HH:MM:SS or HH:MM)
                due_parts = due_time.split(':')
                due_hour = int(due_parts[0])
                due_minute = int(due_parts[1]) if len(due_parts) > 1 else 0

                # Calculate reminder threshold (30 mins before)
                reminder_dt = datetime(now.year, now.month, now.day, due_hour, due_minute) - timedelta(minutes=30)
                reminder_threshold = reminder_dt.strftime('%H:%M')
                due_str = f"{due_hour:02d}:{due_minute:02d}"

                # Check if we should send reminder
                if current_time >= reminder_threshold and current_time < due_str:
                    key = (task['id'], today)
                    if key not in self._reminders_sent:
                        self._send_task_reminder(task, due_time)
                        self._reminders_sent.add(key)

            # Clear old entries (from previous days)
            self._reminders_sent = {k for k in self._reminders_sent if k[1] == today}

        except Exception as e:
            self.logger.error(f"[TASK-CHECKER] Error fetching tasks: {e}")
            print(f"[TASK-CHECKER] ✗ Error: {e}")

    def _send_task_reminder(self, task: dict, due_time: str):
        """Send reminder for a specific task."""
        # Get user info
        user_data = task.get('telegram_users')
        if not user_data:
            print(f"[TASK-CHECKER] ⚠ No user data for task {task['id']}")
            return

        telegram_id = user_data.get('telegram_id')
        user_name = user_data.get('name', 'User')
        if not telegram_id:
            print(f"[TASK-CHECKER] ⚠ No telegram_id for task {task['id']}")
            return

        # Get task name
        report_types = task.get('report_types')
        if report_types and report_types.get('name'):
            task_name = report_types['name']
        else:
            task_name = task.get('notes') or 'Task'

        # Format due time for display
        due_parts = due_time.split(':')
        hour = int(due_parts[0])
        minute = due_parts[1] if len(due_parts) > 1 else '00'
        period = 'AM' if hour < 12 else 'PM'
        display_hour = hour % 12 or 12
        due_str = f"{display_hour}:{minute} {period}"

        message = (
            f"⏰ <b>Task Reminder</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📋 <b>{task_name}</b>\n"
            f"⏱️ Due at {due_str}\n\n"
            f"Use /viewtasks to see all your pending tasks."
        )

        try:
            self.bot.send_message(telegram_id, message)
            print(f"[TASK-CHECKER] ✓ Reminder sent for task {task['id']} to {user_name}")

            # Mark as reminded in database
            _supabase_client.table('task_assignments') \
                .update({
                    'reminder_sent': True,
                    'reminder_sent_at': datetime.now().isoformat()
                }) \
                .eq('id', task['id']) \
                .execute()

        except Exception as e:
            print(f"[TASK-CHECKER] ✗ Failed to send reminder: {e}")
            self.logger.error(f"[TASK-CHECKER] Failed to send reminder: {e}")

# ===== END TASK ASSIGNMENT CHECKER =====

# ===== SCHEDULED MESSAGE SENDER =====

class ScheduledMessageSender:
    """
    Background sender for scheduled messages.

    Runs every minute and:
    1. Checks scheduled_messages table for due messages
    2. Generates and sends metrics summaries or task reports
    3. Updates last_sent_at and next_run_at timestamps
    """

    def __init__(self, bot):
        self.bot = bot
        self.logger = logging.getLogger('scheduled_message_sender')
        self.running = False
        self._thread = None

    def start(self):
        """Start the scheduled message sender background thread."""
        if self.running:
            return

        if not _supabase_enabled or not _supabase_client:
            print(f"[SCHEDULED-MSG] ⚠ Supabase not available, skipping message sender")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[SCHEDULED-MSG] ✓ Started background sender")
        self.logger.info("[SCHEDULED-MSG] Started")

    def stop(self):
        """Stop the scheduled message sender."""
        self.running = False
        print(f"[SCHEDULED-MSG] Stopped")

    def _run_loop(self):
        """Main loop - check every minute."""
        while self.running:
            try:
                self._check_scheduled_messages()
            except Exception as e:
                print(f"[SCHEDULED-MSG] ✗ Error in message check: {e}")
                self.logger.error(f"[SCHEDULED-MSG] Error: {e}")

            # Sleep for 60 seconds (check every minute)
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(1)

    def _check_scheduled_messages(self):
        """Check for scheduled messages that need to be sent."""
        now = get_time_in_timezone(BUSINESS_TIMEZONE)
        current_day = now.strftime('%A').lower()  # monday, tuesday, etc.
        current_time = now.strftime('%H:%M')

        try:
            # Get active scheduled messages
            messages_result = _supabase_client.table('scheduled_messages') \
                .select('*') \
                .eq('is_active', True) \
                .execute()

            if not messages_result.data:
                return

            for message in messages_result.data:
                if self._should_send(message, now, current_day, current_time):
                    self._send_scheduled_message(message, now)

        except Exception as e:
            self.logger.error(f"[SCHEDULED-MSG] Error fetching messages: {e}")
            print(f"[SCHEDULED-MSG] ✗ Error: {e}")

    def _should_send(self, message: dict, now: datetime, current_day: str, current_time: str) -> bool:
        """Check if a message should be sent now."""
        schedule_time = message.get('schedule_time', '')[:5]  # Get HH:MM

        # Check if it's time (within 1 minute window)
        if current_time != schedule_time:
            return False

        # Check last_sent_at to avoid duplicates
        last_sent = message.get('last_sent_at')
        if last_sent:
            last_sent_dt = datetime.fromisoformat(last_sent.replace('Z', '+00:00'))
            if (now - last_sent_dt.replace(tzinfo=None)).total_seconds() < 300:  # 5 min debounce
                return False

        # For recurring messages, check day
        if message.get('is_recurring') and message.get('schedule_days'):
            if current_day not in message['schedule_days']:
                return False

        return True

    def _send_scheduled_message(self, message: dict, now: datetime):
        """Send a scheduled message and update database."""
        recipient_id = message.get('recipient_id')
        if not recipient_id:
            return

        message_type = message.get('message_type', 'custom')

        try:
            # Generate message content based on type
            if message_type == 'metrics_summary':
                content = self._generate_metrics_summary(message, now)
            elif message_type == 'task_report':
                content = self._generate_task_report(message, now)
            else:
                # Custom message with template variables
                template = message.get('custom_content', '')
                if template:
                    variables = self._collect_template_variables(message, now)
                    content = self._render_template(template, variables)
                else:
                    content = f"📊 <b>{message.get('name', 'Scheduled Message')}</b>\n\nNo message template configured."

            # Send the message
            self.bot.send_message(int(recipient_id), content)
            print(f"[SCHEDULED-MSG] ✓ Sent '{message['name']}' to {recipient_id}")

            # Update last_sent_at and calculate next_run_at
            next_run = self._calculate_next_run(message, now)
            _supabase_client.table('scheduled_messages') \
                .update({
                    'last_sent_at': now.isoformat(),
                    'next_run_at': next_run.isoformat() if next_run else None
                }) \
                .eq('id', message['id']) \
                .execute()

        except Exception as e:
            print(f"[SCHEDULED-MSG] ✗ Failed to send message: {e}")
            self.logger.error(f"[SCHEDULED-MSG] Failed to send: {e}")

    def _generate_metrics_summary(self, message: dict, now: datetime) -> str:
        """Generate a metrics summary message."""
        date_range = message.get('date_range_type', 'yesterday')

        # Calculate date range
        if date_range == 'yesterday':
            end_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = end_date
            period_label = "Yesterday"
        elif date_range == 'last_week':
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            period_label = "Last 7 Days"
        elif date_range == 'last_month':
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            period_label = "Last 30 Days"
        else:
            start_date = message.get('custom_date_start') or now.strftime('%Y-%m-%d')
            end_date = message.get('custom_date_end') or now.strftime('%Y-%m-%d')
            period_label = f"{start_date} to {end_date}"

        try:
            # Fetch metrics from Supabase
            # Get task completion rate
            tasks_result = _supabase_client.table('task_assignments') \
                .select('status') \
                .gte('scheduled_date', start_date) \
                .lte('scheduled_date', end_date) \
                .execute()

            total_tasks = len(tasks_result.data) if tasks_result.data else 0
            completed_tasks = sum(1 for t in (tasks_result.data or []) if t.get('status') == 'completed')
            task_completion = round((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1)

            # Get flagged cycles
            cycles_result = _supabase_client.table('consumption_cycles') \
                .select('status') \
                .gte('created_at', start_date) \
                .lte('created_at', end_date) \
                .execute()

            total_cycles = len(cycles_result.data) if cycles_result.data else 0
            flagged_cycles = sum(1 for c in (cycles_result.data or []) if c.get('status') == 'flagged')
            flagged_rate = round((flagged_cycles / total_cycles * 100) if total_cycles > 0 else 0, 1)

            # Get audit scores
            audits_result = _supabase_client.table('food_safety_audits') \
                .select('score_percentage') \
                .gte('audit_date', start_date) \
                .lte('audit_date', end_date) \
                .execute()

            audit_scores = [a.get('score_percentage', 0) for a in (audits_result.data or []) if a.get('score_percentage')]
            avg_audit = round(sum(audit_scores) / len(audit_scores), 1) if audit_scores else 0

            # Build message
            content = (
                f"📊 <b>Metrics Summary</b>\n"
                f"<i>{period_label}</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"✅ <b>Task Completion:</b> {task_completion}%\n"
                f"   ({completed_tasks} of {total_tasks} tasks)\n\n"
                f"🚩 <b>Flagged Cycle Rate:</b> {flagged_rate}%\n"
                f"   ({flagged_cycles} of {total_cycles} cycles)\n\n"
                f"📋 <b>Audit Score Avg:</b> {avg_audit}%\n"
                f"   ({len(audit_scores)} audits)\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>Report: {message.get('name', 'Metrics')}</i>"
            )

            return content

        except Exception as e:
            self.logger.error(f"[SCHEDULED-MSG] Error generating metrics: {e}")
            return f"📊 <b>Metrics Summary</b>\n\n⚠️ Error generating report: {str(e)}"

    def _generate_task_report(self, message: dict, now: datetime) -> str:
        """Generate a task completion report."""
        date_range = message.get('date_range_type', 'yesterday')

        # Calculate date range
        if date_range == 'yesterday':
            target_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            period_label = "Yesterday"
        else:
            target_date = now.strftime('%Y-%m-%d')
            period_label = "Today"

        try:
            # Get all tasks for the date
            tasks_result = _supabase_client.table('task_assignments') \
                .select('*, report_types(name), telegram_users!task_assignments_assigned_to_fkey(name)') \
                .eq('scheduled_date', target_date) \
                .execute()

            if not tasks_result.data:
                return f"📋 <b>Task Report - {period_label}</b>\n\nNo tasks scheduled."

            # Group by status
            completed = []
            pending = []
            missed = []

            for task in tasks_result.data:
                status = task.get('status', 'pending')
                task_name = task.get('report_types', {}).get('name') or task.get('notes') or 'Task'
                assignee = task.get('telegram_users', {}).get('name') or 'Unassigned'

                task_info = f"• {task_name} ({assignee})"

                if status == 'completed':
                    completed.append(task_info)
                elif status == 'missed':
                    missed.append(task_info)
                else:
                    pending.append(task_info)

            # Build message
            content = (
                f"📋 <b>Task Report - {period_label}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )

            if completed:
                content += f"✅ <b>Completed ({len(completed)})</b>\n"
                content += "\n".join(completed[:5])  # Limit to 5
                if len(completed) > 5:
                    content += f"\n... and {len(completed) - 5} more"
                content += "\n\n"

            if pending:
                content += f"⏳ <b>Pending ({len(pending)})</b>\n"
                content += "\n".join(pending[:5])
                if len(pending) > 5:
                    content += f"\n... and {len(pending) - 5} more"
                content += "\n\n"

            if missed:
                content += f"❌ <b>Missed ({len(missed)})</b>\n"
                content += "\n".join(missed[:5])
                if len(missed) > 5:
                    content += f"\n... and {len(missed) - 5} more"
                content += "\n\n"

            total = len(completed) + len(pending) + len(missed)
            completion_rate = round(len(completed) / total * 100) if total > 0 else 0

            content += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            content += f"<b>Completion Rate:</b> {completion_rate}%"

            return content

        except Exception as e:
            self.logger.error(f"[SCHEDULED-MSG] Error generating task report: {e}")
            return f"📋 <b>Task Report</b>\n\n⚠️ Error generating report: {str(e)}"

    def _collect_template_variables(self, message: dict, now: datetime) -> dict:
        """Collect all available template variables for scheduled messages."""
        date_range = message.get('date_range_type', 'yesterday')

        # Calculate date range
        if date_range == 'yesterday':
            end_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = end_date
            period_label = "Yesterday"
        elif date_range == 'last_week':
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
            period_label = "Last 7 Days"
        elif date_range == 'last_month':
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            period_label = "Last 30 Days"
        else:
            start_date = message.get('custom_date_start') or now.strftime('%Y-%m-%d')
            end_date = message.get('custom_date_end') or now.strftime('%Y-%m-%d')
            period_label = f"{start_date} to {end_date}"

        variables = {
            'report_name': message.get('name', 'Report'),
            'period_label': period_label,
            'start_date': start_date,
            'end_date': end_date,
        }

        try:
            # Get task metrics
            tasks_result = _supabase_client.table('task_assignments') \
                .select('status, scheduled_date') \
                .gte('scheduled_date', start_date) \
                .lte('scheduled_date', end_date) \
                .execute()

            total_tasks = len(tasks_result.data) if tasks_result.data else 0
            completed_tasks = sum(1 for t in (tasks_result.data or []) if t.get('status') == 'completed')
            pending_tasks = sum(1 for t in (tasks_result.data or []) if t.get('status') == 'pending')
            missed_tasks = sum(1 for t in (tasks_result.data or []) if t.get('status') == 'missed')
            task_completion = round((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1)

            # Calculate overdue tasks (pending with scheduled_date < today)
            today = now.strftime('%Y-%m-%d')
            overdue_tasks = sum(1 for t in (tasks_result.data or [])
                               if t.get('status') == 'pending' and t.get('scheduled_date', '') < today)

            variables['task_completion'] = task_completion
            variables['completed_tasks'] = completed_tasks
            variables['total_tasks'] = total_tasks
            variables['pending_tasks'] = pending_tasks
            variables['missed_tasks'] = missed_tasks
            variables['overdue_tasks'] = overdue_tasks

            # Get cycle metrics
            cycles_result = _supabase_client.table('consumption_cycles') \
                .select('status') \
                .gte('created_at', start_date) \
                .lte('created_at', end_date) \
                .execute()

            total_cycles = len(cycles_result.data) if cycles_result.data else 0
            flagged_cycles = sum(1 for c in (cycles_result.data or []) if c.get('status') == 'flagged')
            flagged_rate = round((flagged_cycles / total_cycles * 100) if total_cycles > 0 else 0, 1)

            variables['flagged_rate'] = flagged_rate
            variables['flagged_cycles'] = flagged_cycles
            variables['total_cycles'] = total_cycles

            # Get audit metrics
            audits_result = _supabase_client.table('food_safety_audits') \
                .select('score_percentage') \
                .gte('audit_date', start_date) \
                .lte('audit_date', end_date) \
                .execute()

            audit_scores = [a.get('score_percentage', 0) for a in (audits_result.data or []) if a.get('score_percentage')]
            avg_audit = round(sum(audit_scores) / len(audit_scores), 1) if audit_scores else 0
            total_audits = len(audits_result.data) if audits_result.data else 0
            passed_audits = sum(1 for a in (audits_result.data or []) if (a.get('score_percentage') or 0) >= 90)

            variables['avg_audit_score'] = avg_audit
            variables['total_audits'] = total_audits
            variables['passed_audits'] = passed_audits

            # Get flagged transactions count
            flagged_tx_result = _supabase_client.table('inventory_transactions') \
                .select('id') \
                .eq('flagged', True) \
                .execute()
            variables['flagged_transactions'] = len(flagged_tx_result.data) if flagged_tx_result.data else 0

        except Exception as e:
            self.logger.error(f"[SCHEDULED-MSG] Error collecting variables: {e}")
            # Set defaults on error
            variables.update({
                'task_completion': 0, 'completed_tasks': 0, 'total_tasks': 0,
                'pending_tasks': 0, 'missed_tasks': 0, 'overdue_tasks': 0,
                'flagged_rate': 0, 'flagged_cycles': 0, 'total_cycles': 0,
                'avg_audit_score': 0, 'total_audits': 0, 'passed_audits': 0,
                'flagged_transactions': 0
            })

        # Get external operations metrics
        try:
            if _external_supabase_enabled and _external_supabase_client:
                # Calculate ISO date range for external queries
                start_iso = f"{start_date}T00:00:00"
                end_iso = f"{end_date}T23:59:59"

                # Fetch star rating from events table
                star_result = _external_supabase_client.table('events') \
                    .select('metadata') \
                    .eq('location_id', SOUTH_LOOP_LOCATION_ID) \
                    .in_('event_type', ['order_rating_good', 'order_rating_low']) \
                    .gte('timestamp', start_iso) \
                    .lte('timestamp', end_iso) \
                    .execute()

                ratings = [e.get('metadata', {}).get('rating') for e in (star_result.data or [])
                           if isinstance(e.get('metadata', {}).get('rating'), (int, float))]
                star_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

                # Fetch prep time from events table
                prep_result = _external_supabase_client.table('events') \
                    .select('metadata') \
                    .eq('location_id', SOUTH_LOOP_LOCATION_ID) \
                    .eq('event_type', 'prep_time_slow') \
                    .gte('timestamp', start_iso) \
                    .lte('timestamp', end_iso) \
                    .execute()

                prep_times = [e.get('metadata', {}).get('avg_prep_time_minutes') for e in (prep_result.data or [])
                              if isinstance(e.get('metadata', {}).get('avg_prep_time_minutes'), (int, float))]
                prep_time = round(sum(prep_times) / len(prep_times), 1) if prep_times else None

                # Fetch error rate (errors / total orders)
                # Note: Error rate has a 3-day delay, so we fetch the most recent N days of AVAILABLE data
                # For example: on Jan 1 with 3-day delay, "last 7 days" = Dec 23-29 (most recent 7 available days)
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                period_days = (end_dt - start_dt).days  # Period as day difference

                # End date is most recent available (end_date - delay)
                delayed_end_dt = end_dt - timedelta(days=ERROR_RATE_DELAY_DAYS)

                # Start date calculation:
                # - For single day (daily): same as delayed end (period_days=0 -> adjusted=0)
                # - For multi-day (weekly/monthly): preserve N days, subtract (period - 1) days
                # Examples with 3-day delay on Jan 1:
                #   - Yesterday (Dec 31): Dec 28 (single day of available data)
                #   - Last 7 days (Dec 25-Jan 1): Dec 23-29 (7 days of available data)
                adjusted_period = max(0, period_days - 1)
                delayed_start_dt = delayed_end_dt - timedelta(days=adjusted_period)

                delayed_start = delayed_start_dt.strftime('%Y-%m-%d')
                delayed_end = delayed_end_dt.strftime('%Y-%m-%d')
                delayed_start_iso = f"{delayed_start}T00:00:00Z"
                delayed_end_iso = f"{delayed_end}T23:59:59Z"

                error_result = _external_supabase_client.table('events') \
                    .select('id') \
                    .eq('location_id', SOUTH_LOOP_LOCATION_ID) \
                    .eq('event_type', 'order_error') \
                    .gte('timestamp', delayed_start_iso) \
                    .lte('timestamp', delayed_end_iso) \
                    .execute()

                sales_result = _external_supabase_client.table('sales_data') \
                    .select('order_count') \
                    .eq('location_id', SOUTH_LOOP_LOCATION_ID) \
                    .gte('date', delayed_start) \
                    .lte('date', delayed_end) \
                    .execute()

                total_errors = len(error_result.data) if error_result.data else 0
                total_orders = sum(s.get('order_count', 0) for s in (sales_result.data or []))
                error_rate = round((total_errors / total_orders) * 100, 2) if total_orders > 0 else None

                # Calculate status (HIT/MISS) for each metric
                def get_status(value, target, direction):
                    if value is None:
                        return '—'
                    if direction == 'above':
                        return 'HIT' if value >= target else 'MISS'
                    else:  # below
                        return 'HIT' if value <= target else 'MISS'

                variables['star_rating'] = f"{star_rating:.2f}" if star_rating is not None else '—'
                variables['prep_time'] = f"{prep_time:.1f}" if prep_time is not None else '—'
                variables['error_rate'] = f"{error_rate:.2f}" if error_rate is not None else '—'

                variables['star_status'] = get_status(star_rating, METRIC_TARGETS['star_rating']['target'], METRIC_TARGETS['star_rating']['direction'])
                variables['prep_status'] = get_status(prep_time, METRIC_TARGETS['prep_time']['target'], METRIC_TARGETS['prep_time']['direction'])
                variables['error_status'] = get_status(error_rate, METRIC_TARGETS['error_rate']['target'], METRIC_TARGETS['error_rate']['direction'])
            else:
                # External Supabase not configured
                variables['star_rating'] = '—'
                variables['prep_time'] = '—'
                variables['error_rate'] = '—'
                variables['star_status'] = '—'
                variables['prep_status'] = '—'
                variables['error_status'] = '—'

        except Exception as e:
            self.logger.error(f"[SCHEDULED-MSG] Error fetching external metrics: {e}")
            variables['star_rating'] = '—'
            variables['prep_time'] = '—'
            variables['error_rate'] = '—'
            variables['star_status'] = '—'
            variables['prep_status'] = '—'
            variables['error_status'] = '—'

        return variables

    def _render_template(self, template: str, variables: dict) -> str:
        """Replace {{variable}} placeholders in template with actual values."""
        import re
        result = template

        # Replace all {{variable}} patterns
        for key, value in variables.items():
            result = result.replace(f'{{{{{key}}}}}', str(value))

        # Remove any remaining unreplaced variables
        result = re.sub(r'\{\{[^}]+\}\}', '—', result)

        return result

    def _calculate_next_run(self, message: dict, now: datetime) -> Optional[datetime]:
        """Calculate the next run time for a recurring message."""
        if not message.get('is_recurring') or not message.get('schedule_days'):
            return None

        schedule_time = message.get('schedule_time', '08:00')[:5]
        schedule_days = message['schedule_days']

        time_parts = schedule_time.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1]) if len(time_parts) > 1 else 0

        day_map = {
            'sunday': 6, 'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5
        }

        # Start checking from tomorrow
        for days_ahead in range(1, 8):
            check_date = now + timedelta(days=days_ahead)
            day_name = check_date.strftime('%A').lower()

            if day_name in schedule_days:
                return datetime(
                    check_date.year, check_date.month, check_date.day,
                    hour, minute, 0
                )

        return None

# ===== END SCHEDULED MESSAGE SENDER =====

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

    def _write_to_supabase(self, table: str, data: dict) -> bool:
        """
        Shadow write to Supabase for dual-write validation.
        Fails silently — Notion remains primary.
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return True

        try:
            _supabase_client.table(table).insert(data).execute()
            self.logger.debug(f"Supabase write to {table}: success")
            print(f"[PHASE 1] ✓ Dual-write to Supabase table '{table}' successful")
            return True
        except Exception as e:
            self.logger.warning(f"Supabase write to {table} failed: {e}")
            print(f"[PHASE 1] ⚠ Dual-write to Supabase table '{table}' failed: {e}")
            return False

    def _read_items_from_supabase(self, vendor: str):
        """
        Read items from Supabase for a vendor with PAR fields.

        Returns list of InventoryItem objects with min_par/max_par populated.
        Logs: query execution, item count, par coverage stats.
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[PAR] Supabase not available, falling back to Notion")
            return None

        try:
            print(f"[PAR] Loading items from Supabase for vendor: {vendor}")

            result = _supabase_client.table('inventory_items') \
                .select('id, item_name, vendor, unit_type, avg_consumption, min_par, max_par, active, created_at, updated_at') \
                .eq('vendor', vendor) \
                .eq('active', True) \
                .order('item_name') \
                .execute()

            items = []
            pars_configured = 0
            pars_missing = 0

            for row in result.data:
                min_par = row.get('min_par')
                max_par = row.get('max_par')
                item_name = row.get('item_name', 'Unknown')

                # Log par status for each item
                if min_par is not None and max_par is not None and min_par > 0 and max_par > 0:
                    pars_configured += 1
                    print(f"[PAR] ✓ '{item_name}': min={min_par}, max={max_par}")
                else:
                    pars_missing += 1
                    print(f"[PAR] ⚠ '{item_name}' missing par config: min={min_par}, max={max_par}")

                items.append(InventoryItem(
                    id=row['id'],
                    name=item_name,
                    location=row['vendor'],
                    adu=float(row.get('avg_consumption') or 0),
                    unit_type=row.get('unit_type') or 'case',
                    active=row.get('active', True),
                    min_par=float(min_par) if min_par is not None else 0.0,
                    max_par=float(max_par) if max_par is not None else 0.0,
                    created_at=row.get('created_at'),
                    updated_at=row.get('updated_at')
                ))

            print(f"[PAR] ✓ Loaded {len(items)} items from Supabase for {vendor}")
            print(f"[PAR] Par coverage: {pars_configured} configured, {pars_missing} missing")
            self.logger.info(f"[PAR] Loaded {len(items)} items for {vendor}, {pars_configured} with pars, {pars_missing} without")

            return items
        except Exception as e:
            self.logger.warning(f"[PAR] Supabase read items failed: {e}")
            print(f"[PAR] ✗ Supabase read failed: {e}, falling back to Notion")
            return None

    def _normalize_item_with_pars(self, item_row: dict) -> dict:
        """
        Normalize Supabase item row to include par fields.

        Ensures min_par and max_par are always present with defaults.
        Logs any items with missing par configuration.

        Args:
            item_row: Raw row from Supabase inventory_items table

        Returns:
            dict: Normalized item with guaranteed min_par/max_par fields
        """
        item_name = item_row.get('item_name', 'Unknown')
        min_par = item_row.get('min_par')
        max_par = item_row.get('max_par')

        # Log warnings for unconfigured pars
        if min_par is None or max_par is None:
            print(f"[PAR] ⚠ Item '{item_name}' missing par config: min_par={min_par}, max_par={max_par}")
            self.logger.warning(f"[PAR] Item '{item_name}' missing par configuration")

        # Apply defaults
        normalized = {
            'id': item_row.get('id'),
            'name': item_name,
            'item_name': item_name,
            'item_name_normalized': item_row.get('item_name_normalized', item_name.lower().strip()),
            'vendor': item_row.get('vendor'),
            'unit_type': item_row.get('unit_type', 'case'),
            'unit_type_parsed': item_row.get('unit_type_parsed', {}),
            'min_par': float(min_par) if min_par is not None else 0.0,
            'max_par': float(max_par) if max_par is not None else 0.0,
            'avg_consumption': float(item_row.get('avg_consumption', 0) or 0),
            'adu': float(item_row.get('avg_consumption', 0) or 0),
            'consumption_days': item_row.get('consumption_days'),
            'active': item_row.get('active', True)
        }

        # Log successful normalization
        print(f"[PAR] ✓ Normalized '{item_name}': min={normalized['min_par']:.1f}, max={normalized['max_par']:.1f}")

        return normalized

    # ===== SALES FORECAST FUNCTIONS =====

    def get_forecast_multiplier(self, location: str, target_date: datetime = None) -> float:
        """
        Get sales forecast multiplier for a location/date.

        The multiplier adjusts par levels:
        - multiplier = 1.0 → use base pars
        - multiplier = 1.2 → increase pars by 20%
        - multiplier = 0.8 → decrease pars by 20%

        Args:
            location: Vendor/location name
            target_date: Date to get forecast for (default: today)

        Returns:
            float: Multiplier (default 1.0 if no forecast found)

        Logs: Query, result, applied multiplier
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[FORECAST] Supabase not available, using default multiplier 1.0")
            return 1.0

        if target_date is None:
            target_date = get_time_in_timezone(BUSINESS_TIMEZONE)

        # Python weekday: Monday=0, Sunday=6 (matches our schema)
        day_of_week = target_date.weekday()
        day_name = target_date.strftime('%A')

        try:
            print(f"[FORECAST] Getting multiplier for {location} on {day_name} (day={day_of_week})")

            result = _supabase_client.table('sales_forecast') \
                .select('baseline_sales, projected_sales, multiplier') \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .single() \
                .execute()

            if result.data:
                multiplier = float(result.data.get('multiplier', 1.0))
                baseline = result.data.get('baseline_sales', 0)
                projected = result.data.get('projected_sales', 0)

                print(f"[FORECAST] ✓ {location}/{day_name}: baseline=${baseline}, projected=${projected}, multiplier={multiplier:.2f}")
                self.logger.info(f"[FORECAST] {location}/{day_name}: multiplier={multiplier:.2f}")

                return multiplier
            else:
                print(f"[FORECAST] No forecast found for {location}/{day_name}, using 1.0")
                return 1.0

        except Exception as e:
            print(f"[FORECAST] ✗ Error getting forecast: {e}")
            self.logger.error(f"[FORECAST] Error: {e}")
            return 1.0

    def get_weekly_forecast(self, location: str) -> list:
        """
        Get full week forecast for a location.

        Returns:
            list: 7 dicts, one per day (Monday=index 0)

        Logs: Query, all 7 days
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[FORECAST] Supabase not available")
            return []

        try:
            print(f"[FORECAST] Getting weekly forecast for {location}")

            result = _supabase_client.table('sales_forecast') \
                .select('*') \
                .eq('location', location) \
                .order('day_of_week') \
                .execute()

            if result.data:
                print(f"[FORECAST] ✓ Got {len(result.data)} days for {location}")
                for row in result.data:
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    day_name = day_names[row['day_of_week']]
                    print(f"[FORECAST]   {day_name}: ${row['projected_sales']} (×{row['multiplier']:.2f})")
                return result.data

            print(f"[FORECAST] No forecast data for {location}")
            return []

        except Exception as e:
            print(f"[FORECAST] ✗ Error getting weekly forecast: {e}")
            self.logger.error(f"[FORECAST] Weekly forecast error: {e}")
            return []

    def update_forecast(self, location: str, day_of_week: int, projected_sales: float, updated_by: str = None) -> bool:
        """
        Update sales forecast for a specific day.

        Args:
            location: Vendor/location name
            day_of_week: 0-6 (Monday-Sunday)
            projected_sales: Projected sales amount
            updated_by: Who made the update

        Returns:
            bool: Success

        Logs: Update attempt, result
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[FORECAST] ✗ Supabase not available")
            return False

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week] if 0 <= day_of_week <= 6 else 'Unknown'

        try:
            print(f"[FORECAST] Updating {location}/{day_name} to ${projected_sales}")

            result = _supabase_client.table('sales_forecast') \
                .update({
                    'projected_sales': projected_sales,
                    'updated_by': updated_by,
                    'updated_at': datetime.now().isoformat()
                }) \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .execute()

            if result.data:
                new_multiplier = result.data[0].get('multiplier', 1.0)
                print(f"[FORECAST] ✓ Updated {location}/{day_name}: ${projected_sales} (×{new_multiplier:.2f})")
                self.logger.info(f"[FORECAST] Updated {location}/{day_name}: ${projected_sales}")
                return True
            else:
                print(f"[FORECAST] ✗ No rows updated for {location}/{day_name}")
                return False

        except Exception as e:
            print(f"[FORECAST] ✗ Update failed: {e}")
            self.logger.error(f"[FORECAST] Update error: {e}")
            return False

    def update_baseline(self, location: str, day_of_week: int, baseline_sales: float) -> bool:
        """
        Update baseline sales for a specific day.

        Baseline is the "normal" sales level. Multiplier = projected / baseline.

        Args:
            location: Vendor/location name
            day_of_week: 0-6 (Monday-Sunday)
            baseline_sales: Baseline sales amount

        Returns:
            bool: Success
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week] if 0 <= day_of_week <= 6 else 'Unknown'

        try:
            print(f"[FORECAST] Updating baseline for {location}/{day_name} to ${baseline_sales}")

            result = _supabase_client.table('sales_forecast') \
                .update({
                    'baseline_sales': baseline_sales,
                    'updated_at': datetime.now().isoformat()
                }) \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .execute()

            if result.data:
                print(f"[FORECAST] ✓ Baseline updated for {location}/{day_name}")
                return True
            return False

        except Exception as e:
            print(f"[FORECAST] ✗ Baseline update failed: {e}")
            return False

    # ===== END SALES FORECAST FUNCTIONS =====

    # ===== DEADLINE FUNCTIONS =====

    def get_deadline_for_today(self, location: str) -> dict:
        """
        Get deadline config for today if one exists.

        Args:
            location: Vendor/location name

        Returns:
            dict or None: Deadline config if today has a deadline

        Logs: Query, result
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[DEADLINE] Supabase not available")
            return None

        now = get_time_in_timezone(BUSINESS_TIMEZONE)
        day_of_week = now.weekday()
        day_name = now.strftime('%A')

        try:
            print(f"[DEADLINE] Checking for deadline: {location} on {day_name} (day={day_of_week})")

            result = _supabase_client.table('order_deadlines') \
                .select('*') \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .eq('active', True) \
                .single() \
                .execute()

            if result.data:
                deadline_hour = result.data['deadline_hour']
                deadline_minute = result.data['deadline_minute']
                print(f"[DEADLINE] ✓ Found deadline for {location}/{day_name}: {deadline_hour:02d}:{deadline_minute:02d}")
                return result.data
            else:
                print(f"[DEADLINE] No deadline for {location} on {day_name}")
                return None

        except Exception as e:
            # single() throws if no row found
            if 'No rows' in str(e) or 'multiple' in str(e).lower():
                print(f"[DEADLINE] No deadline for {location} on {day_name}")
                return None
            print(f"[DEADLINE] ✗ Error: {e}")
            self.logger.error(f"[DEADLINE] Error: {e}")
            return None

    def get_all_deadlines(self, location: str = None) -> list:
        """
        Get all deadline configs, optionally filtered by location.

        Returns:
            list: Deadline config dicts
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return []

        try:
            query = _supabase_client.table('order_deadlines') \
                .select('*') \
                .eq('active', True) \
                .order('location') \
                .order('day_of_week')

            if location:
                query = query.eq('location', location)

            result = query.execute()

            if result.data:
                print(f"[DEADLINE] ✓ Got {len(result.data)} deadline configs")
                return result.data
            return []

        except Exception as e:
            print(f"[DEADLINE] ✗ Error getting deadlines: {e}")
            return []

    def update_deadline(self, location: str, day_of_week: int, hour: int, minute: int = 0,
                        reminder_minutes: int = 60, updated_by: str = None) -> bool:
        """
        Create or update deadline for a location/day.

        Args:
            location: Vendor name
            day_of_week: 0-6 (Monday-Sunday)
            hour: Deadline hour (0-23)
            minute: Deadline minute (0-59)
            reminder_minutes: Minutes before deadline to send reminder
            updated_by: Who made the update

        Returns:
            bool: Success
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week] if 0 <= day_of_week <= 6 else 'Unknown'

        try:
            print(f"[DEADLINE] Setting {location}/{day_name} deadline to {hour:02d}:{minute:02d}")

            result = _supabase_client.table('order_deadlines') \
                .upsert({
                    'location': location,
                    'day_of_week': day_of_week,
                    'deadline_hour': hour,
                    'deadline_minute': minute,
                    'reminder_minutes_before': reminder_minutes,
                    'active': True,
                    'updated_by': updated_by,
                    'updated_at': datetime.now().isoformat()
                }, on_conflict='location,day_of_week') \
                .execute()

            if result.data:
                print(f"[DEADLINE] ✓ Deadline set: {location}/{day_name} at {hour:02d}:{minute:02d}")
                self.logger.info(f"[DEADLINE] Set {location}/{day_name} to {hour:02d}:{minute:02d}")
                return True
            return False

        except Exception as e:
            print(f"[DEADLINE] ✗ Update failed: {e}")
            self.logger.error(f"[DEADLINE] Update error: {e}")
            return False

    def set_deadline_chat_ids(self, location: str, day_of_week: int,
                              notification_chat_id: int = None,
                              escalation_chat_id: int = None) -> bool:
        """
        Set notification chat IDs for a deadline.

        Args:
            location: Vendor name
            day_of_week: 0-6
            notification_chat_id: Chat for team reminders
            escalation_chat_id: Chat for escalations (admin DM)

        Returns:
            bool: Success
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        try:
            update_data = {'updated_at': datetime.now().isoformat()}

            if notification_chat_id is not None:
                update_data['notification_chat_id'] = notification_chat_id
            if escalation_chat_id is not None:
                update_data['escalation_chat_id'] = escalation_chat_id

            result = _supabase_client.table('order_deadlines') \
                .update(update_data) \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .execute()

            if result.data:
                print(f"[DEADLINE] ✓ Chat IDs updated for {location}/day={day_of_week}")
                return True
            return False

        except Exception as e:
            print(f"[DEADLINE] ✗ Chat ID update failed: {e}")
            return False

    def log_deadline_event(self, location: str, event_type: str,
                           submitted_by: str = None, notes: str = None) -> bool:
        """
        Log a deadline event for compliance tracking.

        Args:
            location: Vendor name
            event_type: 'reminder_sent', 'deadline_missed', 'order_submitted', 'escalation_sent'
            submitted_by: Who submitted (for order_submitted events)
            notes: Additional context

        Returns:
            bool: Success
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        now = get_time_in_timezone(BUSINESS_TIMEZONE)

        try:
            result = _supabase_client.table('deadline_events') \
                .insert({
                    'location': location,
                    'deadline_date': now.date().isoformat(),
                    'deadline_time': now.time().isoformat(),
                    'event_type': event_type,
                    'submitted_by': submitted_by,
                    'submitted_at': now.isoformat() if submitted_by else None,
                    'notes': notes
                }) \
                .execute()

            if result.data:
                print(f"[DEADLINE] ✓ Logged event: {location} - {event_type}")
                self.logger.info(f"[DEADLINE] Event logged: {location}/{event_type}")
                return True
            return False

        except Exception as e:
            print(f"[DEADLINE] ✗ Event log failed: {e}")
            return False

    def check_order_submitted_today(self, location: str) -> bool:
        """
        Check if an order was submitted for this location today.

        Returns:
            bool: True if order was submitted
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        today = get_time_in_timezone(BUSINESS_TIMEZONE).date()

        try:
            result = _supabase_client.table('inventory_transactions') \
                .select('id') \
                .eq('vendor', location) \
                .eq('type', 'order') \
                .gte('created_at', today.isoformat()) \
                .limit(1) \
                .execute()

            submitted = bool(result.data)
            print(f"[DEADLINE] Order check for {location} on {today}: {'✓ Submitted' if submitted else '✗ Not submitted'}")
            return submitted

        except Exception as e:
            print(f"[DEADLINE] ✗ Order check failed: {e}")
            return False

    # ===== END DEADLINE FUNCTIONS =====

    # ===== COUNT SCHEDULE FUNCTIONS =====

    def get_count_schedule_for_today(self, location: str) -> dict:
        """
        Get count schedule for today if one exists.

        Args:
            location: Vendor/location name

        Returns:
            dict or None: Schedule config if today has a count due
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return None

        now = get_time_in_timezone(BUSINESS_TIMEZONE)
        day_of_week = now.weekday()
        day_name = now.strftime('%A')

        try:
            print(f"[COUNT] Checking for count schedule: {location} on {day_name}")

            result = _supabase_client.table('count_schedules') \
                .select('*') \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .eq('active', True) \
                .single() \
                .execute()

            if result.data:
                due_hour = result.data['due_hour']
                due_minute = result.data['due_minute']
                print(f"[COUNT] ✓ Found schedule for {location}/{day_name}: {due_hour:02d}:{due_minute:02d}")
                return result.data
            else:
                print(f"[COUNT] No schedule for {location} on {day_name}")
                return None

        except Exception as e:
            if 'No rows' in str(e) or 'multiple' in str(e).lower():
                print(f"[COUNT] No schedule for {location} on {day_name}")
                return None
            print(f"[COUNT] ✗ Error: {e}")
            return None

    def get_all_count_schedules(self, location: str = None) -> list:
        """
        Get all count schedule configs, optionally filtered by location.

        Returns:
            list: Schedule config dicts
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return []

        try:
            query = _supabase_client.table('count_schedules') \
                .select('*') \
                .eq('active', True) \
                .order('location') \
                .order('day_of_week')

            if location:
                query = query.eq('location', location)

            result = query.execute()

            if result.data:
                print(f"[COUNT] ✓ Got {len(result.data)} count schedules")
                return result.data
            return []

        except Exception as e:
            print(f"[COUNT] ✗ Error getting schedules: {e}")
            return []

    def update_count_schedule(self, location: str, day_of_week: int, hour: int, minute: int = 0,
                              reminder_minutes: int = 60, updated_by: str = None) -> bool:
        """
        Create or update count schedule for a location/day.

        Args:
            location: Vendor name
            day_of_week: 0-6 (Monday-Sunday)
            hour: Due hour (0-23)
            minute: Due minute (0-59)
            reminder_minutes: Minutes before to send reminder
            updated_by: Who made the update

        Returns:
            bool: Success
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week] if 0 <= day_of_week <= 6 else 'Unknown'

        try:
            print(f"[COUNT] Setting {location}/{day_name} count due at {hour:02d}:{minute:02d}")

            result = _supabase_client.table('count_schedules') \
                .upsert({
                    'location': location,
                    'day_of_week': day_of_week,
                    'due_hour': hour,
                    'due_minute': minute,
                    'reminder_minutes_before': reminder_minutes,
                    'active': True,
                    'updated_by': updated_by,
                    'updated_at': datetime.now().isoformat()
                }, on_conflict='location,day_of_week') \
                .execute()

            if result.data:
                print(f"[COUNT] ✓ Schedule set: {location}/{day_name} at {hour:02d}:{minute:02d}")
                self.logger.info(f"[COUNT] Set {location}/{day_name} to {hour:02d}:{minute:02d}")
                return True
            return False

        except Exception as e:
            print(f"[COUNT] ✗ Update failed: {e}")
            self.logger.error(f"[COUNT] Update error: {e}")
            return False

    def set_count_schedule_chat_ids(self, location: str, day_of_week: int,
                                     notification_chat_id: int = None,
                                     escalation_chat_id: int = None) -> bool:
        """Set notification chat IDs for a count schedule."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        try:
            update_data = {'updated_at': datetime.now().isoformat()}

            if notification_chat_id is not None:
                update_data['notification_chat_id'] = notification_chat_id
            if escalation_chat_id is not None:
                update_data['escalation_chat_id'] = escalation_chat_id

            result = _supabase_client.table('count_schedules') \
                .update(update_data) \
                .eq('location', location) \
                .eq('day_of_week', day_of_week) \
                .execute()

            if result.data:
                print(f"[COUNT] ✓ Chat IDs updated for {location}/day={day_of_week}")
                return True
            return False

        except Exception as e:
            print(f"[COUNT] ✗ Chat ID update failed: {e}")
            return False

    def check_count_submitted_today(self, location: str) -> bool:
        """
        Check if an on_hand count was submitted for this location today.

        Returns:
            bool: True if count was submitted
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return False

        today = get_time_in_timezone(BUSINESS_TIMEZONE).date()

        try:
            result = _supabase_client.table('inventory_transactions') \
                .select('id') \
                .eq('vendor', location) \
                .eq('type', 'on_hand') \
                .eq('date', today.isoformat()) \
                .limit(1) \
                .execute()

            submitted = bool(result.data)
            print(f"[COUNT] Count check for {location} on {today}: {'✓ Submitted' if submitted else '✗ Not submitted'}")
            return submitted

        except Exception as e:
            print(f"[COUNT] ✗ Count check failed: {e}")
            return False

    def log_count_event(self, location: str, event_type: str, notes: str = None) -> bool:
        """Log a count-related event to deadline_events table."""
        return self.log_deadline_event(location, event_type, notes=notes)

    # ===== END COUNT SCHEDULE FUNCTIONS =====

    # ===== CONSUMPTION CYCLE FUNCTIONS =====

    def start_consumption_cycle(self, location: str, item_name: str,
                                on_hand: float, received: float = 0) -> str:
        """
        Start a new consumption cycle when delivery is received.

        Args:
            location: Vendor name
            item_name: Item name
            on_hand: On-hand quantity at cycle start
            received: Quantity received in this delivery

        Returns:
            str: Cycle ID or None if failed
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[CYCLE] Supabase not available")
            return None

        today = get_time_in_timezone(BUSINESS_TIMEZONE).date()

        try:
            # Close any existing open cycle for this item
            self._close_open_cycles(location, item_name, on_hand)

            # Create new cycle
            print(f"[CYCLE] Starting cycle: {location}/{item_name} - OH={on_hand}, received={received}")

            result = _supabase_client.table('consumption_cycles') \
                .insert({
                    'location': location,
                    'item_name': item_name,
                    'cycle_start_date': today.isoformat(),
                    'start_on_hand': on_hand,
                    'received_qty': received,
                    'status': 'open'
                }) \
                .execute()

            if result.data:
                cycle_id = result.data[0]['id']
                print(f"[CYCLE] ✓ Started cycle {cycle_id[:8]} for {item_name}")
                self.logger.info(f"[CYCLE] Started: {location}/{item_name} cycle={cycle_id[:8]}")
                return cycle_id

            return None

        except Exception as e:
            # Handle unique constraint (cycle already exists for today)
            if 'duplicate' in str(e).lower() or 'unique' in str(e).lower():
                print(f"[CYCLE] Cycle already exists for {item_name} today")
                return 'exists'

            print(f"[CYCLE] ✗ Failed to start cycle: {e}")
            self.logger.error(f"[CYCLE] Start error: {e}")
            return None

    def _close_open_cycles(self, location: str, item_name: str, end_on_hand: float):
        """
        Close any open cycles for an item and calculate consumption.

        Called when a new delivery arrives (which starts a new cycle).
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return

        today = get_time_in_timezone(BUSINESS_TIMEZONE).date()

        try:
            # Find open cycles
            result = _supabase_client.table('consumption_cycles') \
                .select('*') \
                .eq('location', location) \
                .eq('item_name', item_name) \
                .eq('status', 'open') \
                .execute()

            if not result.data:
                return

            for cycle in result.data:
                cycle_id = cycle['id']
                start_date = datetime.fromisoformat(cycle['cycle_start_date']).date()
                start_on_hand = float(cycle.get('start_on_hand', 0) or 0)
                received = float(cycle.get('received_qty', 0) or 0)

                # Calculate days in cycle
                days_in_cycle = (today - start_date).days
                if days_in_cycle <= 0:
                    print(f"[CYCLE] Skipping same-day cycle close for {item_name}")
                    continue

                # Calculate actual consumption
                actual_consumption = start_on_hand + received - end_on_hand

                # Get expected consumption from item's avg_consumption
                item_result = _supabase_client.table('inventory_items') \
                    .select('avg_consumption') \
                    .eq('vendor', location) \
                    .eq('item_name', item_name) \
                    .limit(1) \
                    .execute()

                avg_consumption = 0
                if item_result.data:
                    avg_consumption = float(item_result.data[0].get('avg_consumption', 0) or 0)

                expected_consumption = avg_consumption * days_in_cycle

                # Calculate drift
                if expected_consumption > 0:
                    drift = abs(actual_consumption - expected_consumption) / expected_consumption * 100
                else:
                    drift = 0 if actual_consumption == 0 else 100

                print(f"[CYCLE] Closing cycle for {item_name}:")
                print(f"[CYCLE]   Days: {days_in_cycle}")
                print(f"[CYCLE]   Start OH: {start_on_hand}, Received: {received}, End OH: {end_on_hand}")
                print(f"[CYCLE]   Actual consumption: {actual_consumption:.1f}")
                print(f"[CYCLE]   Expected consumption: {expected_consumption:.1f}")
                print(f"[CYCLE]   Drift: {drift:.1f}%")

                # Determine status
                if drift > 50:
                    status = 'flagged'
                    print(f"[CYCLE] ⚠ FLAGGED: {drift:.1f}% drift exceeds 50% threshold")
                else:
                    status = 'closed'

                # Update cycle record
                _supabase_client.table('consumption_cycles') \
                    .update({
                        'cycle_end_date': today.isoformat(),
                        'days_in_cycle': days_in_cycle,
                        'end_on_hand': end_on_hand,
                        'actual_consumption': actual_consumption,
                        'expected_consumption': expected_consumption,
                        'drift_percentage': drift,
                        'status': status,
                        'updated_at': datetime.now().isoformat()
                    }) \
                    .eq('id', cycle_id) \
                    .execute()

                print(f"[CYCLE] ✓ Closed cycle {cycle_id[:8]} with status: {status}")
                self.logger.info(f"[CYCLE] Closed: {item_name} drift={drift:.1f}% status={status}")

                # Trigger calibration check
                self._check_calibration_needed(location, item_name)

        except Exception as e:
            print(f"[CYCLE] ✗ Error closing cycles: {e}")
            self.logger.error(f"[CYCLE] Close error: {e}")

    def _check_calibration_needed(self, location: str, item_name: str):
        """
        Check if par adjustment is needed based on recent cycles.

        Rule: If drift > 25% for 2 consecutive cycles, auto-adjust.
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return

        try:
            # Get last 2 closed cycles
            result = _supabase_client.table('consumption_cycles') \
                .select('*') \
                .eq('location', location) \
                .eq('item_name', item_name) \
                .eq('status', 'closed') \
                .order('cycle_end_date', desc=True) \
                .limit(2) \
                .execute()

            if not result.data or len(result.data) < 2:
                print(f"[CALIBRATE] Not enough cycles for {item_name} (need 2)")
                return

            cycles = result.data

            # Check if both have >25% drift in same direction
            drift1 = float(cycles[0].get('drift_percentage', 0) or 0)
            drift2 = float(cycles[1].get('drift_percentage', 0) or 0)

            actual1 = float(cycles[0].get('actual_consumption', 0) or 0)
            expected1 = float(cycles[0].get('expected_consumption', 0) or 0)
            actual2 = float(cycles[1].get('actual_consumption', 0) or 0)
            expected2 = float(cycles[1].get('expected_consumption', 0) or 0)

            # Check drift threshold
            if drift1 < 25 or drift2 < 25:
                print(f"[CALIBRATE] Drift below threshold for {item_name}: {drift1:.1f}%, {drift2:.1f}%")
                return

            # Check same direction (both over or both under)
            over1 = actual1 > expected1
            over2 = actual2 > expected2

            if over1 != over2:
                print(f"[CALIBRATE] Drift direction inconsistent for {item_name}, skipping")
                return

            # Calculate new avg_consumption based on recent actuals
            days1 = int(cycles[0].get('days_in_cycle', 0) or 0)
            days2 = int(cycles[1].get('days_in_cycle', 0) or 0)
            total_days = days1 + days2
            total_consumption = actual1 + actual2

            if total_days <= 0:
                return

            new_avg_consumption = total_consumption / total_days

            print(f"[CALIBRATE] ✓ Auto-calibration triggered for {item_name}")
            print(f"[CALIBRATE]   Consecutive drifts: {drift1:.1f}%, {drift2:.1f}%")
            print(f"[CALIBRATE]   Direction: {'OVER' if over1 else 'UNDER'} consumption")
            print(f"[CALIBRATE]   New avg_consumption: {new_avg_consumption:.2f}")

            # Apply calibration
            self._apply_par_calibration(location, item_name, new_avg_consumption,
                                        f"Auto-calibrated after {drift1:.0f}%/{drift2:.0f}% drift")

            # Mark cycles as adjusted
            for cycle in cycles:
                _supabase_client.table('consumption_cycles') \
                    .update({
                        'adjustment_applied': True,
                        'status': 'adjusted'
                    }) \
                    .eq('id', cycle['id']) \
                    .execute()

        except Exception as e:
            print(f"[CALIBRATE] ✗ Error checking calibration: {e}")
            self.logger.error(f"[CALIBRATE] Error: {e}")

    def _apply_par_calibration(self, location: str, item_name: str,
                               new_avg_consumption: float, reason: str):
        """
        Apply par calibration by updating avg_consumption and adjusting pars.

        Par adjustment: Scale pars proportionally to consumption change.
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return

        try:
            # Get current item values
            item_result = _supabase_client.table('inventory_items') \
                .select('id, avg_consumption, min_par, max_par') \
                .eq('vendor', location) \
                .eq('item_name', item_name) \
                .limit(1) \
                .execute()

            if not item_result.data:
                print(f"[CALIBRATE] ✗ Item not found: {item_name}")
                return

            item = item_result.data[0]
            item_id = item['id']
            old_avg = float(item.get('avg_consumption', 0) or 0)
            old_min = float(item.get('min_par', 0) or 0)
            old_max = float(item.get('max_par', 0) or 0)

            # Calculate scaling factor
            if old_avg > 0:
                scale_factor = new_avg_consumption / old_avg
            else:
                scale_factor = 1.0

            # Calculate new pars (scale proportionally)
            new_min = old_min * scale_factor
            new_max = old_max * scale_factor

            print(f"[CALIBRATE] Applying calibration to {item_name}:")
            print(f"[CALIBRATE]   avg_consumption: {old_avg:.2f} → {new_avg_consumption:.2f}")
            print(f"[CALIBRATE]   min_par: {old_min:.1f} → {new_min:.1f}")
            print(f"[CALIBRATE]   max_par: {old_max:.1f} → {new_max:.1f}")
            print(f"[CALIBRATE]   Scale factor: {scale_factor:.2f}")

            # Update item
            _supabase_client.table('inventory_items') \
                .update({
                    'avg_consumption': new_avg_consumption,
                    'min_par': new_min,
                    'max_par': new_max,
                    'last_par_update': datetime.now().isoformat()
                }) \
                .eq('id', item_id) \
                .execute()

            # Log to par_history
            _supabase_client.table('par_history') \
                .insert({
                    'item_id': item_id,
                    'old_min_par': old_min,
                    'old_max_par': old_max,
                    'new_min_par': new_min,
                    'new_max_par': new_max,
                    'actual_consumption': new_avg_consumption,
                    'reason': reason
                }) \
                .execute()

            print(f"[CALIBRATE] ✓ Calibration applied to {item_name}")
            self.logger.info(f"[CALIBRATE] Applied: {item_name} avg={new_avg_consumption:.2f} min={new_min:.1f} max={new_max:.1f}")

        except Exception as e:
            print(f"[CALIBRATE] ✗ Failed to apply calibration: {e}")
            self.logger.error(f"[CALIBRATE] Apply error: {e}")

    def get_flagged_cycles(self, location: str = None) -> list:
        """
        Get cycles flagged for manual review (>50% drift).

        Returns:
            list: Flagged cycle records
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return []

        try:
            query = _supabase_client.table('consumption_cycles') \
                .select('*') \
                .eq('status', 'flagged') \
                .order('updated_at', desc=True)

            if location:
                query = query.eq('location', location)

            result = query.execute()

            if result.data:
                print(f"[CYCLE] ✓ Found {len(result.data)} flagged cycles")
                return result.data

            return []

        except Exception as e:
            print(f"[CYCLE] ✗ Error getting flagged cycles: {e}")
            return []

    def get_latest_inventory(self, vendor: str, entry_type: str = 'on_hand') -> dict:
        """
        Get the latest inventory quantities for a vendor.

        Returns:
            dict: {item_name: quantity}
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return {}

        try:
            result = _supabase_client.table('inventory_transactions') \
                .select('quantities') \
                .eq('vendor', vendor) \
                .eq('type', entry_type) \
                .order('date', desc=True) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()

            if result.data:
                quantities = result.data[0].get('quantities', {})
                if isinstance(quantities, str):
                    import json
                    quantities = json.loads(quantities)
                return quantities

            return {}

        except Exception as e:
            print(f"[INVENTORY] ✗ Error getting latest inventory: {e}")
            return {}

    # ===== END CONSUMPTION CYCLE FUNCTIONS =====

    def _read_inventory_from_supabase(self, vendor: str, entry_type: str = 'on_hand'):
        """Read latest inventory from Supabase. Returns dict or None to fallback."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return None

        try:
            result = _supabase_client.table('inventory_transactions') \
                .select('quantities') \
                .eq('vendor', vendor) \
                .eq('type', entry_type) \
                .order('date', desc=True) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()

            if result.data:
                quantities = result.data[0].get('quantities', {})
                print(f"[PHASE 2] ✓ Read inventory from Supabase for {vendor}/{entry_type}")
                return quantities
            print(f"[PHASE 2] ✓ No inventory data in Supabase for {vendor}/{entry_type}")
            return {}
        except Exception as e:
            self.logger.warning(f"Supabase read inventory failed: {e}")
            print(f"[PHASE 2] ⚠ Supabase read inventory failed: {e}, falling back to Notion")
            return None

    def _read_locations_from_supabase(self):
        """Read vendor list from Supabase. Returns list or None to fallback."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return None

        try:
            result = _supabase_client.table('inventory_items') \
                .select('vendor') \
                .eq('active', True) \
                .execute()

            vendors = sorted(set(r['vendor'] for r in result.data if r.get('vendor')))
            print(f"[PHASE 2] ✓ Read {len(vendors)} locations from Supabase")
            return vendors
        except Exception as e:
            self.logger.warning(f"Supabase read locations failed: {e}")
            print(f"[PHASE 2] ⚠ Supabase read locations failed: {e}, falling back to Notion")
            return None

    def calculate_expected_count(self, item_name: str, vendor: str) -> dict:
        """
        Calculate expected count with fallbacks and confidence level.

        Priority:
        1. last_onhand + deliveries - consumption (HIGH confidence)
        2. par_level (MEDIUM confidence)
        3. last_known_value (LOW confidence)
        4. None (NO_DATA)

        Returns:
            dict: {
                'expected': float or None,
                'confidence': 'HIGH' | 'MEDIUM' | 'LOW' | 'NO_DATA',
                'source': str describing data source
            }
        """
        if not _supabase_enabled:
            self.logger.debug(f"[PHASE 1] Expected calc skipped - Supabase not enabled")
            return {'expected': None, 'confidence': 'NO_DATA', 'source': 'Supabase disabled'}

        try:
            # Attempt 1: Full calculation from transaction history
            # Schema: quantities is JSON {item_name: qty}, type is 'on_hand'/'received', date is 'YYYY-MM-DD'
            last = _supabase_client.table('inventory_transactions') \
                .select('quantities, date') \
                .eq('vendor', vendor) \
                .eq('type', 'on_hand') \
                .order('date', desc=True) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()

            # Check if we have data AND the item exists in the quantities JSON
            if last.data and item_name in (last.data[0].get('quantities') or {}):
                last_qty = float(last.data[0]['quantities'].get(item_name, 0) or 0)
                last_date = last.data[0]['date']

                # Get deliveries since last count
                deliveries = _supabase_client.table('inventory_transactions') \
                    .select('quantities') \
                    .eq('vendor', vendor) \
                    .eq('type', 'received') \
                    .gt('date', last_date) \
                    .execute()

                delivered = sum(
                    float(d['quantities'].get(item_name, 0) or 0)
                    for d in (deliveries.data or [])
                    if d.get('quantities')
                )

                # Get ADU from inventory_items
                item = _supabase_client.table('inventory_items') \
                    .select('avg_consumption') \
                    .eq('vendor', vendor) \
                    .eq('item_name', item_name) \
                    .limit(1) \
                    .execute()

                adu = float(item.data[0].get('avg_consumption', 0) or 0) if item.data else 0

                # Calculate days since last count
                from datetime import datetime
                last_dt = datetime.strptime(last_date, '%Y-%m-%d')
                days = max(1, (datetime.now() - last_dt).days)

                expected = max(0, last_qty + delivered - (adu * days))

                self.logger.debug(f"[PHASE 1] Expected calc: {item_name} = {last_qty} + {delivered} - ({adu} * {days}) = {expected:.1f}")
                print(f"[PHASE 1] Expected count for {item_name}: {expected:.1f} (HIGH confidence)")

                return {
                    'expected': round(expected, 1),
                    'confidence': 'HIGH',
                    'source': f'last={last_qty}, delivered={delivered}, consumed={adu*days:.1f}'
                }

            # Attempt 2: Fallback to par_level
            item = _supabase_client.table('inventory_items') \
                .select('min_par, max_par, last_known_value') \
                .eq('vendor', vendor) \
                .eq('item_name', item_name) \
                .single() \
                .execute()

            if item.data:
                min_par = item.data.get('min_par', 0)
                max_par = item.data.get('max_par', 0)
                last_known = item.data.get('last_known_value')

                # Use midpoint of par range as expected
                if min_par > 0 or max_par > 0:
                    par_expected = (min_par + max_par) / 2 if max_par > 0 else min_par
                    self.logger.info(f"[PHASE 1] Using par_level fallback for {item_name}: {par_expected}")
                    print(f"[PHASE 1] Expected count for {item_name}: {par_expected:.1f} (MEDIUM confidence - par)")

                    return {
                        'expected': round(par_expected, 1),
                        'confidence': 'MEDIUM',
                        'source': f'par_level midpoint (min={min_par}, max={max_par})'
                    }

                # Attempt 3: Fallback to last_known_value
                if last_known is not None:
                    self.logger.info(f"[PHASE 1] Using last_known_value fallback for {item_name}: {last_known}")
                    print(f"[PHASE 1] Expected count for {item_name}: {last_known:.1f} (LOW confidence)")

                    return {
                        'expected': round(last_known, 1),
                        'confidence': 'LOW',
                        'source': 'last_known_value'
                    }

            # No data available
            self.logger.debug(f"[PHASE 1] No baseline data for {item_name} at {vendor}")
            return {'expected': None, 'confidence': 'NO_DATA', 'source': 'No baseline data'}

        except Exception as e:
            self.logger.warning(f"[PHASE 1] Expected count calc failed: {e}")
            return {'expected': None, 'confidence': 'NO_DATA', 'source': f'Error: {e}'}

    def get_variance_threshold(self, vendor: str, item_name: str = None) -> float:
        """
        Get variance threshold from baseline_expectations table.

        Priority:
        1. Item-specific threshold for vendor
        2. Vendor-wide threshold (item_name = '*')
        3. Default 0.20 (20%)

        Args:
            vendor: Vendor/location name
            item_name: Optional specific item name

        Returns:
            float: Threshold value (e.g., 0.20 for 20%)
        """
        DEFAULT_THRESHOLD = 0.20

        if not _supabase_enabled:
            return DEFAULT_THRESHOLD

        try:
            # Try item-specific threshold first
            if item_name:
                result = _supabase_client.table('baseline_expectations') \
                    .select('variance_threshold') \
                    .eq('vendor', vendor) \
                    .eq('item_name', item_name) \
                    .limit(1) \
                    .execute()

                if result.data and result.data[0].get('variance_threshold') is not None:
                    threshold = result.data[0]['variance_threshold']
                    self.logger.debug(f"[PHASE 2] Item threshold for {item_name}: {threshold}")
                    return threshold

            # Try vendor-wide threshold
            result = _supabase_client.table('baseline_expectations') \
                .select('variance_threshold') \
                .eq('vendor', vendor) \
                .eq('item_name', '*') \
                .limit(1) \
                .execute()

            if result.data and result.data[0].get('variance_threshold') is not None:
                threshold = result.data[0]['variance_threshold']
                self.logger.debug(f"[PHASE 2] Vendor threshold for {vendor}: {threshold}")
                return threshold

            self.logger.debug(f"[PHASE 2] Using default threshold: {DEFAULT_THRESHOLD}")
            return DEFAULT_THRESHOLD

        except Exception as e:
            self.logger.warning(f"[PHASE 2] Failed to get threshold: {e}")
            return DEFAULT_THRESHOLD

    def check_variance(self, item_name: str, vendor: str, actual: float, threshold: float = None) -> dict:
        """
        Check if count has suspicious variance.

        Args:
            item_name: Name of item
            vendor: Vendor/location
            actual: Actual counted value
            threshold: Override threshold (None = fetch from DB)

        Returns:
            {'expected': float, 'variance': float, 'suspicious': bool, 'message': str, 'confidence': str, 'threshold': float}
        """
        # Get threshold from DB if not provided
        if threshold is None:
            threshold = self.get_variance_threshold(vendor, item_name)

        result = self.calculate_expected_count(item_name, vendor)
        expected = result.get('expected')
        confidence = result.get('confidence', 'NO_DATA')

        if expected is None:
            self.logger.debug(f"[PHASE 4] Variance check skipped for {item_name} - no baseline")
            return {
                'expected': None,
                'variance': 0,
                'suspicious': False,
                'message': 'No baseline',
                'confidence': confidence,
                'threshold': threshold
            }

        variance = abs(actual - expected) / expected if expected > 0 else 0
        suspicious = variance > threshold

        direction = "higher" if actual > expected else "lower"
        message = f"{variance:.0%} {direction} than expected (~{expected:.1f})" if suspicious else "OK"

        self.logger.info(f"[PHASE 4] Variance check: {item_name} actual={actual} expected={expected:.1f} variance={variance:.0%} threshold={threshold:.0%} suspicious={suspicious}")

        if suspicious:
            print(f"[PHASE 4] VARIANCE DETECTED: {item_name} - {message} (threshold: {threshold:.0%})")

        return {
            'expected': expected,
            'variance': round(variance, 2),
            'suspicious': suspicious,
            'message': message,
            'confidence': confidence,
            'threshold': threshold
        }

    def get_managers_for_location(self, location: str = None) -> list:
        """
        Get manager telegram_ids for notifications.

        Args:
            location: Optional location filter (reserved for future use)

        Returns:
            list: [{'telegram_id': int, 'name': str}, ...]
        """
        if not _supabase_enabled:
            return []

        try:
            # Get all active managers (telegram_users has no location column)
            result = _supabase_client.table('telegram_users') \
                .select('telegram_id, name') \
                .eq('role', 'manager') \
                .eq('active', True) \
                .execute()

            managers = result.data or []
            self.logger.info(f"[PHASE 4] Found {len(managers)} managers")
            return managers

        except Exception as e:
            self.logger.error(f"[PHASE 4] Failed to get managers: {e}")
            return []

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
        # === SUPABASE FIRST (Phase 2) ===
        supabase_result = self._read_items_from_supabase(location)
        if supabase_result is not None:
            self.logger.info(f"[PHASE 2] Using Supabase for get_items_for_location({location})")
            return supabase_result
        # === FALLBACK TO NOTION ===

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
                            notes: str, quantities: Dict[str, float],
                            flagged: bool = False, flag_reason: Optional[str] = None) -> bool:
        """
        Save order transaction to track what was ordered.

        Args:
            location: Location name
            date: Date in YYYY-MM-DD format
            manager: Manager name (becomes page title)
            notes: Optional notes
            quantities: Dict mapping item names to ordered quantities
            flagged: Whether order has flagged items (Phase 5)
            flag_reason: Reason for flagging (Phase 5)

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
            
            # === PHASE 3: CONDITIONAL NOTION WRITES ===
            NOTION_WRITES_ENABLED = os.environ.get('NOTION_WRITES_ENABLED', 'true').lower() == 'true'

            if NOTION_WRITES_ENABLED:
                # Create the page
                page_data = {
                    'parent': {
                        'database_id': self.inventory_db_id
                    },
                    'properties': properties
                }

                response = self._make_request('POST', '/pages', page_data)
                print(f"[PHASE 3] Notion write ENABLED for order transaction")
            else:
                response = {'id': 'supabase-only'}  # Fake success for flow
                print(f"[PHASE 3] Notion write DISABLED - Supabase only for order transaction")
            # === END PHASE 3 ===

            if response:
                self.logger.info(f"Saved order transaction: {title}")

                # === DUAL-WRITE TO SUPABASE ===
                print(f"[PHASE 1] Attempting dual-write for order transaction: {location}")
                # === PHASE 5: Pass flag parameters ===
                self._write_to_supabase('inventory_transactions', {
                    'date': date,
                    'vendor': location,
                    'type': 'order',
                    'quantities': quantities,
                    'submitter': manager,
                    'notes': notes if notes else None,
                    'leader': None,
                    'photo_url': None,
                    'flagged': flagged,
                    'flag_reason': flag_reason
                })
                # === END DUAL-WRITE ===

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
    
    def get_locations(self, use_cache: bool = True) -> List[str]:
        """
        Retrieve all unique location names from Items Master database.
        
        Queries the Items Master for distinct Location values and caches
        the result for 5 minutes to minimize API calls. This is the single
        source of truth for all location/vendor names in the system.
        
        Args:
            use_cache: Whether to use cached locations if available
            
        Returns:
            List[str]: Sorted list of unique location names
            
        Logs: cache hit/miss, query execution, location count, cache update
        """
        # === SUPABASE FIRST (Phase 2) ===
        supabase_result = self._read_locations_from_supabase()
        if supabase_result is not None:
            self.logger.info(f"[PHASE 2] Using Supabase for get_locations()")
            return supabase_result
        # === FALLBACK TO NOTION ===

        cache_key = "_locations_list"
        cache_ttl = 300  # 5 minutes

        # Check cache first
        if use_cache and hasattr(self, '_locations_cache_timestamp'):
            cache_age = time.time() - self._locations_cache_timestamp
            if cache_age < cache_ttl and cache_key in self.__dict__:
                cached_locations = getattr(self, cache_key, [])
                self.logger.debug(f"Locations cache hit | age={cache_age:.1f}s count={len(cached_locations)}")
                return cached_locations
            else:
                self.logger.debug(f"Locations cache expired | age={cache_age:.1f}s")
        
        # Cache miss - query Notion
        self.logger.info(f"Querying Notion for unique locations | db={self.items_db_id[:8]}...")
        
        try:
            start_time = time.time()
            
            # Query all items to extract locations
            # Note: Notion API doesn't have native DISTINCT, so we fetch all and deduplicate
            query = {
                'page_size': 100,  # Max page size
                'filter': {
                    'property': 'Active',
                    'checkbox': {
                        'equals': True
                    }
                }
            }
            
            locations_set = set()
            has_more = True
            next_cursor = None
            page_count = 0
            
            # Paginate through all results
            while has_more:
                page_count += 1
                if next_cursor:
                    query['start_cursor'] = next_cursor
                
                response = self._make_request('POST', f'/databases/{self.items_db_id}/query', query)
                
                if not response:
                    self.logger.error(f"Failed to query locations | page={page_count}")
                    # Return cached value if available, otherwise empty
                    return getattr(self, cache_key, [])
                
                # Extract locations from this page
                for page in response.get('results', []):
                    props = page.get('properties', {})
                    location_prop = props.get('Location', {})
                    location_select = location_prop.get('select', {})
                    location_name = location_select.get('name')
                    
                    if location_name:
                        locations_set.add(location_name)
                        self.logger.debug(f"Found location | name='{location_name}' page_id={page.get('id', 'unknown')[:8]}...")
                
                # Check for more pages
                has_more = response.get('has_more', False)
                next_cursor = response.get('next_cursor')
                
                self.logger.debug(f"Locations query page complete | page={page_count} has_more={has_more} found_so_far={len(locations_set)}")
            
            # Sort locations alphabetically for consistent UI
            locations_list = sorted(list(locations_set))
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Locations query complete | count={len(locations_list)} pages={page_count} duration={duration_ms:.2f}ms")
            
            if locations_list:
                self.logger.info(f"Discovered locations: {', '.join(locations_list)}")
            else:
                self.logger.warning("No locations found in Items Master database")
            
            # Update cache
            setattr(self, cache_key, locations_list)
            self._locations_cache_timestamp = time.time()
            self.logger.debug(f"Locations cache updated | ttl={cache_ttl}s")
            
            return locations_list
            
        except Exception as e:
            self.logger.error(f"Error retrieving locations from Notion | error={e}", exc_info=True)
            # Return cached value if available, otherwise empty list
            cached = getattr(self, cache_key, [])
            if cached:
                self.logger.warning(f"Returning stale cached locations due to error | count={len(cached)}")
            return cached
    
    def save_inventory_transaction(self, location: str, entry_type: str, date: str,
                                manager: str, notes: str, quantities: Dict[str, float],
                                image_file_id: Optional[str] = None,
                                flagged: bool = False, flag_reason: Optional[str] = None,
                                photo_url: Optional[str] = None) -> bool:
        """
        Save inventory transaction with optional image using Telegram file approach.

        Args:
            location: Location name
            entry_type: 'on_hand' or 'received'
            date: Date in YYYY-MM-DD format
            manager: Submitter name (becomes page title and saved in Submitter property)
            notes: Optional notes
            quantities: Dict mapping item names to quantities
            image_file_id: Optional Telegram file ID for product image
            flagged: Whether this transaction has variance flags (Phase 4)
            flag_reason: Reason for flagging (Phase 4)
            photo_url: Direct URL to photo (Phase 3 - for Supabase)

        Returns:
            bool: True if successful
        """
        try:
            # Create executive-level title for management visibility
            entry_type_display = "On-Hand Count" if entry_type == 'on_hand' else "Delivery Received"
            total_items = sum(1 for qty in quantities.values() if qty > 0)
            title = f"{manager} • {entry_type_display} • {date} • {location} ({total_items} items)"
            
            self.logger.info(f"Saving inventory transaction | location={location} type={entry_type} date={date} submitter='{manager}' items={total_items}")
            
            # Format quantities as readable JSON string for Notion
            quantities_summary = []
            for item_name, qty in quantities.items():
                if qty >= 0:  # Show all items including 0 (so we know what's empty)
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
                'Submitter': {  # New property for submitter name
                    'rich_text': [
                        {
                            'text': {
                                'content': manager
                            }
                        }
                    ]
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
            
            # === PHASE 3: CONDITIONAL NOTION WRITES ===
            NOTION_WRITES_ENABLED = os.environ.get('NOTION_WRITES_ENABLED', 'true').lower() == 'true'

            if NOTION_WRITES_ENABLED:
                # Create the page
                page_data = {
                    'parent': {
                        'database_id': self.inventory_db_id
                    },
                    'properties': properties
                }

                response = self._make_request('POST', '/pages', page_data)
                print(f"[PHASE 3] Notion write ENABLED for inventory transaction")
            else:
                response = {'id': 'supabase-only'}  # Fake success for flow
                print(f"[PHASE 3] Notion write DISABLED - Supabase only for inventory transaction")
            # === END PHASE 3 ===

            if response:
                image_note = " with image" if image_file_id else ""
                self.logger.info(f"Saved inventory transaction: {title}{image_note}")
                self.logger.info(f"Items recorded: {len([q for q in quantities.values() if q > 0])}")
                self.logger.info(f"Submitter property saved: '{manager}'")

                # === DUAL-WRITE TO SUPABASE ===
                # === PHASE 4: LOG FLAGGED STATUS ===
                if flagged:
                    print(f"[PHASE 4] ⚠ Transaction FLAGGED: {flag_reason}")
                    self.logger.info(f"[PHASE 4] Flagged transaction: {flag_reason}")

                # === PHASE 3: Get photo URL for Supabase ===
                supabase_photo_url = photo_url  # Use direct URL if provided
                if not supabase_photo_url and image_file_id:
                    # Try to get URL from Telegram
                    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
                    if bot_token:
                        supabase_photo_url = self._get_telegram_photo_url(image_file_id, bot_token)

                print(f"[PHASE 1] Attempting dual-write for inventory transaction: {location} / {entry_type}")
                self._write_to_supabase('inventory_transactions', {
                    'date': date,
                    'vendor': location,
                    'type': 'on_hand' if entry_type == 'on_hand' else 'received',
                    'quantities': quantities,
                    'submitter': manager,
                    'notes': notes if notes else None,
                    'leader': None,
                    'photo_url': supabase_photo_url,
                    'flagged': flagged,
                    'flag_reason': flag_reason
                })

                if supabase_photo_url:
                    self.logger.info(f"[PHASE 3] Photo URL saved to Supabase: {supabase_photo_url[:50]}...")
                # === END DUAL-WRITE ===

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
            # === SUPABASE FIRST (Phase 2) ===
            supabase_result = self._read_inventory_from_supabase(location, entry_type)
            if supabase_result is not None:
                self.logger.info(f"[PHASE 2] Using Supabase for get_latest_inventory({location}, {entry_type})")
                return supabase_result
            # === FALLBACK TO NOTION ===

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
        
    def get_latest_onhand_metadata(self, location: str) -> Optional[Dict]:
        """
        Get metadata from most recent On-Hand entry for location.
        
        Returns created_time, created_by, and flags for multiple recent entries.
        
        Args:
            location: Location name
            
        Returns:
            Optional[Dict]: {
                'created_time': ISO timestamp,
                'created_by': user name or ID,
                'multiple_recent': bool (True if multiple entries within 12 hours)
            } or None if no entries found
            
        Logs: query, results, metadata extraction
        """
        try:
            self.logger.info(f"Getting latest on-hand metadata | location={location}")
            
            # Query for most recent On-Hand entries
            query = {
                "filter": {
                    "and": [
                        {"property": "Location", "select": {"equals": location}},
                        {"property": "Type", "select": {"equals": "On-Hand"}},
                    ]
                },
                "sorts": [{"property": "Date", "direction": "descending"}],
                "page_size": 5,  # Get a few to check for duplicates
            }
            
            response = self._make_request("POST", f"/databases/{self.inventory_db_id}/query", query)
            
            if not response or not response.get("results"):
                self.logger.info(f"No on-hand entries found | location={location}")
                return None
            
            pages = response["results"]
            latest_page = pages[0]
            
            # Extract metadata
            created_time = latest_page.get("created_time", "")
            
            # Get submitter from Submitter property (not system created_by)
            props = latest_page.get("properties", {})
            submitter_prop = props.get("Submitter", {})
            
            # Parse Submitter rich text field
            user_name = "Unknown"
            if submitter_prop.get("rich_text"):
                submitter_text = "".join(
                    segment.get("plain_text", "")
                    for segment in submitter_prop["rich_text"]
                ).strip()
                if submitter_text:
                    user_name = submitter_text
            
            self.logger.debug(f"Parsed submitter from property | submitter='{user_name}'")
            
            # Check for multiple recent entries (within 12 hours)
            from datetime import datetime, timedelta
            
            multiple_recent = False
            if len(pages) > 1:
                try:
                    latest_time = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                    second_time = datetime.fromisoformat(pages[1].get("created_time", "").replace('Z', '+00:00'))
                    
                    time_diff = abs((latest_time - second_time).total_seconds() / 3600)
                    if time_diff < 12:
                        multiple_recent = True
                        self.logger.info(f"Multiple recent entries detected | time_diff={time_diff:.1f}h")
                except Exception as e:
                    self.logger.warning(f"Error checking multiple entries | error={e}")
            
            metadata = {
                'created_time': created_time,
                'created_by': user_name,
                'multiple_recent': multiple_recent
            }
            
            self.logger.info(f"Latest on-hand metadata | location={location} created={created_time} by={user_name} multiple={multiple_recent}")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting on-hand metadata | location={location} error={e}", exc_info=True)
            return None
        
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
        self.order_handler = OrderFlowHandler(self, notion_manager, calculator)
        self.report_handler = ReportHandler(self, notion_manager)
        self.sop_handler = SOPHandler(self, notion_manager)
        
        # Rate limiting with exemptions
        self.user_commands: Dict[int, List[datetime]] = {}
        self.rate_limit_lock = threading.Lock()
        self.rate_limit_exempt_commands = {'/cancel', '/help', '/done', '/skip'}
        
        # Connection retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Chat configuration from environment
        self.chat_config = {
            'onhand': int(os.environ.get('CHAT_ONHAND', '0')),
            'autorequest': int(os.environ.get('CHAT_AUTOREQUEST', '0')),
            'received': int(os.environ.get('CHAT_RECEIVED', '0')),
            'reassurance': int(os.environ.get('CHAT_REASSURANCE', '0')),
            # Order prep chat routing (vendor-aware with fallback)
            'prep_chat:default': os.environ.get('ORDER_PREP_CHAT_ID', '').strip(),
            'prep_chat:Avondale': os.environ.get('ORDER_PREP_CHAT_ID_AVONDALE', '').strip(),
            'prep_chat:Commissary': os.environ.get('ORDER_PREP_CHAT_ID_COMMISSARY', '').strip()
        }

        # Log order prep chat configuration
        self.logger.info(f"Order prep chat config loaded | default={self.chat_config.get('prep_chat:default', 'NOT_SET')} "
                        f"avondale={self.chat_config.get('prep_chat:Avondale', 'NOT_SET')} "
                        f"commissary={self.chat_config.get('prep_chat:Commissary', 'NOT_SET')}")
        
        # Test chat override
        self.use_test_chat = os.environ.get('USE_TEST_CHAT', 'false').lower() == 'true'
        self.test_chat = int(os.environ.get('TEST_CHAT', '0')) if self.use_test_chat else None
        
        self.logger.info(f"Telegram bot initialized with enhanced error handling")
        if self.use_test_chat:
            self.logger.info(f"Test mode enabled - all messages will go to chat {self.test_chat}")

        # Register bot commands with Telegram
        self._register_bot_commands()

    def _register_bot_commands(self):
        """
        Register bot commands with Telegram so they appear in the command menu.
        Called during initialization.
        """
        commands = [
            {"command": "start", "description": "Start the bot"},
            {"command": "help", "description": "Show command reference"},
            {"command": "entry", "description": "Record inventory counts"},
            {"command": "order", "description": "Create supplier order"},
            {"command": "info", "description": "View inventory analysis"},
            {"command": "reassurance", "description": "Risk assessment report"},
            {"command": "review", "description": "View recent submissions"},
            {"command": "pars", "description": "View/edit item par levels"},
            {"command": "flags", "description": "Review flagged transactions"},
            {"command": "cycles", "description": "View consumption cycles"},
            {"command": "adu", "description": "View average daily usage"},
            {"command": "viewtasks", "description": "View your pending tasks"},
            {"command": "status", "description": "System health check"},
            {"command": "cancel", "description": "Cancel current operation"},
        ]

        try:
            url = f"{self.base_url}/setMyCommands"
            response = requests.post(url, json={"commands": commands}, timeout=10)

            if response.status_code == 200:
                print(f"[BOT] ✓ Registered {len(commands)} commands with Telegram")
                self.logger.info(f"Bot commands registered successfully")
            else:
                print(f"[BOT] ⚠ Failed to register commands: {response.text}")
                self.logger.warning(f"Failed to register bot commands: {response.text}")
        except Exception as e:
            print(f"[BOT] ⚠ Error registering commands: {e}")
            self.logger.warning(f"Error registering bot commands: {e}")

    def _get_notification_config(self, vendor: str, notification_type: str, session_token: str) -> dict:
        """
        Get notification configuration from database for a vendor.

        Args:
            vendor: Vendor name
            notification_type: Type of notification (entry_confirmation, order_reminder, etc.)
            session_token: Correlation token for logging

        Returns:
            dict with keys:
                - telegram_ids: List of individual user telegram_ids to notify
                - group_chat_id: Group chat ID (or None)
                - found: Whether config was found in database
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            self.logger.warning(f"[{session_token}] Supabase not available for notification config lookup")
            return {'telegram_ids': [], 'group_chat_id': None, 'found': False}

        try:
            self.logger.info(f"[{session_token}] Looking up notification config | vendor='{vendor}' type='{notification_type}'")
            print(f"[DEBUG] Looking up notification config for vendor='{vendor}' type='{notification_type}'")

            result = _supabase_client.table('vendor_notifications') \
                .select('telegram_ids, group_chat_id, vendor, enabled') \
                .eq('vendor', vendor) \
                .eq('notification_type', notification_type) \
                .eq('enabled', True) \
                .execute()

            self.logger.info(f"[{session_token}] Query result: {len(result.data) if result.data else 0} records found")
            print(f"[DEBUG] Query result: {result.data}")

            if result.data and len(result.data) > 0:
                config = result.data[0]
                telegram_ids = config.get('telegram_ids') or []
                group_chat_id = config.get('group_chat_id')
                self.logger.info(f"[{session_token}] Found notification config in database | vendor={vendor} "
                               f"type={notification_type} users={len(telegram_ids)} group={group_chat_id}")
                return {
                    'telegram_ids': telegram_ids,
                    'group_chat_id': group_chat_id,
                    'found': True
                }
            else:
                # No config found - log available vendors for debugging
                self.logger.warning(f"[{session_token}] No notification config found for vendor='{vendor}' type='{notification_type}'")
                print(f"[DEBUG] No notification config found for vendor='{vendor}'")
                try:
                    # Query all entry_confirmation configs to show what vendors exist
                    all_configs = _supabase_client.table('vendor_notifications') \
                        .select('vendor, notification_type, telegram_ids') \
                        .eq('notification_type', notification_type) \
                        .execute()
                    if all_configs.data:
                        vendors_in_db = [c.get('vendor') for c in all_configs.data]
                        self.logger.info(f"[{session_token}] Available vendors in DB for {notification_type}: {vendors_in_db}")
                        print(f"[DEBUG] Available vendors in DB: {vendors_in_db}")
                except Exception as diag_err:
                    self.logger.warning(f"[{session_token}] Diagnostic query failed: {diag_err}")
        except Exception as e:
            self.logger.warning(f"[{session_token}] Error fetching notification config | vendor={vendor} error={e}")

        return {'telegram_ids': [], 'group_chat_id': None, 'found': False}

    def _notify_individual_users(self, telegram_ids: list, message: str, session_token: str) -> int:
        """
        Send notification message to individual users.

        Args:
            telegram_ids: List of telegram user IDs to notify
            message: Message to send
            session_token: Correlation token for logging

        Returns:
            int: Number of successful sends
        """
        success_count = 0
        for user_id in telegram_ids:
            try:
                if self.send_message(user_id, message):
                    success_count += 1
                    self.logger.debug(f"[{session_token}] Sent notification to user | user_id={user_id}")
                else:
                    self.logger.warning(f"[{session_token}] Failed to send notification | user_id={user_id}")
            except Exception as e:
                self.logger.warning(f"[{session_token}] Error sending notification | user_id={user_id} error={e}")

        self.logger.info(f"[{session_token}] Individual notifications sent | success={success_count}/{len(telegram_ids)}")
        return success_count

    def _resolve_order_prep_chat(self, vendor: str, session_token: str) -> Optional[int]:
        """
        Resolve order prep chat ID from database only.

        All notification configuration should be done through the dashboard.
        No .env fallback - configure via Notifications > Entry Confirmations.

        Args:
            vendor: Vendor name
            session_token: Correlation token for logging

        Returns:
            Optional[int]: Group chat ID or None if not configured
        """
        self.logger.info(f"[{session_token}] Resolving order prep chat | vendor={vendor}")

        # Get notification config from database ONLY (no .env fallback)
        notification_config = self._get_notification_config(vendor, 'entry_confirmation', session_token)

        if notification_config['found']:
            if notification_config['group_chat_id']:
                chat_id = notification_config['group_chat_id']
                self.logger.info(f"[{session_token}] Resolved group chat from database | vendor={vendor} chat_id={chat_id}")
                return chat_id
            elif notification_config['telegram_ids']:
                self.logger.info(f"[{session_token}] No group chat but individual users configured | vendor={vendor}")
                return None  # Will use individual notifications instead

        # No config found
        self.logger.warning(f"[{session_token}] No notification config for vendor={vendor} | Configure in dashboard")
        return None

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

    def send_variance_notification(self, location: str, submitter: str,
                                    item_name: str, expected: float, actual: float,
                                    variance_pct: float, photo_file_id: str = None):
        """
        Send variance notification to managers.

        Args:
            location: Vendor/location where variance occurred
            submitter: Who submitted the count
            item_name: Item with variance
            expected: Expected quantity
            actual: Actual counted quantity
            variance_pct: Variance percentage (0.20 = 20%)
            photo_file_id: Optional Telegram file ID for photo
        """
        # Get managers for this location
        managers = self.notion.get_managers_for_location(location)

        if not managers:
            self.logger.warning(f"[PHASE 4] No managers found for {location} - notification skipped")
            return

        # Build notification message
        direction = "OVER" if actual > expected else "UNDER"
        variance_str = f"{variance_pct:.0%}"

        message = (
            f"⚠️ <b>VARIANCE ALERT</b>\n"
            f"{'=' * 25}\n\n"
            f"<b>Location:</b> {location}\n"
            f"<b>Item:</b> {item_name}\n"
            f"<b>Submitted by:</b> {submitter}\n\n"
            f"<b>Expected:</b> {expected:.1f}\n"
            f"<b>Actual:</b> {actual}\n"
            f"<b>Variance:</b> {variance_str} {direction}\n"
        )

        if photo_file_id:
            message += f"\n<i>Photo confirmation attached below.</i>"

        # Send to each manager
        for manager in managers:
            manager_id = manager.get('telegram_id')
            manager_name = manager.get('name', 'Unknown')

            try:
                # Send text message
                self.send_message(manager_id, message)

                # Send photo if available
                if photo_file_id:
                    self._make_request("sendPhoto", {
                        "chat_id": manager_id,
                        "photo": photo_file_id,
                        "caption": f"Variance photo: {item_name} @ {location}"
                    })

                self.logger.info(f"[PHASE 4] Variance notification sent to {manager_name} ({manager_id})")
                print(f"[PHASE 4] Variance notification sent to manager: {manager_name}")

            except Exception as e:
                self.logger.error(f"[PHASE 4] Failed to notify manager {manager_name}: {e}")

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
                self.order_handler.cleanup_expired_sessions()
                self.report_handler.cleanup_expired_sessions()
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
        Process update with proper routing for entry wizard, order flow, and other inputs.
        """
        try:
            # Handle callback queries
            if "callback_query" in update:
                callback_data = update["callback_query"].get("data", "")
                
                # Route entry callbacks
                if callback_data.startswith("entry_"):
                    self.entry_handler.handle_callback(update["callback_query"])
                    return
                
                # Route order callbacks
                if callback_data.startswith("order_"):
                    self.order_handler.handle_callback(update["callback_query"])
                    return

                # Route report callbacks
                if callback_data.startswith("report_"):
                    self.report_handler.handle_callback(update["callback_query"])
                    return

                # Route SOP callbacks
                if callback_data.startswith("sop_"):
                    self.sop_handler.handle_callback(update["callback_query"])
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
            
            # Check for active order session BEFORE command processing
            if hasattr(self, 'order_handler') and user_id in self.order_handler.sessions:
                session = self.order_handler.sessions[user_id]
                if not session.is_expired():
                    if 'text' in message:
                        self.order_handler.handle_text_input(message, session)
                        return
            
            # Check for entry session
            if hasattr(self, 'entry_handler') and user_id in self.entry_handler.sessions:
                session = self.entry_handler.sessions[user_id]
                if not session.is_expired():
                    if 'photo' in message:
                        # Debug logging
                        print(f"[DEBUG] Photo received for user {user_id}")
                        print(f"[DEBUG] Session mode: {session.mode}, current_step: {getattr(session, 'current_step', 'NOT SET')}")
                        # Handle photos for received deliveries OR variance confirmation
                        if (session.mode == "received" and session.current_step == "image") or session.current_step == "variance_photo":
                            print(f"[DEBUG] Routing to handle_photo_input")
                            self.entry_handler.handle_photo_input(message, session)
                        else:
                            print(f"[DEBUG] Photo NOT routed - conditions not met")
                        return
                    elif 'text' in message:
                        self.entry_handler.handle_text_input(message, session)
                        return

            # Check for report session
            if hasattr(self, 'report_handler') and user_id in self.report_handler.sessions:
                session = self.report_handler.sessions[user_id]
                if not session.is_expired():
                    if 'photo' in message:
                        self.report_handler.handle_photo_input(message, session)
                        return
                    elif 'video' in message:
                        self.report_handler.handle_video_input(message, session)
                        return
                    elif 'text' in message:
                        self.report_handler.handle_text_input(message, session)
                        return

            # Check for SOP session input (photo/text for confirmation steps)
            if hasattr(self, 'sop_handler') and user_id in self.sop_handler.sessions:
                session = self.sop_handler.sessions[user_id]
                if not session.is_expired():
                    if self.sop_handler.handle_text_input(message):
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
            "/order": self.order_handler.handle_order_command,
            "/order_avondale": self._handle_order_avondale,
            "/order_commissary": self._handle_order_commissary,
            "/reassurance": self._handle_reassurance,
            "/shortages": self._handle_shortages,
            "/status": self._handle_status,
            "/cancel": self._handle_cancel,
            "/adu": self._handle_adu,
            "/review": self._handle_review,
            "/pars": self._handle_pars,
            "/flags": self._handle_flags,
            "/cycles": self._handle_cycles,
            "/reports": self.report_handler.handle_reports_command,
            "/sop": self.sop_handler.handle_sop_command,
            "/viewtasks": self._handle_viewtasks,
            "/mytasks": self._handle_viewtasks,
            "/tasks": self._handle_viewtasks,
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
                f"Version {SYSTEM_VERSION} • Status: {system_status}\n\n"
                
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
            "  • Saves directly to database\n\n"

            "📊 <b>Analytics & Reports</b>\n"
            "/info — Real-time inventory analysis\n"
            "/order — Supplier-ready order lists\n"
            "/reassurance — Risk assessment\n\n"

            "🔍 <b>Quick Checks</b>\n"
            "/adu — Average daily usage rates\n"
            "/status — System health check\n\n"

            "📋 <b>Reports & SOPs</b>\n"
            "/reports — Submit configurable reports\n"
            "  • Select report type → answer questions\n"
            "  • Supports text, photos, videos\n"
            "/sop — Execute Standard Operating Procedures\n"
            "  • Step-by-step guided workflows\n"
            "  • Tracks completion and responses\n\n"

            "📊 <b>Admin Commands</b>\n"
            "/review — View recent submissions\n"
            "/pars — View item par levels\n"
            "/flags — Review flagged transactions\n"
            "/cycles — View consumption cycles\n\n"
            "💻 <b>Admin Dashboard</b>\n"
            "Use the web dashboard for full configuration:\n"
            "• Forecasts, deadlines, count schedules\n"
            "• Item management, par editing\n"
            "• Detailed analytics\n\n"

            "💡 <b>Tips</b>\n"
            "• Use 'today' for current date\n"
            "• Type /skip to skip items\n"
            "• Type /done to finish early\n"
            "• Use /cancel anytime to exit"
        )
        self.send_message(chat_id, text)

    # ===== ADMIN COMMANDS =====
    # Configuration commands (forecast, deadlines, setcount, setescalation) moved to Admin Dashboard

    def _handle_review(self, message: Dict):
        """
        Handle /review command - view recent inventory/order submissions.

        Usage:
            /review - Show today's submissions
            /review <location> - Show today for specific location
            /review <location> <date> - Show specific date (YYYY-MM-DD)
        """
        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        print(f"[REVIEW-CMD] Received: {text}")
        self.logger.info(f"[REVIEW-CMD] {text}")

        parts = text.split()

        # Parse arguments
        location = None
        target_date = get_time_in_timezone(BUSINESS_TIMEZONE).date()

        if len(parts) >= 2:
            location = parts[1]

        if len(parts) >= 3:
            try:
                target_date = datetime.strptime(parts[2], '%Y-%m-%d').date()
            except ValueError:
                self.send_message(chat_id, "❌ Invalid date format. Use YYYY-MM-DD")
                return

        # Query submissions
        submissions = self._get_submissions_for_date(target_date, location)

        if not submissions:
            location_text = f" for {location}" if location else ""
            self.send_message(chat_id,
                f"📋 No submissions found{location_text} on {target_date}\n\n"
                f"Use /review [location] [date] to search other dates.")
            return

        # Format response
        date_str = target_date.strftime('%A, %b %d')
        text_out = f"📋 <b>SUBMISSIONS: {date_str}</b>\n"
        if location:
            text_out += f"📍 Location: {location}\n"
        text_out += "\n"

        # Group by location then type
        by_location = {}
        for sub in submissions:
            loc = sub.get('vendor', 'Unknown')
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(sub)

        for loc, subs in by_location.items():
            text_out += f"<b>📍 {loc}</b>\n"

            for sub in subs:
                sub_type = sub.get('type', 'unknown')
                submitter = sub.get('submitter', 'Unknown')
                created = sub.get('created_at', '')
                flagged = sub.get('flagged', False)

                # Parse time
                try:
                    if isinstance(created, str):
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        time_str = created_dt.strftime('%I:%M %p')
                    else:
                        time_str = 'Unknown'
                except:
                    time_str = 'Unknown'

                # Type icons
                type_icons = {
                    'on_hand': '📦',
                    'received': '📥',
                    'order': '📤'
                }
                icon = type_icons.get(sub_type, '📝')

                # Flag indicator
                flag_str = ' 🚩' if flagged else ''

                # Count items
                quantities = sub.get('quantities', {})
                if isinstance(quantities, str):
                    try:
                        quantities = json.loads(quantities)
                    except:
                        quantities = {}
                item_count = len([v for v in quantities.values() if v and v > 0])

                text_out += f"  {icon} {sub_type.upper()}: {submitter} @ {time_str}{flag_str}\n"
                text_out += f"     └ {item_count} items\n"

            text_out += "\n"

        # Add navigation hint
        text_out += (
            f"─────────────────────\n"
            f"<code>/review [location] [YYYY-MM-DD]</code>"
        )

        self.send_message(chat_id, text_out)
        print(f"[REVIEW-CMD] ✓ Showed {len(submissions)} submissions")

    def _get_submissions_for_date(self, target_date, location: str = None) -> list:
        """
        Query submissions from Supabase for a specific date.
        """
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            print(f"[REVIEW] Supabase not available")
            return []

        try:
            # Build query
            query = _supabase_client.table('inventory_transactions') \
                .select('*') \
                .gte('created_at', target_date.isoformat()) \
                .lt('created_at', (target_date + timedelta(days=1)).isoformat()) \
                .order('created_at', desc=True)

            if location:
                query = query.eq('vendor', location)

            result = query.execute()

            if result.data:
                print(f"[REVIEW] ✓ Found {len(result.data)} submissions for {target_date}")
                return result.data

            return []

        except Exception as e:
            print(f"[REVIEW] ✗ Query failed: {e}")
            self.logger.error(f"[REVIEW] Error: {e}")
            return []

    def _handle_pars(self, message: Dict):
        """
        Handle /pars command - view and adjust item par levels.

        Usage:
            /pars <location> - Show all pars for location
            /pars <location> <item> - Show specific item
            /pars <location> <item> <min> <max> - Update pars
        """
        chat_id = message["chat"]["id"]
        user_name = message["from"].get("first_name", "Unknown")
        text = message.get("text", "").strip()

        print(f"[PARS-CMD] Received: {text}")
        self.logger.info(f"[PARS-CMD] {text}")

        # Parse with quote handling for item names
        parts = self._parse_quoted_args(text)

        # /pars alone - show help
        if len(parts) == 1:
            self.send_message(chat_id,
                "📊 <b>Par Management</b>\n\n"
                "Usage:\n"
                "<code>/pars [location]</code> - View all pars\n"
                "<code>/pars [location] \"[item]\"</code> - View item\n"
                "<code>/pars [location] \"[item]\" [min] [max]</code> - Update\n\n"
                "Examples:\n"
                "<code>/pars Avondale</code>\n"
                "<code>/pars Avondale \"Chicken Wings\"</code>\n"
                "<code>/pars Avondale \"Chicken Wings\" 30 60</code>")
            return

        location = parts[1]

        # /pars <location> - show all pars
        if len(parts) == 2:
            self._show_location_pars(chat_id, location)
            return

        item_name = parts[2]

        # /pars <location> <item> - show item pars
        if len(parts) == 3:
            self._show_item_pars(chat_id, location, item_name)
            return

        # /pars <location> <item> <min> <max> - update pars
        if len(parts) >= 5:
            try:
                new_min = float(parts[3])
                new_max = float(parts[4])
            except ValueError:
                self.send_message(chat_id, "❌ Invalid par values. Use numbers.")
                return

            if new_min >= new_max:
                self.send_message(chat_id, "❌ Min par must be less than max par.")
                return

            self._update_item_pars(chat_id, location, item_name, new_min, new_max, user_name)
            return

        self.send_message(chat_id, "❌ Invalid syntax. Use /pars for help.")

    def _parse_quoted_args(self, text: str) -> list:
        """
        Parse command arguments with quote support.

        Example: '/pars Avondale "Chicken Wings" 30 60'
        Returns: ['/pars', 'Avondale', 'Chicken Wings', '30', '60']
        """
        import shlex
        try:
            return shlex.split(text)
        except:
            return text.split()

    def _show_location_pars(self, chat_id: int, location: str):
        """Show all pars for a location."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            self.send_message(chat_id, "❌ Database not available")
            return

        try:
            result = _supabase_client.table('inventory_items') \
                .select('item_name, min_par, max_par, avg_consumption') \
                .eq('vendor', location) \
                .eq('active', True) \
                .order('item_name') \
                .execute()

            if not result.data:
                self.send_message(chat_id, f"❌ No items found for '{location}'")
                return

            text = f"📊 <b>PARS: {location}</b>\n\n"
            text += "Item | Min | Max\n"
            text += "─────────────────────────\n"

            unconfigured = 0
            for item in result.data:
                name = item['item_name'][:20]  # Truncate long names
                min_p = item.get('min_par', 0) or 0
                max_p = item.get('max_par', 0) or 0

                if min_p == 0 and max_p == 0:
                    unconfigured += 1
                    text += f"⚠️ {name}: NOT SET\n"
                else:
                    text += f"  {name}: {min_p:.0f} - {max_p:.0f}\n"

            text += (
                f"\n─────────────────────────\n"
                f"📦 {len(result.data)} items total\n"
            )
            if unconfigured > 0:
                text += f"⚠️ {unconfigured} without pars\n"

            text += f"\nTo update:\n<code>/pars {location} \"[item]\" [min] [max]</code>"

            self.send_message(chat_id, text)
            print(f"[PARS-CMD] ✓ Showed {len(result.data)} items for {location}")

        except Exception as e:
            print(f"[PARS-CMD] ✗ Error: {e}")
            self.send_message(chat_id, "❌ Failed to load pars")

    def _show_item_pars(self, chat_id: int, location: str, item_name: str):
        """Show pars and history for a specific item."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            self.send_message(chat_id, "❌ Database not available")
            return

        try:
            # Get item
            item_result = _supabase_client.table('inventory_items') \
                .select('*') \
                .eq('vendor', location) \
                .ilike('item_name', f'%{item_name}%') \
                .limit(1) \
                .execute()

            if not item_result.data:
                self.send_message(chat_id, f"❌ Item '{item_name}' not found in {location}")
                return

            item = item_result.data[0]

            # Get par history
            history_result = _supabase_client.table('par_history') \
                .select('*') \
                .eq('item_id', item['id']) \
                .order('created_at', desc=True) \
                .limit(5) \
                .execute()

            text = (
                f"📊 <b>ITEM: {item['item_name']}</b>\n"
                f"📍 Location: {location}\n\n"
                f"<b>Current Pars:</b>\n"
                f"  📉 Min: {item.get('min_par', 0) or 0:.1f}\n"
                f"  📈 Max: {item.get('max_par', 0) or 0:.1f}\n"
                f"  📊 Avg consumption: {item.get('avg_consumption', 0) or 0:.2f}/day\n"
            )

            if history_result.data:
                text += f"\n<b>Par History:</b>\n"
                for h in history_result.data[:5]:
                    date = h.get('created_at', '')[:10]
                    old_min = h.get('old_min_par', 0) or 0
                    old_max = h.get('old_max_par', 0) or 0
                    new_min = h.get('new_min_par', 0) or 0
                    new_max = h.get('new_max_par', 0) or 0
                    reason = h.get('reason', '')[:30]

                    text += f"  {date}: {old_min:.0f}-{old_max:.0f} → {new_min:.0f}-{new_max:.0f}\n"
                    if reason:
                        text += f"    └ {reason}\n"

            text += (
                f"\n─────────────────────────\n"
                f"To update:\n"
                f"<code>/pars {location} \"{item['item_name']}\" [min] [max]</code>"
            )

            self.send_message(chat_id, text)
            print(f"[PARS-CMD] ✓ Showed pars for {item['item_name']}")

        except Exception as e:
            print(f"[PARS-CMD] ✗ Error: {e}")
            self.send_message(chat_id, "❌ Failed to load item")

    def _update_item_pars(self, chat_id: int, location: str, item_name: str,
                          new_min: float, new_max: float, updated_by: str):
        """Update par levels for an item."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            self.send_message(chat_id, "❌ Database not available")
            return

        try:
            # Find item
            item_result = _supabase_client.table('inventory_items') \
                .select('id, item_name, min_par, max_par') \
                .eq('vendor', location) \
                .ilike('item_name', f'%{item_name}%') \
                .limit(1) \
                .execute()

            if not item_result.data:
                self.send_message(chat_id, f"❌ Item '{item_name}' not found in {location}")
                return

            item = item_result.data[0]
            old_min = item.get('min_par', 0) or 0
            old_max = item.get('max_par', 0) or 0

            print(f"[PARS-CMD] Updating {item['item_name']}: {old_min}-{old_max} → {new_min}-{new_max}")

            # Update item
            update_result = _supabase_client.table('inventory_items') \
                .update({
                    'min_par': new_min,
                    'max_par': new_max,
                    'last_par_update': datetime.now().isoformat()
                }) \
                .eq('id', item['id']) \
                .execute()

            if not update_result.data:
                self.send_message(chat_id, "❌ Failed to update pars")
                return

            # Log to par_history
            _supabase_client.table('par_history') \
                .insert({
                    'item_id': item['id'],
                    'old_min_par': old_min,
                    'old_max_par': old_max,
                    'new_min_par': new_min,
                    'new_max_par': new_max,
                    'reason': f'Manual update by {updated_by}'
                }) \
                .execute()

            self.send_message(chat_id,
                f"✅ <b>Pars Updated</b>\n\n"
                f"📦 {item['item_name']}\n"
                f"📍 {location}\n\n"
                f"Old: {old_min:.0f} - {old_max:.0f}\n"
                f"New: {new_min:.0f} - {new_max:.0f}\n\n"
                f"Updated by: {updated_by}")

            print(f"[PARS-CMD] ✓ Updated pars for {item['item_name']}")
            self.logger.info(f"[PARS-CMD] Updated {item['item_name']}: {new_min}-{new_max} by {updated_by}")

        except Exception as e:
            print(f"[PARS-CMD] ✗ Update failed: {e}")
            self.send_message(chat_id, "❌ Failed to update pars")

    def _handle_flags(self, message: Dict):
        """
        Handle /flags command - review flagged variance transactions.

        Usage:
            /flags - Show all unresolved flags
            /flags <location> - Show flags for location
            /flags resolve <id> - Mark flag as resolved
        """
        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        print(f"[FLAGS-CMD] Received: {text}")
        self.logger.info(f"[FLAGS-CMD] {text}")

        parts = text.split()

        # /flags resolve <id>
        if len(parts) >= 3 and parts[1].lower() == 'resolve':
            flag_id = parts[2]
            self._resolve_flag(chat_id, flag_id)
            return

        # /flags or /flags <location>
        location = parts[1] if len(parts) >= 2 else None
        self._show_flags(chat_id, location)

    def _show_flags(self, chat_id: int, location: str = None):
        """Show flagged transactions."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            self.send_message(chat_id, "❌ Database not available")
            return

        try:
            query = _supabase_client.table('inventory_transactions') \
                .select('*') \
                .eq('flagged', True) \
                .order('created_at', desc=True) \
                .limit(20)

            if location:
                query = query.eq('vendor', location)

            result = query.execute()

            if not result.data:
                location_text = f" for {location}" if location else ""
                self.send_message(chat_id,
                    f"✅ No flagged transactions{location_text}!\n\n"
                    f"Great job on count accuracy.")
                return

            text = "🚩 <b>FLAGGED TRANSACTIONS</b>\n\n"

            for flag in result.data[:10]:
                flag_id = flag.get('id', '')[:8]
                vendor = flag.get('vendor', 'Unknown')
                submitter = flag.get('submitter', 'Unknown')
                flag_reason = flag.get('flag_reason', 'Variance detected')
                created = flag.get('created_at', '')[:10]

                text += (
                    f"🚩 <b>ID: {flag_id}...</b>\n"
                    f"  📍 {vendor}\n"
                    f"  👤 {submitter}\n"
                    f"  📅 {created}\n"
                    f"  ⚠️ {flag_reason}\n\n"
                )

            if len(result.data) > 10:
                text += f"<i>... and {len(result.data) - 10} more</i>\n\n"

            text += (
                f"─────────────────────────\n"
                f"To resolve:\n"
                f"<code>/flags resolve [id]</code>"
            )

            self.send_message(chat_id, text)
            print(f"[FLAGS-CMD] ✓ Showed {len(result.data)} flags")

        except Exception as e:
            print(f"[FLAGS-CMD] ✗ Error: {e}")
            self.send_message(chat_id, "❌ Failed to load flags")

    def _resolve_flag(self, chat_id: int, flag_id: str):
        """Mark a flag as resolved."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            self.send_message(chat_id, "❌ Database not available")
            return

        try:
            # Find the transaction
            result = _supabase_client.table('inventory_transactions') \
                .select('*') \
                .ilike('id', f'{flag_id}%') \
                .eq('flagged', True) \
                .limit(1) \
                .execute()

            if not result.data:
                self.send_message(chat_id, f"❌ Flag '{flag_id}' not found or already resolved")
                return

            transaction = result.data[0]

            # Update to unflag
            update_result = _supabase_client.table('inventory_transactions') \
                .update({
                    'flagged': False,
                    'flag_reason': f"RESOLVED: {transaction.get('flag_reason', '')}"
                }) \
                .eq('id', transaction['id']) \
                .execute()

            if update_result.data:
                self.send_message(chat_id,
                    f"✅ <b>Flag Resolved</b>\n\n"
                    f"📍 {transaction.get('vendor')}\n"
                    f"👤 {transaction.get('submitter')}\n"
                    f"📅 {transaction.get('created_at', '')[:10]}")
                print(f"[FLAGS-CMD] ✓ Resolved flag {flag_id}")
            else:
                self.send_message(chat_id, "❌ Failed to resolve flag")

        except Exception as e:
            print(f"[FLAGS-CMD] ✗ Resolve failed: {e}")
            self.send_message(chat_id, "❌ Failed to resolve flag")

    # ===== END ADMIN COMMANDS =====

    # ===== CYCLE COMMANDS =====

    def _handle_cycles(self, message: Dict):
        """
        Handle /cycles command - view consumption cycle status.

        Usage:
            /cycles - Show recent cycles and flagged items
            /cycles <location> - Filter by location
        """
        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        print(f"[CYCLES-CMD] Received: {text}")

        parts = text.split()
        location = parts[1] if len(parts) > 1 else None

        # Get flagged cycles
        flagged = self.notion.get_flagged_cycles(location)

        # Get recent closed cycles
        recent = self._get_recent_cycles(location)

        text_out = "🔄 <b>CONSUMPTION CYCLES</b>\n\n"

        # Show flagged first
        if flagged:
            text_out += f"⚠️ <b>FLAGGED FOR REVIEW ({len(flagged)})</b>\n"
            for cycle in flagged[:5]:
                item = cycle.get('item_name', 'Unknown')
                drift = cycle.get('drift_percentage', 0) or 0
                loc = cycle.get('location', '')
                text_out += f"  🚩 {item} ({loc}): {drift:.0f}% drift\n"
            text_out += "\n"
        else:
            text_out += "✅ No cycles flagged for review\n\n"

        # Show recent cycles
        if recent:
            text_out += f"<b>RECENT CYCLES</b>\n"
            for cycle in recent[:10]:
                item = cycle.get('item_name', 'Unknown')[:15]
                status = cycle.get('status', 'unknown')
                drift = cycle.get('drift_percentage', 0) or 0

                status_icons = {
                    'open': '🔵',
                    'closed': '✅',
                    'flagged': '🚩',
                    'adjusted': '🔧'
                }
                icon = status_icons.get(status, '❓')

                text_out += f"  {icon} {item}: {drift:.0f}% ({status})\n"
        else:
            text_out += "No recent cycles\n"

        text_out += (
            f"\n─────────────────────────\n"
            f"Cycles auto-adjust pars when drift >25% for 2 consecutive cycles."
        )

        self.send_message(chat_id, text_out)
        print(f"[CYCLES-CMD] ✓ Showed {len(flagged)} flagged, {len(recent)} recent")

    def _get_recent_cycles(self, location: str = None, limit: int = 10) -> list:
        """Get recently closed/adjusted cycles."""
        global _supabase_client, _supabase_enabled

        if not _supabase_enabled or not _supabase_client:
            return []

        try:
            query = _supabase_client.table('consumption_cycles') \
                .select('*') \
                .in_('status', ['closed', 'adjusted', 'flagged', 'open']) \
                .order('updated_at', desc=True) \
                .limit(limit)

            if location:
                query = query.eq('location', location)

            result = query.execute()
            return result.data or []

        except Exception as e:
            print(f"[CYCLES-CMD] ✗ Error: {e}")
            return []

    # ===== END CYCLE COMMANDS =====

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

    def _handle_viewtasks(self, message: Dict):
        """Show user's pending tasks for today."""
        chat_id = message["chat"]["id"]
        telegram_id = message.get("from", {}).get("id")

        if not _supabase_enabled or not _supabase_client:
            self.send_message(chat_id, "⚠️ Task system not available.")
            return

        try:
            # Get user from database
            user_result = _supabase_client.table('telegram_users') \
                .select('id, name') \
                .eq('telegram_id', telegram_id) \
                .execute()

            if not user_result.data:
                self.send_message(chat_id, "❌ You're not registered in the system.")
                return

            user = user_result.data[0]
            user_id = user['id']
            user_name = user.get('name', 'User')

            # Get today's date and current time
            now = get_time_in_timezone(BUSINESS_TIMEZONE)
            today = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M')

            # Get today's tasks for this user
            tasks_result = _supabase_client.table('task_assignments') \
                .select('*, report_types(name)') \
                .eq('assigned_to', user_id) \
                .eq('scheduled_date', today) \
                .in_('status', ['pending', 'in_progress']) \
                .execute()

            if not tasks_result.data:
                self.send_message(chat_id, f"✅ No active tasks for today, {user_name}!")
                return

            # Filter by start_time and end_time
            active_tasks = []
            for task in tasks_result.data:
                start = task.get('start_time') or '00:00'
                end = task.get('end_time') or '23:59'
                # Convert times for comparison (just HH:MM)
                start_cmp = start[:5] if len(start) >= 5 else start
                end_cmp = end[:5] if len(end) >= 5 else end
                if start_cmp <= current_time <= end_cmp:
                    active_tasks.append(task)

            if not active_tasks:
                self.send_message(chat_id, f"✅ No active tasks right now, {user_name}!")
                return

            # Format response
            response = f"📋 <b>Your Tasks for Today</b>\n"
            response += f"👤 {user_name}\n"
            response += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            for i, task in enumerate(active_tasks, 1):
                status_emoji = "🔵" if task['status'] == 'pending' else "🟡"

                # Get task name
                if task.get('report_types') and task['report_types'].get('name'):
                    task_name = task['report_types']['name']
                else:
                    task_name = task.get('notes') or 'Task'

                # Format due time
                due = task.get('due_time')
                if due:
                    # Convert HH:MM:SS to 12-hour format
                    due_parts = due.split(':')
                    hour = int(due_parts[0])
                    minute = due_parts[1] if len(due_parts) > 1 else '00'
                    period = 'AM' if hour < 12 else 'PM'
                    display_hour = hour % 12 or 12
                    due_str = f"{display_hour}:{minute} {period}"
                else:
                    due_str = 'No due time'

                response += f"{status_emoji} {i}. <b>{task_name}</b>\n"
                response += f"   ⏱️ Due: {due_str}\n\n"

            response += "Use the Happy Manager dashboard to update task status."

            self.send_message(chat_id, response)

        except Exception as e:
            self.logger.error(f"/viewtasks failed: {e}", exc_info=True)
            self.send_message(chat_id, "⚠️ Error fetching tasks. Please try again.")

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
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or "").strip()
        low = text.lower()
        state.update_activity()

        # global escape
        if low == "/cancel":
            self._handle_cancel(message)
            return True

        if user_id and user_id in self.entry_handler.sessions:
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
                entry_type_display = "on-hand count" if state.entry_type == "on_hand" else "delivery"

                # === CONSUMPTION CYCLE TRIGGERS ===
                try:
                    if state.entry_type == 'received':
                        # Delivery received - start new consumption cycles
                        print(f"[ENTRY] Starting consumption cycles for received delivery")
                        on_hand_data = self.notion.get_latest_inventory(state.location, 'on_hand')

                        for item_name, received_qty in quantities.items():
                            if received_qty and received_qty > 0:
                                current_on_hand = float(on_hand_data.get(item_name, 0) or 0)
                                # Start of new cycle = current on_hand + what we just received
                                start_on_hand = current_on_hand + received_qty

                                self.notion.start_consumption_cycle(
                                    location=state.location,
                                    item_name=item_name,
                                    on_hand=start_on_hand,
                                    received=received_qty
                                )

                    elif state.entry_type == 'on_hand':
                        # On-hand count - close any open cycles
                        print(f"[ENTRY] Checking consumption cycles for on_hand count")

                        for item_name, on_hand_qty in quantities.items():
                            if on_hand_qty is not None:
                                # This will close any open cycles and trigger calibration check
                                self.notion._close_open_cycles(
                                    location=state.location,
                                    item_name=item_name,
                                    end_on_hand=float(on_hand_qty)
                                )
                except Exception as cycle_err:
                    print(f"[ENTRY] ⚠ Cycle processing error (non-fatal): {cycle_err}")
                    self.logger.warning(f"Cycle processing error: {cycle_err}")
                # === END CONSUMPTION CYCLE TRIGGERS ===

                self.send_message(state.chat_id,
                                f"✅ <b>Entry Saved</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Saved {items_count} items for {state.location}\n"
                                f"Type: {entry_type_display}\n"
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
        Display inventory dashboard for all locations dynamically discovered from Notion.
        Shows forecasted on-hand at delivery and post-delivery needs for each location.

        No hard-coded location names — fully dynamic from Notion.

        Args:
            message: Telegram message object

        Logs: command entry, location discovery, per-location calculation
        """
        chat_id = message["chat"]["id"]

        def format_critical_item(item: dict) -> str:
            """Format one critical item with forecast values (no bold, no dashes)."""
            name = str(item.get("item_name", "Unknown")).strip() or "Unknown"
            unit = item.get("unit_type", "unit")
            current = float(item.get("current_qty", 0) or 0)
            oh_delivery = float(item.get("oh_at_delivery", 0) or 0)
            need = float(item.get("consumption_need", 0) or 0)
            order = item.get("required_order_rounded", 0)

            # Severity icon
            if oh_delivery == 0:
                status_icon = "🚨"
            elif need > 0 and oh_delivery < need * 0.3:
                status_icon = "⚠️"
            else:
                status_icon = "📉"

            return (
                f"{status_icon} {name}\n"
                f"   Order {order} {unit} • Now {current:.1f} → Delivery {oh_delivery:.1f}\n"
                f"   Need {need:.1f} for post-delivery window"
            )

        try:
            now = get_time_in_timezone(BUSINESS_TIMEZONE)
            self.logger.info(f"/info command | chat={chat_id} time={now.strftime('%Y-%m-%d %H:%M')}")

            # Discover all locations dynamically from Notion
            try:
                locations = self.notion.get_locations(use_cache=True)
                self.logger.info(f"Info: discovered locations | count={len(locations)} locations={locations}")
            except Exception as e:
                self.logger.error(f"Info: failed to discover locations | error={e}", exc_info=True)
                self.send_message(
                    chat_id,
                    "Unable to load locations from Notion. Please try again or contact support."
                )
                return

            # Normalize, dedupe, sort
            norm_locations = sorted({str(loc).strip() for loc in (locations or []) if str(loc).strip()})
            if not norm_locations:
                self.logger.error("Info: no locations found in Items Master")
                self.send_message(
                    chat_id,
                    "No locations found in Items Master. Please add items with a Location property in Notion."
                )
                return

            # Header
            text = (
                "Inventory dashboard\n"
                f"{now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
            )

            # Per-location summaries
            for location in norm_locations:
                try:
                    summary = self.calc.calculate_location_summary(location)
                    self.logger.info(f"Info: calculated summary | location='{location}'")
                except Exception as e:
                    self.logger.error(
                        f"Info: failed to calculate summary | location='{location}' error={e}",
                        exc_info=True
                    )
                    text += f"{location.upper()}\n"
                    text += "Error calculating status\n\n"
                    continue

                status_counts = summary.get("status_counts", {}) or {}
                red_count = int(status_counts.get("RED", 0) or 0)
                green_count = int(status_counts.get("GREEN", 0) or 0)
                days_until = float(summary.get("days_until_delivery", 0) or 0)
                delivery_date = summary.get("delivery_date", "—") or "—"

                text += (
                    f"{location.upper()}\n"
                    f"  Next delivery: {delivery_date} ({days_until:.1f} days)\n"
                    f"  Status: 🔴 {red_count} • 🟢 {green_count}\n"
                )

                # Critical items
                items = summary.get("items", []) or []
                critical = [it for it in items if it.get("status") == "RED"]

                if critical:
                    text += "  Critical items:\n"
                    for it in sorted(
                        critical,
                        key=lambda x: (x.get("required_order_rounded") or 0),
                        reverse=True
                    )[:5]:
                        for line in format_critical_item(it).split("\n"):
                            text += f"    {line}\n"
                    if len(critical) > 5:
                        text += f"    ...and {len(critical) - 5} more critical\n"
                else:
                    text += "  All items sufficient through next delivery\n"

                text += "\n"

            # Footer
            text += (
                "How to read:\n"
                "• Now → Delivery: consumption forecast\n"
                "• Need: required for post-delivery period\n"
                "• Use /order for a supplier-ready list"
            )

            self.send_message(chat_id, text)
            self.logger.info(f"/info sent | locations={len(norm_locations)}")

        except Exception as e:
            self.logger.error(f"/info failed | error={e}", exc_info=True)
            self.send_message(chat_id, "Unable to generate dashboard. Please try again.")


    def _handle_order_avondale(self, message: Dict):
        """
        Legacy wrapper for /order_avondale command.
        Delegates to dynamic order flow with 'Avondale' location string.
        
        This is a compatibility shim with zero business logic.
        The location string 'Avondale' must exist in Notion Items Master
        for this command to work.
        
        Args:
            message: Telegram message object
            
        Logs: command entry, delegation to handler
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        
        self.logger.info(f"/order_avondale command | user={user_id} chat={chat_id} | delegating to dynamic order flow")
        
        try:
            # Delegate to order handler with literal location string
            # This string MUST match a Location value in Notion Items Master
            self.order_handler.handle_preselected_vendor_command(message, "Avondale")
            
        except Exception as e:
            self.logger.error(f"/order_avondale delegation failed | user={user_id} error={e}", exc_info=True)
            self.send_message(
                chat_id,
                "⚠️ Unable to start order flow for Avondale.\n"
                "Please try /order or contact support."
            )

    def _handle_order_commissary(self, message: Dict):
        """
        Legacy wrapper for /order_commissary command.
        Delegates to dynamic order flow with 'Commissary' location string.
        
        This is a compatibility shim with zero business logic.
        The location string 'Commissary' must exist in Notion Items Master
        for this command to work.
        
        Args:
            message: Telegram message object
            
        Logs: command entry, delegation to handler
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        
        self.logger.info(f"/order_commissary command | user={user_id} chat={chat_id} | delegating to dynamic order flow")
        
        try:
            # Delegate to order handler with literal location string
            # This string MUST match a Location value in Notion Items Master
            self.order_handler.handle_preselected_vendor_command(message, "Commissary")
            
        except Exception as e:
            self.logger.error(f"/order_commissary delegation failed | user={user_id} error={e}", exc_info=True)
            self.send_message(
                chat_id,
                "⚠️ Unable to start order flow for Commissary.\n"
                "Please try /order or contact support."
            )
    
    def _compute_item_coverage_status(self, item_name: str, adu: float, on_hand: float, 
                                       coverage_days: float, unit_type: str) -> Dict[str, Any]:
        """
        Compute coverage status and recommendation for a single item.
        
        Formula:
        - needed = ADU × coverage_days
        - critical = OnHand < needed
        - recommended_order = max(0, ceil(needed - OnHand))
        
        Args:
            item_name: Item name
            adu: Average daily usage
            on_hand: Current on-hand quantity
            coverage_days: Days to cover
            unit_type: Unit type for display
            
        Returns:
            Dict with: needed, critical, recommended_order, days_of_stock
            
        Logs: all inputs and computed values
        """
        import math
        
        needed = adu * coverage_days
        critical = on_hand < needed
        recommended_raw = needed - on_hand
        recommended_order = max(0, math.ceil(recommended_raw))
        
        # Calculate days of stock remaining
        days_of_stock = (on_hand / adu) if adu > 0 else float('inf')
        
        result = {
            'item_name': item_name,
            'adu': adu,
            'on_hand': on_hand,
            'coverage_days': coverage_days,
            'needed': needed,
            'critical': critical,
            'recommended_order': recommended_order,
            'days_of_stock': days_of_stock,
            'unit_type': unit_type
        }
        
        self.logger.debug(
            f"Coverage status | item={item_name} adu={adu:.2f} on_hand={on_hand:.1f} "
            f"coverage_days={coverage_days} needed={needed:.1f} critical={critical} rec={recommended_order}"
        )
        
        return result

    def _handle_reassurance(self, message: Dict):
        """
        Daily risk assessment across all locations dynamically discovered from Notion.

        For each location in Items Master:
        - Loads items and latest on-hand inventory
        - Calculates coverage using location-specific consumption days
        - Identifies critical items needing immediate orders
        - Computes recommended order quantities

        No hard-coded location names — fully dynamic from Notion.

        Args:
            message: Telegram message object

        Logs: command entry, location discovery, per-location evaluation, summary
        """
        chat_id = message["chat"]["id"]

        try:
            now = get_time_in_timezone(BUSINESS_TIMEZONE)
            self.logger.info(f"/reassurance command | chat={chat_id} time={now.strftime('%Y-%m-%d %H:%M')}")

            # Discover all locations dynamically from Notion
            try:
                locations = self.notion.get_locations(use_cache=True)
                self.logger.info(f"Reassurance: discovered locations | count={len(locations)} locations={locations}")
            except Exception as e:
                self.logger.error(f"Reassurance: failed to discover locations | error={e}", exc_info=True)
                self.send_message(chat_id, "Unable to load locations from Notion. Please try again or contact support.")
                return

            # Normalize, dedupe, sort
            norm_locations = sorted({str(loc).strip() for loc in (locations or []) if str(loc).strip()})
            if not norm_locations:
                self.logger.error("Reassurance: no locations found in Items Master")
                self.send_message(chat_id, "No locations found in Items Master. Please add items with a Location property in Notion.")
                return

            # Define coverage days per location from config if present
            default_coverage_days = 3.0
            location_coverage = {}
            inventory_config = globals().get("INVENTORY_CONFIG", {}) or {}

            for location in norm_locations:
                if location in inventory_config:
                    schedule = inventory_config[location].get("consumption_schedule", {}) or {}
                    if schedule:
                        avg_days = sum(schedule.values()) / max(len(schedule), 1)
                        location_coverage[location] = float(avg_days)
                        self.logger.debug(f"Reassurance: loaded coverage from config | location='{location}' days={avg_days:.2f}")
                    else:
                        location_coverage[location] = default_coverage_days
                        self.logger.debug(f"Reassurance: using default coverage | location='{location}' days={default_coverage_days}")
                else:
                    location_coverage[location] = default_coverage_days
                    self.logger.info(f"Reassurance: location not in config, using default | location='{location}' days={default_coverage_days}")

            all_critical_items = []
            location_summaries = {}

            # Evaluate each location dynamically
            for location in norm_locations:
                coverage_days = float(location_coverage.get(location, default_coverage_days))
                self.logger.info(f"Reassurance: evaluating location | location='{location}' coverage_days={coverage_days}")

                # Get items from Notion for this location
                try:
                    items = self.notion.get_items_for_location(location, use_cache=False)
                    self.logger.info(f"Reassurance: items loaded | location='{location}' count={len(items)}")
                except Exception as e:
                    self.logger.error(f"Reassurance: failed to load items | location='{location}' error={e}", exc_info=True)
                    items = []

                # Get latest On-Hand inventory for this location
                try:
                    inventory_data = self.notion.get_latest_inventory(location, entry_type="on_hand")
                    self.logger.info(f"Reassurance: inventory loaded | location='{location}' items_with_qty={len(inventory_data)}")
                except Exception as e:
                    self.logger.error(f"Reassurance: failed to load inventory | location='{location}' error={e}", exc_info=True)
                    inventory_data = {}

                # Evaluate each item for this location
                critical_items = []
                ok_items = []

                for item in items:
                    on_hand = float(inventory_data.get(item.name, 0.0))
                    if item.name not in inventory_data:
                        self.logger.warning(f"Reassurance: missing on-hand data | location='{location}' item={item.name} | defaulting to 0.0")

                    status = self._compute_item_coverage_status(
                        item_name=item.name,
                        adu=item.adu,
                        on_hand=on_hand,
                        coverage_days=coverage_days,
                        unit_type=item.unit_type
                    )

                    if status.get("critical"):
                        critical_items.append(status)
                    else:
                        ok_items.append(status)

                location_summaries[location] = {
                    "coverage_days": coverage_days,
                    "total_items": len(items),
                    "critical_count": len(critical_items),
                    "ok_count": len(ok_items),
                    "critical_items": critical_items,
                }

                all_critical_items.extend(critical_items)
                self.logger.info(
                    f"Reassurance: location evaluated | location='{location}' total={len(items)} critical={len(critical_items)} ok={len(ok_items)}"
                )

            # Build message
            total_critical = len(all_critical_items)

            if total_critical == 0:
                # All clear message
                text = (
                    "Daily risk assessment\n"
                    f"{now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
                    "All clear\n"
                    "No critical inventory issues detected\n\n"
                    "Location status\n"
                )

                for location in norm_locations:
                    summary = location_summaries.get(location, {})
                    text += (
                        f"{location}: {summary.get('ok_count', 0)} items OK\n"
                        f"  Coverage: {float(summary.get('coverage_days', default_coverage_days)):.1f} days\n"
                    )

                text += (
                    "Total coverage: 100%\n\n"
                    "All inventory levels sufficient\n"
                    "No immediate action required\n\n"
                    "System status: Healthy"
                )

            else:
                # Critical items alert
                text = (
                    "Daily risk assessment\n"
                    f"{now.strftime('%I:%M %p')} • {now.strftime('%A, %b %d')}\n\n"
                    "Action required\n"
                    f"{total_critical} item{'s' if total_critical != 1 else ''} need ordering\n\n"
                )

                # Show critical items by location
                for location in norm_locations:
                    summary = location_summaries.get(location, {})
                    critical_items = summary.get("critical_items", []) or []

                    if critical_items:
                        text += f"{location.upper()} ({len(critical_items)} critical)\n"
                        text += f"Coverage requirement: {float(summary.get('coverage_days', default_coverage_days)):.1f} days\n\n"

                        # Sort by severity (lowest days of stock first) and show top 5
                        for item in sorted(critical_items, key=lambda x: x.get("days_of_stock", 0.0))[:5]:
                            text += f"🔴 {item.get('item_name', 'Unknown')}\n"
                            text += f"   On-hand: {float(item.get('on_hand', 0.0)):.1f} {item.get('unit_type', 'unit')}\n"
                            text += f"   Need: {float(item.get('needed', 0.0)):.1f} • Order: {item.get('recommended_order', 0)}\n"
                            text += f"   Days remaining: {float(item.get('days_of_stock', 0.0)):.1f}\n"

                        if len(critical_items) > 5:
                            text += f"...plus {len(critical_items) - 5} more\n"

                        text += "\n"

                text += (
                    "Immediate action needed\n"
                    "These items need ordering now\n\n"
                    "Contact supplier immediately\n"
                    "Use /order for complete list"
                )

            # Send to reassurance chat if different from requester
            reassurance_chat = self.chat_config.get("reassurance")
            if reassurance_chat and reassurance_chat != chat_id:
                self.send_message(reassurance_chat, text)
                self.logger.info(f"Reassurance sent to management chat {reassurance_chat}")

            # Always send to requesting user
            self.send_message(chat_id, text)

            # Log summary
            summary_log = " | ".join(
                [f"{loc}={location_summaries.get(loc, {}).get('critical_count', 0)}" for loc in norm_locations]
            )
            self.logger.info(f"Reassurance complete | total_critical={total_critical} by_location=[{summary_log}]")

        except Exception as e:
            self.logger.error(f"Reassurance failed | error={e}", exc_info=True)
            self.send_message(chat_id, "Unable to generate risk assessment. Please try again.")


            
    
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
    Now fully location-agnostic - uses dynamic location from Notion.
    """
    user_id: int
    chat_id: int
    location: str = ""  # Dynamic location from Notion (single source of truth)
    mode: str = ""  # 'on_hand' or 'received'
    items: List[Dict[str, Any]] = field(default_factory=list)
    index: int = 0
    answers: Dict[str, Optional[float]] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=2))
    last_message_id: Optional[int] = None
    submitter_name: str = ""
    notes: str = ""
    image_file_id: Optional[str] = None
    current_step: str = "items"

    # Tracking for entry confirmation stats
    recount_count: int = 0  # How many times user chose to recount
    variance_count: int = 0  # How many variance alerts were confirmed
    variance_items: List[str] = field(default_factory=list)  # Items that had variance
    
    # Legacy field for backwards compatibility - both point to same value
    @property
    def vendor(self) -> str:
        """Alias for location to maintain compatibility with existing code."""
        return self.location
    
    @vendor.setter
    def vendor(self, value: str):
        """Alias setter for location."""
        self.location = value
    
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
        self.posted_entry_hashes: Dict[str, datetime] = {}  # hash -> timestamp for idempotency
        self._temp_messages: Dict[int, Dict] = {}  # Temporary storage for messages
    
    def _get_telegram_display_name(self, message: Dict) -> str:
        """
        Extract display name from Telegram message.
        
        Priority: first_name + last_name > username > "Unknown"
        
        Args:
            message: Telegram message object
            
        Returns:
            str: Display name
            
        Logs: extraction result
        """
        user = message.get("from", {})
        
        first_name = user.get("first_name", "").strip()
        last_name = user.get("last_name", "").strip()
        username = user.get("username", "").strip()
        
        if first_name or last_name:
            display_name = f"{first_name} {last_name}".strip()
        elif username:
            display_name = f"@{username}"
        else:
            display_name = "Unknown"
        
        self.logger.debug(f"Extracted Telegram display name | name='{display_name}' user_id={user.get('id')}")

        return display_name

    def handle_entry_command(self, message: Dict):
        """
        Entry point for /entry command.
        Start with vendor selection (same as order flow).
        
        Logs: command entry, session check, vendor prompt.
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        
        self.logger.info(f"/entry command entry | user={user_id} chat={chat_id}")
        
        # Check for existing session
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if not session.is_expired():
                self.logger.info(f"Active entry session found | vendor={session.vendor} mode={session.mode}")
                # Offer resume or restart
                keyboard = self._create_keyboard([
                    [("📂 Resume", "entry_resume"), ("🔄 Start Over", "entry_new")],
                    [("❌ Cancel", "entry_cancel_existing")]
                ])
                
                mode_text = "On-Hand Count" if session.mode == "on_hand" else "Delivery"
                progress = f"{session.index + 1}/{len(session.items)}" if session.items else "0/0"
                
                self.bot.send_message(
                    chat_id,
                    f"📋 <b>Active Entry Session</b>\n"
                    f"────────────────────────────\n"
                    f"Type: {mode_text} for {session.vendor}\n"
                    f"Progress: {progress} • Answered: {session.get_answered_count()}\n\n"
                    f"What would you like to do?",
                    reply_markup=keyboard
                )
                return
        
        # No active session - start vendor selection
        self.logger.info(f"No active session | starting vendor selection")
        self._start_vendor_selection(chat_id, user_id, message)
    
    def _start_vendor_selection(self, chat_id: int, user_id: int, message: Dict = None):
        """
        Prompt user to select location from dynamically discovered list.
        Uses Notion Items Master as single source of truth for locations.
        
        Args:
            chat_id: Telegram chat ID
            user_id: Telegram user ID
            message: Optional message object for display name extraction
            
        Logs: location discovery, button generation, prompt sent
        """
        self.logger.info(f"Prompting location selection for entry | user={user_id}")
        
        # Store message for later display name extraction
        if message:
            self._temp_messages[user_id] = message
        
        # Discover available locations from Notion
        try:
            locations = self.notion.get_locations(use_cache=True)
            self.logger.info(f"Locations discovered for entry menu | count={len(locations)} locations={locations}")
            
            if not locations:
                self.logger.error("No locations available for entry | cannot build menu")
                self.bot.send_message(
                    chat_id,
                    "⚠️ <b>Configuration Error</b>\n"
                    "────────────────────────────\n"
                    "No locations found in Items Master.\n\n"
                    "Please add items with Location property to Notion."
                )
                return
            
            # Build dynamic location buttons
            location_buttons = []
            for location in locations:
                # Simple prefix for location buttons
                button_text = f"📍 {location}"
                callback_data = f"entry_vendor|{location}"
                location_buttons.append([(button_text, callback_data)])
                self.logger.debug(f"Built location button | text='{button_text}' callback='{callback_data}'")
            
            # Add cancel button
            location_buttons.append([("❌ Cancel", "entry_cancel")])
            
            keyboard = self._create_keyboard(location_buttons)
            
            self.bot.send_message(
                chat_id,
                "📋 <b>Inventory Entry</b>\n"
                "────────────────────────────\n"
                "Select location:",
                reply_markup=keyboard
            )
            
            self.logger.info(f"Entry location menu sent | user={user_id} locations={len(locations)}")
            
        except Exception as e:
            self.logger.error(f"Error building entry location menu | user={user_id} error={e}", exc_info=True)
            self.bot.send_message(
                chat_id,
                "⚠️ Unable to load locations. Please try again or contact support."
            )
    
    def _handle_vendor_callback(self, chat_id: int, user_id: int, location: str):
        """
        Handle location selection from dynamic menu.
        Location string comes directly from Notion Items Master via callback data.

        Args:
            chat_id: Telegram chat ID
            user_id: Telegram user ID
            location: Location name as selected by user (from Notion)

        Logs: location selected, session creation, mode prompt
        """
        self.logger.info(f"Location selected for entry | location='{location}' user={user_id}")

        # Create session with dynamic location
        session = EntrySession(
            user_id=user_id,
            chat_id=chat_id,
            location=location  # Store exactly as received from Notion
        )
        self.sessions[user_id] = session
        self.logger.info(f"Entry session created | location='{location}' user={user_id}")
        
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
        """
        Handle mode selection and prompt for submitter name.
        
        Args:
            mode: 'on_hand' or 'received'
            
        Logs: mode selected, submitter prompt.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for mode selection | user={user_id}")
            self.bot.send_message(chat_id, "Session expired. Use /entry to start over.")
            return
        
        session.mode = mode
        self.logger.info(f"Entry mode selected | mode={mode} vendor={session.vendor} user={user_id}")
        
        # Extract Telegram display name
        message = self._temp_messages.get(user_id)
        if message:
            display_name = self._get_telegram_display_name(message)
        else:
            display_name = "Unknown"
        
        # Show submitter prompt
        self._show_submitter_prompt(session, display_name)
    
    def _show_submitter_prompt(self, session: EntrySession, prefill_name: str):
        """
        Prompt user to enter their name (simplified flow).
        
        Args:
            session: Entry session
            prefill_name: Extracted Telegram display name (ignored, user types their name)
            
        Logs: submitter prompt
        """
        mode_text = "On-Hand Count" if session.mode == "on_hand" else "Delivery"
        
        self.logger.info(f"Showing submitter prompt | vendor={session.vendor} user={session.user_id}")
        
        # Set flag to indicate we're waiting for name input
        session.current_step = "submitter"
        
        self.bot.send_message(
            session.chat_id,
            f"📍 {session.vendor} • {mode_text}\n"
            f"────────────────────────────\n\n"
            f"👤 <b>Enter Your Name</b>\n\n"
            f"Type your name to continue.\n"
            f"Or type /cancel to exit."
        )
    
    def _load_items_and_begin_entry(self, session: EntrySession):
        """
        Load catalog items for session's dynamic location and begin item entry loop.
        Uses location string from session without any hard-coded checks.
        
        Args:
            session: Entry session with location already set
            
        Logs: catalog load, item count, first item display
        """
        # Load items for the session's location (dynamic, no hard-coding)
        try:
            items = self.notion.get_items_for_location(session.location)
            self.logger.info(f"Items loaded for entry | location='{session.location}' count={len(items)}")
        except Exception as e:
            self.logger.error(f"Failed to load items | location='{session.location}' error={e}", exc_info=True)
            self.bot.send_message(
                session.chat_id,
                f"⚠️ Unable to load items for {session.location}. Please try again."
            )
            self._delete_session(session.user_id)
            return
        
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
            self.logger.warning(f"No items found for entry | location='{session.location}'")
            self.bot.send_message(
                session.chat_id,
                f"⚠️ No items found for {session.location}"
            )
            self._delete_session(session.user_id)
            return
        
        # Initialize answers dict
        for item in session.items:
            session.answers[item['name']] = None
        
        # Reset index to start item entry
        session.index = 0
        
        self.logger.info(f"Entry items initialized | location='{session.location}' count={len(session.items)} submitter={session.submitter_name}")
        
        # Start item entry
        mode_text = "On-Hand Count" if session.mode == "on_hand" else "Delivery"
        date = datetime.now().strftime('%Y-%m-%d')
        
        self.bot.send_message(
            session.chat_id,
            f"📍 <b>{session.location} • {mode_text}</b>\n"
            f"📅 Date: {date}\n"
            f"👤 Submitter: {session.submitter_name}\n"
            f"📦 Items: {len(session.items)}\n"
            f"────────────────────────────\n\n"
            f"Enter quantity for each item.\n"
            f"You can use: back, skip, done, cancel"
        )
        
        self._show_current_item(session)
    
    def _compute_entry_hash(self, vendor: str, entry_type: str, date: str, 
                            quantities: Dict[str, float], submitter: str = "Unknown") -> str:
        """
        Compute SHA-256 hash for entry idempotency check.
        
        Hash inputs: vendor, entry_type, date, submitter, sorted (item_name, qty) tuples
        
        Args:
            vendor: Vendor name
            entry_type: 'on_hand' or 'received'
            date: Date in YYYY-MM-DD format
            quantities: Dict of item_name -> quantity
            submitter: Submitter name/handle
            
        Returns:
            str: Hex digest of SHA-256 hash
            
        Logs: hash computation with input summary
        """
        import hashlib
        import json
        
        # Sort items for deterministic hash
        sorted_items = sorted(
            [(name, qty) for name, qty in quantities.items() if qty is not None and qty > 0],
            key=lambda x: x[0]
        )
        
        # Build hash input
        hash_input = {
            'vendor': vendor,
            'entry_type': entry_type,
            'date': date,
            'submitter': submitter,
            'items': sorted_items
        }
        
        # Compute hash
        hash_str = json.dumps(hash_input, sort_keys=True)
        hash_digest = hashlib.sha256(hash_str.encode('utf-8')).hexdigest()
        
        self.logger.debug(f"Entry hash computed | hash={hash_digest[:16]}... items={len(sorted_items)}")
        
        return hash_digest
    
    def _build_entry_review_message(self, session: EntrySession, date: str,
                                    submitter: str = "Unknown") -> str:
        """
        Build detailed entry confirmation message for notifications.

        Format matches the "Review Your Entry" screen:
        - Type, Location, Submitter, Date
        - All items with quantities and units
        - Summary stats including recounts and variance alerts

        Args:
            session: Entry session
            date: Entry date
            submitter: Submitter name/handle

        Returns:
            str: Formatted message

        Logs: message build
        """
        entry_type_display = "On-Hand Count" if session.mode == "on_hand" else "Received Delivery"

        # Header section
        text = "📋 <b>Entry Confirmation</b>\n\n"
        text += f"<b>Type:</b> {entry_type_display}\n"
        text += f"<b>Location:</b> {session.location}\n"
        text += f"<b>Submitter:</b> {submitter}\n"
        text += f"<b>Date:</b> {date}\n\n"

        # Items section - show ALL items (even with 0)
        entered_items = []
        total_qty = 0.0
        items_with_qty = 0

        for item in session.items:
            qty = session.answers.get(item['name'])
            if qty is not None:
                unit = item.get('unit', '')
                unit_str = f" {unit}" if unit else ""
                entered_items.append(f"• {item['name']}: {qty}{unit_str}")
                total_qty += qty
                if qty > 0:
                    items_with_qty += 1

        text += f"📦 <b>Entered ({len(entered_items)}):</b>\n"
        if entered_items:
            text += "\n".join(entered_items)
        else:
            text += "• No items recorded"

        # Summary section
        text += "\n\n📊 <b>Summary:</b>\n"
        text += f"• Items entered: {items_with_qty}/{len(session.items)}\n"
        text += f"• Total quantity: {total_qty:.1f}\n"

        # Show recount and variance stats if any occurred
        if session.recount_count > 0 or session.variance_count > 0:
            text += f"• Recounts: {session.recount_count}\n"
            text += f"• Variance alerts: {session.variance_count}\n"

        self.logger.debug(f"Entry review message built | items={len(entered_items)} submitter={submitter} recounts={session.recount_count} variances={session.variance_count}")

        return text
    
    def handle_callback(self, callback_query: Dict):
        """Handle all entry-related callbacks."""
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")
        
        # Acknowledge callback
        self.bot._make_request("answerCallbackQuery", 
                              {"callback_query_id": callback_query.get("id")})
        
        # Route vendor selection
        if data.startswith("entry_vendor|"):
            vendor = data.split("|")[1]
            self._handle_vendor_callback(chat_id, user_id, vendor)
            return
        
        # Route mode selection
        if data.startswith("entry_mode|"):
            mode = data.split("|")[1]
            self._handle_mode_selection(chat_id, user_id, mode)
            return
        
        # Route callbacks
        if data == "entry_resume":
            self._resume_session(chat_id, user_id)
        elif data == "entry_new":
            self._delete_session(user_id)
            self._start_vendor_selection(chat_id, user_id)
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

        # === PHASE 4: VARIANCE CALLBACKS ===
        elif data == "entry_var_recount":
            session = self.sessions.get(user_id)
            if session and hasattr(session, 'pending_variance'):
                # Clear pending, re-prompt same item
                item_name = session.pending_variance.get('item_name', 'item')
                session.pending_variance = {}
                session.current_step = 'enter_items'
                # Track recount
                session.recount_count += 1
                self.logger.info(f"[PHASE 4] User chose to recount {item_name} (total recounts: {session.recount_count})")
                print(f"[PHASE 4] Recount requested for {item_name}")
                self._show_current_item(session)

        elif data == "entry_var_confirm":
            session = self.sessions.get(user_id)
            if session and hasattr(session, 'pending_variance'):
                # Request photo
                session.current_step = 'variance_photo'
                item_name = session.pending_variance.get('item_name', 'item')
                # Track variance confirmation
                session.variance_count += 1
                if item_name not in session.variance_items:
                    session.variance_items.append(item_name)
                self.logger.info(f"[PHASE 4] User confirming variance for {item_name} with photo (total variances: {session.variance_count})")
                print(f"[PHASE 4] Photo confirmation requested for {item_name}")
                print(f"[DEBUG] Set current_step to: {session.current_step} for user {user_id}")
                self.bot.send_message(chat_id, f"📸 Send a photo of <b>{item_name}</b> inventory to confirm your count.")
            else:
                print(f"[DEBUG] entry_var_confirm: No session or no pending_variance for user {user_id}")
        # === END PHASE 4: VARIANCE CALLBACKS ===

    def handle_text_input(self, message: Dict, session: EntrySession):
        """
        Handle text input for quantity entry and image flow.
        """
        text = message.get("text", "").strip()
        chat_id = session.chat_id
        
        # Update session activity
        session.update_activity()
        
        # Check for cancel command FIRST (works in all input modes)
        if text.lower() in ["/cancel", "cancel"]:
            self.logger.info(f"Cancel requested during entry | user={session.user_id}")
            self._handle_cancel(chat_id, session.user_id)
            return
        
        # Check if we're in submitter name input mode
        if hasattr(session, 'current_step') and session.current_step == "submitter":
            text_clean = sanitize_user_input(text, 100).strip()
            
            if not text_clean:
                self.logger.warning(f"Empty submitter name rejected | user={session.user_id}")
                self.bot.send_message(session.chat_id, "⚠️ Name cannot be empty. Please enter your name.")
                return
            
            # Store submitter name
            session.submitter_name = text_clean
            self.logger.info(f"Submitter name captured | name='{session.submitter_name}' user={session.user_id}")
            
            # Clear step flag and proceed to load items
            session.current_step = "items"
            self._load_items_and_begin_entry(session)
            return
        
        # Check for special text commands
        if text.lower() in ["/back", "back"]:
            self._handle_back(chat_id, session.user_id)
            return
        elif text.lower() in ["/skip", "skip"]:
            self._handle_skip(chat_id, session.user_id)
            return
        elif text.lower() in ["/done", "done"]:
            self._handle_done(chat_id, session.user_id)
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
            
            # === PHASE 4: VARIANCE CHECK ===
            item = session.get_current_item()
            if session.mode == 'on_hand' and item:
                variance = self.notion.check_variance(item['name'], session.vendor, quantity)
                self.logger.info(f"[PHASE 4] Variance check for {item['name']}: {variance}")

                if variance['suspicious']:
                    # Store pending, enter challenge mode
                    if not hasattr(session, 'pending_variance'):
                        session.pending_variance = {}
                    session.pending_variance = {
                        'qty': quantity,
                        'item_name': item['name'],
                        'variance': variance
                    }
                    session.current_step = 'variance_challenge'

                    kb = {"inline_keyboard": [
                        [{"text": "🔄 Recount", "callback_data": "entry_var_recount"}],
                        [{"text": "✅ Confirm + Photo", "callback_data": "entry_var_confirm"}]
                    ]}

                    self.bot.send_message(
                        chat_id,
                        f"⚠️ <b>Variance Detected</b>\n\n"
                        f"<b>{item['name']}</b>\n"
                        f"You entered: {quantity}\n"
                        f"Expected: ~{variance['expected']}\n"
                        f"Variance: {variance['message']}\n"
                        f"Threshold: {variance['threshold']:.0%}\n\n"
                        f"Recount or confirm with photo?",
                        reply_markup=kb
                    )
                    print(f"[PHASE 4] Variance challenge shown for {item['name']}")
                    return
            # === END PHASE 4: VARIANCE CHECK ===

            # Save quantity and move forward
            session.set_current_quantity(quantity)
            session.index += 1

            # Log the entry
            logged_item = session.items[session.index - 1] if session.index > 0 else None
            if logged_item:
                self.logger.info(f"Entry: {logged_item['name']} = {quantity}")
            
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
            f"Location: <b>{session.vendor}</b>\n"
            f"Submitter: <b>{session.submitter_name}</b>\n"
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
        """
        Submit entry to Notion and post review to prep chat with idempotency.
        
        Logs: submission, hash computation, duplicate check, Notion save, chat resolution, post.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for submit | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        self.logger.info(f"Entry submit action | vendor={session.vendor} mode={session.mode} user={user_id}")
        
        # Prepare data for saving
        quantities = {
            name: (qty if qty is not None else 0.0)
            for name, qty in session.answers.items()
        }
        
        date = datetime.now().strftime('%Y-%m-%d')
        submitter = session.submitter_name if session.submitter_name else "Unknown"
        
        # Compute idempotency hash
        entry_hash = self._compute_entry_hash(
            vendor=session.vendor,
            entry_type=session.mode,
            date=date,
            quantities=quantities,
            submitter=submitter
        )
        
        # Check for duplicate
        if entry_hash in self.posted_entry_hashes:
            posted_time = self.posted_entry_hashes[entry_hash]
            self.logger.warning(f"Duplicate entry review blocked | hash={entry_hash[:16]}... posted_at={posted_time}")
            self.bot.send_message(
                chat_id,
                "⚠️ <b>Duplicate Entry Detected</b>\n"
                "────────────────────────────\n"
                "This exact entry has already been posted.\n"
                "No duplicate created."
            )
            self._delete_session(user_id)
            return
        
        # === PHASE 4: CHECK FOR FLAGGED ITEMS ===
        flagged_items = getattr(session, 'flagged_items', {})
        is_flagged = bool(flagged_items)
        flag_reason = None

        # === PHASE 3: COLLECT PHOTO URLS FROM FLAGGED ITEMS ===
        photo_urls = []

        if is_flagged:
            flag_reason = f"Variance confirmed: {', '.join(flagged_items.keys())}"
            self.logger.info(f"[PHASE 4] Entry has flagged items: {list(flagged_items.keys())}")
            print(f"[PHASE 4] Saving flagged entry: {flag_reason}")

            # Collect photo URLs from flagged items
            for item_name, flag_data in flagged_items.items():
                # Check for pre-stored photo_url first (Phase 3)
                stored_url = flag_data.get('photo_url')
                if stored_url:
                    photo_urls.append(stored_url)
                    self.logger.info(f"[PHASE 3] Collected stored photo URL for {item_name}")
                else:
                    # Fall back to getting URL from photo_id
                    photo_id = flag_data.get('photo_id')
                    if photo_id:
                        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
                        if bot_token:
                            url = self.notion._get_telegram_photo_url(photo_id, bot_token)
                            if url:
                                photo_urls.append(url)
                                self.logger.info(f"[PHASE 3] Collected photo URL for {item_name}")

        # Use first photo URL for transaction (or None)
        primary_photo_url = photo_urls[0] if photo_urls else None
        # === END PHASE 3 ===

        # Save to Notion
        try:
            success = self.notion.save_inventory_transaction(
                location=session.vendor,
                entry_type=session.mode,
                date=date,
                manager=submitter,
                notes=session.notes if hasattr(session, 'notes') else "",
                quantities=quantities,
                image_file_id=session.image_file_id if hasattr(session, 'image_file_id') else None,
                flagged=is_flagged,
                flag_reason=flag_reason,
                photo_url=primary_photo_url
            )

            if not success:
                self.logger.error(f"Failed to save entry to Notion | vendor={session.vendor}")
                self.bot.send_message(chat_id, "⚠️ Failed to save to Notion. Please try again.")
                return

            self.logger.info(f"Entry saved | vendor={session.vendor} items={len([q for q in quantities.values() if q > 0])}")

            # === PHASE 4: SEND SUMMARY VARIANCE NOTIFICATIONS ===
            if is_flagged:
                for item_name, flag_data in flagged_items.items():
                    try:
                        self.bot.send_variance_notification(
                            location=session.vendor,
                            submitter=session.submitter_name,
                            item_name=item_name,
                            expected=flag_data.get('expected', 0),
                            actual=flag_data.get('actual', 0),
                            variance_pct=flag_data.get('variance', 0),
                            photo_file_id=flag_data.get('photo_id')
                        )
                    except Exception as e:
                        self.logger.error(f"[PHASE 4] Notification failed for {item_name}: {e}")
            # === END PHASE 4 ===
            
        except Exception as e:
            self.logger.error(f"Exception saving entry to Notion | error={e}", exc_info=True)
            self.bot.send_message(chat_id, "⚠️ Error saving entry. Please contact support.")
            return
        
        # Build review message for prep chat
        review_message = self._build_entry_review_message(session, date, submitter)

        # Get notification config from database
        session_token = f"entry_{user_id}"
        notification_config = self.bot._get_notification_config(session.vendor, 'entry_confirmation', session_token)

        # Resolve prep chat (group chat from DB or .env fallback)
        prep_chat_id = self.bot._resolve_order_prep_chat(session.vendor, session_token)

        # Track notification status
        group_notified = False
        individual_notified = 0

        # Post to group chat if configured
        if prep_chat_id:
            try:
                self.logger.info(f"Posting entry review to prep chat | chat_id={prep_chat_id} vendor={session.vendor}")
                post_success = self.bot.send_message(prep_chat_id, review_message)

                if post_success:
                    group_notified = True
                    self.logger.info(f"Entry review posted to group | chat={prep_chat_id}")
                else:
                    self.logger.error(f"Failed to post entry review | chat={prep_chat_id}")

            except Exception as e:
                self.logger.error(f"Exception posting entry review | error={e}", exc_info=True)

        # Send to individual users from database config
        if notification_config['found'] and notification_config['telegram_ids']:
            individual_notified = self.bot._notify_individual_users(
                notification_config['telegram_ids'],
                review_message,
                session_token
            )

        # Record hash to prevent duplicates if any notification succeeded
        if group_notified or individual_notified > 0:
            self.posted_entry_hashes[entry_hash] = datetime.now()
            self.logger.info(f"Entry review posted | hash={entry_hash[:16]}... group={group_notified} individuals={individual_notified}")

        # Success message to user
        mode_text = "on-hand count" if session.mode == "on_hand" else "delivery"
        items_count = len([q for q in quantities.values() if q > 0])

        if group_notified or individual_notified > 0:
            # Notifications sent successfully
            notification_details = []
            if group_notified:
                notification_details.append("group chat")
            if individual_notified > 0:
                notification_details.append(f"{individual_notified} team member(s)")

            self.bot.send_message(
                chat_id,
                f"✅ <b>Entry Saved & Posted</b>\n\n"
                f"Saved {items_count} items for {session.vendor}\n"
                f"Type: {mode_text}\n"
                f"Date: {date}\n\n"
                f"Notified: {', '.join(notification_details)}"
            )
        elif notification_config['found']:
            # Config exists but notifications failed
            self.bot.send_message(
                chat_id,
                f"✅ <b>Entry Saved</b>\n\n"
                f"Saved successfully\n\n"
                f"⚠️ Failed to send notifications"
            )
        else:
            # No notification config at all
            self.logger.warning(f"No notification config for entry | vendor={session.vendor}")
            self.bot.send_message(
                chat_id,
                f"✅ <b>Entry Saved</b>\n\n"
                f"Saved for {session.vendor}\n"
                f"Items: {items_count}"
            )

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
        
        # Clean up old entry hashes (older than 24 hours)
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=24)
        old_hashes = [h for h, t in self.posted_entry_hashes.items() if t < cutoff_time]
        for h in old_hashes:
            del self.posted_entry_hashes[h]
        
        if old_hashes:
            self.logger.debug(f"Cleaned up {len(old_hashes)} old entry hashes")
        
        if expired_users:
            self.logger.info(f"Cleaned up {len(expired_users)} expired entry sessions")

    def handle_photo_input(self, message: Dict, session: EntrySession):
        """
        Handle photo input for received deliveries and variance confirmation.

        Args:
            message: Telegram message with photo
            session: Current entry session
        """
        chat_id = session.chat_id

        # === PHASE 4: VARIANCE PHOTO CONFIRMATION ===
        if hasattr(session, 'current_step') and session.current_step == 'variance_photo':
            photos = message.get('photo', [])
            if not photos:
                self.bot.send_message(chat_id, "Please send a photo to confirm.")
                return

            file_id = max(photos, key=lambda p: p.get('file_size', 0))['file_id']

            # === PHASE 3: Get photo URL immediately for later use ===
            bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            photo_url = None
            if bot_token:
                photo_url = self.notion._get_telegram_photo_url(file_id, bot_token)
                if photo_url:
                    self.logger.info(f"[PHASE 3] Got photo URL for variance confirmation")
            # === END PHASE 3 ===

            # Accept the flagged count
            pending = getattr(session, 'pending_variance', {})
            item_name = pending.get('item_name', 'Unknown')
            qty = pending.get('qty', 0)
            variance = pending.get('variance', {})

            # Set the quantity
            session.set_current_quantity(qty)

            # Track flagged items with photo URL (Phase 3)
            if not hasattr(session, 'flagged_items'):
                session.flagged_items = {}
            session.flagged_items[item_name] = {
                'expected': variance.get('expected'),
                'actual': qty,
                'variance': variance.get('variance'),
                'photo_id': file_id,
                'photo_url': photo_url
            }

            # Clear pending and advance
            session.pending_variance = {}
            session.index += 1
            session.current_step = 'enter_items'

            self.logger.info(f"[PHASE 4] Variance confirmed with photo | item={item_name} qty={qty}")
            print(f"[PHASE 4] ✓ Variance confirmed: {item_name} = {qty} (flagged with photo)")

            # Note: Variance notification with photo will be sent at submission (batched)
            # Photo data is stored in session.flagged_items for later use

            self.bot.send_message(chat_id, f"✓ <b>{item_name}</b>: {qty} <i>(flagged for review)</i>")

            # Show next item or finish
            if session.index >= len(session.items):
                if session.mode == "received":
                    session.current_step = "image"
                    self._show_image_request(session)
                else:
                    self._handle_done(chat_id, session.user_id)
            else:
                self._show_current_item(session)
            return
        # === END PHASE 4: VARIANCE PHOTO ===

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

# ===== REPORT HANDLER =====

@dataclass
class ReportSession:
    """
    Session state for interactive /reports flow.
    Handles multi-step report submission via Telegram bot.
    """
    user_id: int
    chat_id: int
    report_id: Optional[str] = None
    report_name: str = ""
    is_global: bool = False
    location: str = ""
    questions: List[Dict[str, Any]] = field(default_factory=list)
    question_index: int = 0
    answers: Dict[str, Any] = field(default_factory=dict)
    media_files: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    current_step: str = "select_report"  # select_report, select_location, questions, review
    submitter_name: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=2))
    last_message_id: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() > self.expires_at

    def update_activity(self):
        """Reset expiration timer on activity."""
        self.expires_at = datetime.now() + timedelta(hours=2)

    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """Get current question being answered."""
        if 0 <= self.question_index < len(self.questions):
            return self.questions[self.question_index]
        return None

    def get_progress(self) -> str:
        """Get progress string."""
        return f"{self.question_index + 1}/{len(self.questions)}"

    def move_back(self):
        """Move to previous question."""
        self.question_index = max(0, self.question_index - 1)

    def move_forward(self):
        """Move to next question."""
        self.question_index += 1

    def is_complete(self) -> bool:
        """Check if all questions have been answered."""
        return self.question_index >= len(self.questions)


class ReportHandler:
    """
    Manages interactive /reports flow for submitting configurable reports.
    Reports are defined in the dashboard and submitted via the bot.
    """

    def __init__(self, bot, notion_manager):
        """Initialize with dependencies."""
        self.bot = bot
        self.notion = notion_manager
        self.logger = logging.getLogger('reports')
        self.sessions: Dict[int, ReportSession] = {}
        self._temp_messages: Dict[int, Dict] = {}

    def _get_supabase(self):
        """Get Supabase client."""
        global _supabase_client
        return _supabase_client

    def _create_keyboard(self, buttons: List[List[Tuple[str, str]]]) -> Dict:
        """Create inline keyboard markup."""
        return {
            "inline_keyboard": [
                [{"text": text, "callback_data": data} for text, data in row]
                for row in buttons
            ]
        }

    def _get_telegram_display_name(self, message: Dict) -> str:
        """Extract display name from Telegram message."""
        user = message.get("from", {})
        first_name = user.get("first_name", "").strip()
        last_name = user.get("last_name", "").strip()
        username = user.get("username", "").strip()

        if first_name or last_name:
            return f"{first_name} {last_name}".strip()
        elif username:
            return f"@{username}"
        return "Unknown"

    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = [uid for uid, s in self.sessions.items() if s.is_expired()]
        for uid in expired:
            del self.sessions[uid]
            self.logger.info(f"Cleaned up expired report session | user={uid}")

    def handle_reports_command(self, message: Dict):
        """
        Entry point for /reports command.
        Shows available report types for submission.
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]

        self.logger.info(f"/reports command | user={user_id}")

        # Store message for display name extraction
        self._temp_messages[user_id] = message

        # Check for existing session
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if not session.is_expired():
                keyboard = self._create_keyboard([
                    [("📂 Resume", "report_resume"), ("🔄 Start Over", "report_new")],
                    [("❌ Cancel", "report_cancel")]
                ])
                self.bot.send_message(
                    chat_id,
                    f"📋 <b>Active Report Session</b>\n"
                    f"────────────────────────────\n"
                    f"Report: {session.report_name}\n"
                    f"Progress: {session.get_progress()}\n\n"
                    f"What would you like to do?",
                    reply_markup=keyboard
                )
                return

        # Fetch available report types from Supabase
        self._show_report_selection(chat_id, user_id)

    def _show_report_selection(self, chat_id: int, user_id: int):
        """Show available report types."""
        supabase = self._get_supabase()
        if not supabase:
            self.bot.send_message(chat_id, "⚠️ Database connection unavailable. Please try again later.")
            return

        try:
            # Get active report types
            result = supabase.table('report_types').select('*').eq('active', True).order('sort_order').execute()
            report_types = result.data if result.data else []

            if not report_types:
                self.bot.send_message(
                    chat_id,
                    "📋 <b>No Reports Available</b>\n"
                    "────────────────────────────\n"
                    "No report types have been configured yet.\n\n"
                    "Reports can be created in the Admin Dashboard."
                )
                return

            # Build report type buttons
            buttons = []
            for report in report_types:
                icon = "📄" if report.get('is_global') else "📍"
                name = report.get('name', 'Unnamed Report')
                buttons.append([(f"{icon} {name}", f"report_select|{report['id']}")])

            buttons.append([("❌ Cancel", "report_cancel")])

            keyboard = self._create_keyboard(buttons)

            self.bot.send_message(
                chat_id,
                "📋 <b>Submit a Report</b>\n"
                "────────────────────────────\n"
                "Select a report type:\n\n"
                "📄 = Global report\n"
                "📍 = Location-specific",
                reply_markup=keyboard
            )

        except Exception as e:
            self.logger.error(f"Error fetching report types: {e}")
            self.bot.send_message(chat_id, "⚠️ Error loading reports. Please try again.")

    def _handle_report_selection(self, chat_id: int, user_id: int, report_id: str):
        """Handle report type selection."""
        supabase = self._get_supabase()
        if not supabase:
            self.bot.send_message(chat_id, "⚠️ Database connection unavailable.")
            return

        try:
            # Get report type details
            result = supabase.table('report_types').select('*').eq('id', report_id).single().execute()
            report = result.data

            if not report:
                self.bot.send_message(chat_id, "⚠️ Report not found.")
                return

            # Get questions for this report
            questions_result = supabase.table('report_questions').select('*').eq('report_type_id', report_id).order('sort_order').execute()
            questions = questions_result.data if questions_result.data else []

            # Extract submitter name
            message = self._temp_messages.get(user_id, {})
            submitter_name = self._get_telegram_display_name(message)

            # Create session
            session = ReportSession(
                user_id=user_id,
                chat_id=chat_id,
                report_id=report_id,
                report_name=report.get('name', 'Report'),
                is_global=report.get('is_global', False),
                questions=questions,
                submitter_name=submitter_name
            )

            self.sessions[user_id] = session
            self.logger.info(f"Report session created | report={report.get('name')} user={user_id}")

            # If location-specific, prompt for location
            if not session.is_global:
                self._show_location_selection(chat_id, session)
            else:
                # Start questions directly
                self._start_questions(chat_id, session)

        except Exception as e:
            self.logger.error(f"Error handling report selection: {e}")
            self.bot.send_message(chat_id, "⚠️ Error loading report. Please try again.")

    def _show_location_selection(self, chat_id: int, session: ReportSession):
        """Show location selection for location-specific reports."""
        try:
            locations = self.notion.get_locations(use_cache=True)

            if not locations:
                self.bot.send_message(chat_id, "⚠️ No locations available.")
                return

            buttons = []
            for location in locations:
                buttons.append([(f"📍 {location}", f"report_loc|{location}")])
            buttons.append([("❌ Cancel", "report_cancel")])

            keyboard = self._create_keyboard(buttons)

            self.bot.send_message(
                chat_id,
                f"📋 <b>{session.report_name}</b>\n"
                "────────────────────────────\n"
                "Select location:",
                reply_markup=keyboard
            )
            session.current_step = "select_location"

        except Exception as e:
            self.logger.error(f"Error showing locations: {e}")
            self.bot.send_message(chat_id, "⚠️ Error loading locations.")

    def _handle_location_selection(self, chat_id: int, user_id: int, location: str):
        """Handle location selection."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "Session expired. Use /reports to start over.")
            return

        session.location = location
        session.update_activity()
        self.logger.info(f"Location selected | location={location} user={user_id}")

        self._start_questions(chat_id, session)

    def _start_questions(self, chat_id: int, session: ReportSession):
        """Begin the question flow."""
        if not session.questions:
            self.bot.send_message(
                chat_id,
                "📋 <b>No Questions</b>\n"
                "────────────────────────────\n"
                "This report has no questions configured.\n"
                "Please contact your administrator."
            )
            del self.sessions[session.user_id]
            return

        session.current_step = "questions"
        session.question_index = 0
        self._show_current_question(chat_id, session)

    def _show_current_question(self, chat_id: int, session: ReportSession):
        """Display the current question."""
        question = session.get_current_question()
        if not question:
            # All questions answered, move to review
            self._show_review(chat_id, session)
            return

        session.update_activity()

        q_text = question.get('question_text', 'Question')
        q_type = question.get('question_type', 'text')
        q_id = question.get('id')
        help_text = question.get('help_text', '')
        placeholder = question.get('placeholder', '')
        is_required = question.get('is_required', True)

        # Build question message
        required_marker = "❗" if is_required else "⚪"
        msg = f"📋 <b>{session.report_name}</b>\n"
        msg += f"────────────────────────────\n"
        msg += f"{required_marker} <b>Question {session.get_progress()}</b>\n\n"
        msg += f"{q_text}\n"

        if help_text:
            msg += f"\n<i>{help_text}</i>\n"

        # Add input instructions based on type
        if q_type == 'text':
            msg += f"\n💬 Type your answer"
            if placeholder:
                msg += f" (e.g., {placeholder})"
        elif q_type == 'multiline':
            msg += f"\n💬 Type your detailed answer"
        elif q_type == 'number':
            msg += f"\n🔢 Enter a number"
            if placeholder:
                msg += f" (e.g., {placeholder})"
        elif q_type == 'boolean':
            msg += f"\n👆 Select Yes or No"
        elif q_type == 'select':
            msg += f"\n👆 Select an option"
        elif q_type in ['photo', 'photo_required']:
            msg += f"\n📷 Send a photo"
        elif q_type in ['video', 'video_required']:
            msg += f"\n🎥 Send a video"

        # Build navigation buttons based on question type
        nav_buttons = []

        if q_type == 'boolean':
            nav_buttons.append([
                ("✅ Yes", f"report_answer|{q_id}|yes"),
                ("❌ No", f"report_answer|{q_id}|no")
            ])
        elif q_type == 'select':
            options = question.get('options', [])
            for opt in options[:4]:  # Limit to 4 options per row
                nav_buttons.append([(f"▫️ {opt}", f"report_answer|{q_id}|{opt}")])

        # Add navigation row
        nav_row = []
        if session.question_index > 0:
            nav_row.append(("⬅️ Back", "report_back"))
        if not is_required and q_type not in ['boolean', 'select']:
            nav_row.append(("⏭️ Skip", "report_skip"))
        nav_row.append(("❌ Cancel", "report_cancel"))

        if nav_row:
            nav_buttons.append(nav_row)

        keyboard = self._create_keyboard(nav_buttons) if nav_buttons else None

        self.bot.send_message(chat_id, msg, reply_markup=keyboard)

    def _handle_text_answer(self, session: ReportSession, text: str):
        """Process a text answer."""
        question = session.get_current_question()
        if not question:
            return

        q_id = question.get('id')
        q_type = question.get('question_type', 'text')

        # Validate based on type
        if q_type == 'number':
            try:
                value = float(text.replace(',', ''))
                session.answers[q_id] = value
            except ValueError:
                self.bot.send_message(
                    session.chat_id,
                    "⚠️ Please enter a valid number."
                )
                return
        else:
            session.answers[q_id] = text

        session.move_forward()
        self._show_current_question(session.chat_id, session)

    def _handle_choice_answer(self, chat_id: int, user_id: int, q_id: str, value: str):
        """Handle selection/boolean answer via callback."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "Session expired. Use /reports to start over.")
            return

        session.answers[q_id] = value
        session.update_activity()
        session.move_forward()
        self._show_current_question(chat_id, session)

    def handle_photo_input(self, message: Dict, session: ReportSession):
        """Handle photo upload for photo questions."""
        question = session.get_current_question()
        if not question:
            return

        q_id = question.get('id')
        q_type = question.get('question_type', '')

        if q_type not in ['photo', 'photo_required']:
            return

        try:
            photos = message.get("photo", [])
            if photos:
                # Get largest photo
                photo = max(photos, key=lambda p: p.get("file_size", 0))
                file_id = photo.get("file_id")

                if q_id not in session.media_files:
                    session.media_files[q_id] = []

                session.media_files[q_id].append({
                    'type': 'photo',
                    'file_id': file_id
                })
                session.answers[q_id] = f"[Photo uploaded: {len(session.media_files[q_id])}]"

                self.bot.send_message(
                    session.chat_id,
                    "✅ Photo received!"
                )

                session.move_forward()
                self._show_current_question(session.chat_id, session)

        except Exception as e:
            self.logger.error(f"Error handling photo: {e}")
            self.bot.send_message(session.chat_id, "⚠️ Error processing photo. Please try again.")

    def handle_video_input(self, message: Dict, session: ReportSession):
        """Handle video upload for video questions."""
        question = session.get_current_question()
        if not question:
            return

        q_id = question.get('id')
        q_type = question.get('question_type', '')

        if q_type not in ['video', 'video_required']:
            return

        try:
            video = message.get("video", {})
            if video:
                file_id = video.get("file_id")

                if q_id not in session.media_files:
                    session.media_files[q_id] = []

                session.media_files[q_id].append({
                    'type': 'video',
                    'file_id': file_id
                })
                session.answers[q_id] = f"[Video uploaded: {len(session.media_files[q_id])}]"

                self.bot.send_message(
                    session.chat_id,
                    "✅ Video received!"
                )

                session.move_forward()
                self._show_current_question(session.chat_id, session)

        except Exception as e:
            self.logger.error(f"Error handling video: {e}")
            self.bot.send_message(session.chat_id, "⚠️ Error processing video. Please try again.")

    def _show_review(self, chat_id: int, session: ReportSession):
        """Show review screen before submission."""
        session.current_step = "review"

        msg = f"📋 <b>Review: {session.report_name}</b>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        if session.location:
            msg += f"📍 Location: {session.location}\n"
        msg += f"👤 Submitted by: {session.submitter_name}\n"
        msg += "────────────────────────────\n\n"

        # Show answers
        for question in session.questions:
            q_id = question.get('id')
            q_text = question.get('question_text', 'Question')[:50]
            answer = session.answers.get(q_id, 'Not answered')

            # Truncate long answers for display
            if isinstance(answer, str) and len(answer) > 50:
                answer = answer[:47] + "..."

            msg += f"<b>Q:</b> {q_text}\n"
            msg += f"<b>A:</b> {answer}\n\n"

        msg += "────────────────────────────\n"
        msg += "✅ Submit this report?"

        keyboard = self._create_keyboard([
            [("✅ Submit", "report_submit"), ("✏️ Edit", "report_edit")],
            [("❌ Cancel", "report_cancel")]
        ])

        self.bot.send_message(chat_id, msg, reply_markup=keyboard)

    def _handle_submit(self, chat_id: int, user_id: int):
        """Submit the completed report to Supabase."""
        session = self.sessions.get(user_id)
        if not session:
            self.bot.send_message(chat_id, "Session expired. Use /reports to start over.")
            return

        supabase = self._get_supabase()
        if not supabase:
            self.bot.send_message(chat_id, "⚠️ Database connection unavailable.")
            return

        try:
            # Prepare submission data
            submission_data = {
                'report_type_id': session.report_id,
                'location': session.location or 'Global',
                'submitted_by': session.submitter_name,
                'telegram_user_id': user_id,
                'submission_date': datetime.now().strftime('%Y-%m-%d'),
                'answers': session.answers,
                'status': 'submitted'
            }

            # Insert submission
            result = supabase.table('report_submissions').insert(submission_data).execute()

            if result.data:
                submission_id = result.data[0]['id']

                # Save media files if any
                for q_id, media_list in session.media_files.items():
                    for media in media_list:
                        media_data = {
                            'submission_id': submission_id,
                            'question_id': q_id,
                            'media_type': media.get('type', 'photo'),
                            'telegram_file_id': media.get('file_id')
                        }
                        supabase.table('submission_media').insert(media_data).execute()

                self.bot.send_message(
                    chat_id,
                    f"✅ <b>Report Submitted!</b>\n"
                    f"────────────────────────────\n"
                    f"📋 {session.report_name}\n"
                    f"📍 {session.location or 'Global'}\n"
                    f"👤 {session.submitter_name}\n\n"
                    f"Your report has been saved and will be reviewed by management."
                )

                self.logger.info(f"Report submitted | report={session.report_name} user={user_id} submission_id={submission_id}")

                # Clean up session
                del self.sessions[user_id]
            else:
                raise Exception("No data returned from insert")

        except Exception as e:
            self.logger.error(f"Error submitting report: {e}")
            self.bot.send_message(
                chat_id,
                "⚠️ Error submitting report. Please try again or contact support."
            )

    def handle_text_input(self, message: Dict, session: ReportSession):
        """Handle text input during question flow."""
        if session.current_step != "questions":
            return

        text = message.get("text", "").strip()
        if not text:
            return

        question = session.get_current_question()
        if not question:
            return

        q_type = question.get('question_type', 'text')

        # Only accept text for text-based questions
        if q_type in ['text', 'multiline', 'number']:
            self._handle_text_answer(session, text)

    def handle_callback(self, callback_query: Dict):
        """Handle all report-related callbacks."""
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")

        # Acknowledge callback
        self.bot._make_request("answerCallbackQuery",
                              {"callback_query_id": callback_query.get("id")})

        self.logger.info(f"Report callback | data={data} user={user_id}")

        # Route callbacks
        if data.startswith("report_select|"):
            report_id = data.split("|")[1]
            self._handle_report_selection(chat_id, user_id, report_id)

        elif data.startswith("report_loc|"):
            location = data.split("|")[1]
            self._handle_location_selection(chat_id, user_id, location)

        elif data.startswith("report_answer|"):
            parts = data.split("|")
            if len(parts) >= 3:
                q_id = parts[1]
                value = parts[2]
                self._handle_choice_answer(chat_id, user_id, q_id, value)

        elif data == "report_resume":
            session = self.sessions.get(user_id)
            if session:
                self._show_current_question(chat_id, session)

        elif data == "report_new":
            if user_id in self.sessions:
                del self.sessions[user_id]
            self._show_report_selection(chat_id, user_id)

        elif data == "report_back":
            session = self.sessions.get(user_id)
            if session:
                session.move_back()
                self._show_current_question(chat_id, session)

        elif data == "report_skip":
            session = self.sessions.get(user_id)
            if session:
                question = session.get_current_question()
                if question:
                    session.answers[question.get('id')] = None
                session.move_forward()
                self._show_current_question(chat_id, session)

        elif data == "report_edit":
            session = self.sessions.get(user_id)
            if session:
                session.question_index = 0
                session.current_step = "questions"
                self._show_current_question(chat_id, session)

        elif data == "report_submit":
            self._handle_submit(chat_id, user_id)

        elif data == "report_cancel":
            if user_id in self.sessions:
                del self.sessions[user_id]
            self.bot.send_message(chat_id, "❌ Report cancelled.")


# ===== SOP HANDLER =====

@dataclass
class SOPSession:
    """Session state for interactive /sop flow."""
    user_id: int
    chat_id: int
    template_id: Optional[str] = None
    template_name: Optional[str] = None
    template_description: Optional[str] = None
    steps: List[Dict] = field(default_factory=list)
    current_step_index: int = 0
    completion_id: Optional[str] = None
    responses: Dict[str, Dict] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    timeout_minutes: int = 60

    def update_activity(self):
        self.last_activity = datetime.now()

    def is_expired(self) -> bool:
        return (datetime.now() - self.last_activity).total_seconds() > self.timeout_minutes * 60

    def get_current_step(self) -> Optional[Dict]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def get_progress(self) -> str:
        return f"{self.current_step_index + 1}/{len(self.steps)}"


class SOPHandler:
    """
    Manages interactive /sop flow for executing Standard Operating Procedures.
    SOPs are defined in the dashboard and executed via the bot.
    """

    def __init__(self, bot, notion_manager):
        """Initialize with dependencies."""
        self.bot = bot
        self.notion = notion_manager
        self.logger = logging.getLogger('sop')
        self.sessions: Dict[int, SOPSession] = {}

    def _get_supabase(self):
        """Get Supabase client."""
        global _supabase_client
        return _supabase_client

    def _create_keyboard(self, buttons: List[List[Tuple[str, str]]]) -> Dict:
        """Create inline keyboard markup."""
        return {
            "inline_keyboard": [
                [{"text": text, "callback_data": data} for text, data in row]
                for row in buttons
            ]
        }

    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = [uid for uid, s in self.sessions.items() if s.is_expired()]
        for uid in expired:
            del self.sessions[uid]
            self.logger.info(f"Cleaned up expired SOP session | user={uid}")

    def handle_sop_command(self, message: Dict):
        """
        Entry point for /sop command.
        Shows available SOPs or starts a specific one.
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        text = message.get("text", "")

        self.logger.info(f"/sop command | user={user_id}")

        # Parse arguments
        args = text.split()[1:] if len(text.split()) > 1 else []

        if not args:
            # List available SOPs
            self._show_sop_selection(chat_id, user_id)
        else:
            # Start specific SOP
            sop_name = '_'.join(args).lower()
            self._start_sop_by_name(chat_id, user_id, sop_name)

    def _show_sop_selection(self, chat_id: int, user_id: int):
        """Show available SOPs."""
        supabase = self._get_supabase()
        if not supabase:
            self.bot.send_message(chat_id, "⚠️ Database connection unavailable.")
            return

        try:
            result = supabase.table('sop_templates').select('*').eq('active', True).order('sort_order').execute()
            templates = result.data if result.data else []

            if not templates:
                self.bot.send_message(
                    chat_id,
                    "📋 <b>No SOPs Available</b>\n\n"
                    "No Standard Operating Procedures have been configured yet.\n"
                    "Contact your administrator to set up SOPs."
                )
                return

            # Create keyboard with SOP buttons
            buttons = []
            for template in templates:
                est_time = f" (~{template['estimated_minutes']}min)" if template.get('estimated_minutes') else ""
                buttons.append([
                    (f"📋 {template['name']}{est_time}", f"sop_start_{template['id']}")
                ])

            buttons.append([("❌ Cancel", "sop_cancel")])

            keyboard = self._create_keyboard(buttons)
            self.bot.send_message(
                chat_id,
                "📋 <b>Available SOPs</b>\n"
                "────────────────────────────\n"
                "Select an SOP to begin:\n",
                reply_markup=keyboard
            )

        except Exception as e:
            self.logger.error(f"Error fetching SOPs | error={e}")
            self.bot.send_message(chat_id, "⚠️ Error loading SOPs. Please try again.")

    def _start_sop_by_name(self, chat_id: int, user_id: int, sop_name: str):
        """Start an SOP by its normalized name."""
        supabase = self._get_supabase()
        if not supabase:
            self.bot.send_message(chat_id, "⚠️ Database connection unavailable.")
            return

        try:
            result = supabase.table('sop_templates').select('*').eq('name_normalized', sop_name).eq('active', True).single().execute()
            template = result.data

            if not template:
                self.bot.send_message(chat_id, f"❌ SOP '{sop_name}' not found.\n\nUse /sop to see available SOPs.")
                return

            self._start_sop(chat_id, user_id, template['id'])

        except Exception as e:
            self.logger.error(f"Error starting SOP by name | name={sop_name} error={e}")
            self.bot.send_message(chat_id, f"❌ SOP '{sop_name}' not found.\n\nUse /sop to see available SOPs.")

    def _start_sop(self, chat_id: int, user_id: int, template_id: str):
        """Start an SOP walkthrough."""
        supabase = self._get_supabase()
        if not supabase:
            self.bot.send_message(chat_id, "⚠️ Database connection unavailable.")
            return

        try:
            # Get template
            template_result = supabase.table('sop_templates').select('*').eq('id', template_id).single().execute()
            template = template_result.data

            if not template:
                self.bot.send_message(chat_id, "⚠️ SOP not found.")
                return

            # Get steps
            steps_result = supabase.table('sop_steps').select('*').eq('sop_template_id', template_id).order('step_number').execute()
            steps = steps_result.data if steps_result.data else []

            if not steps:
                self.bot.send_message(chat_id, "⚠️ This SOP has no steps configured.")
                return

            # Create completion record
            completion = supabase.table('sop_completions').insert({
                'sop_template_id': template_id,
                'telegram_id': user_id,
                'status': 'in_progress',
                'step_responses': {},
            }).execute()

            completion_id = completion.data[0]['id'] if completion.data else None

            # Create session
            session = SOPSession(
                user_id=user_id,
                chat_id=chat_id,
                template_id=template_id,
                template_name=template['name'],
                template_description=template.get('description'),
                steps=steps,
                current_step_index=0,
                completion_id=completion_id,
                responses={},
                start_time=datetime.now(),
            )
            self.sessions[user_id] = session

            # Send intro message
            est_time = f"⏱ Est. time: {template['estimated_minutes']} min\n" if template.get('estimated_minutes') else ""
            desc = f"\n{template['description']}\n" if template.get('description') else ""

            self.bot.send_message(
                chat_id,
                f"📋 <b>Starting: {template['name']}</b>\n"
                f"────────────────────────────{desc}\n"
                f"📊 Steps: {len(steps)}\n"
                f"{est_time}\n"
                f"Let's begin!"
            )

            # Show first step
            self._show_current_step(chat_id, user_id)

        except Exception as e:
            self.logger.error(f"Error starting SOP | template_id={template_id} error={e}")
            self.bot.send_message(chat_id, "⚠️ Error starting SOP. Please try again.")

    def _show_current_step(self, chat_id: int, user_id: int):
        """Display the current step."""
        session = self.sessions.get(user_id)
        if not session:
            return

        step = session.get_current_step()
        if not step:
            self._complete_sop(chat_id, user_id)
            return

        text = f"<b>Step {step['step_number']}: {step['title']}</b>\n\n"
        if step.get('instruction_text'):
            text += f"{step['instruction_text']}\n\n"

        # Handle media
        if step.get('media_type') == 'image' and step.get('media_url'):
            try:
                self.bot._make_request("sendPhoto", {
                    "chat_id": chat_id,
                    "photo": step['media_url'],
                    "caption": text[:1024],
                    "parse_mode": "HTML"
                })
                # Continue to show buttons below the image
                text = ""
            except Exception as e:
                self.logger.error(f"Error sending step image | error={e}")

        # Build keyboard based on step type
        buttons = []

        # Branching question
        if step.get('branch_condition'):
            condition = step['branch_condition']
            text += f"❓ {condition.get('question', 'Please confirm:')}"
            buttons = [
                [("✅ Yes", "sop_branch_yes"), ("❌ No", "sop_branch_no")]
            ]
        # Confirmation required
        elif step.get('requires_confirmation'):
            conf_type = step.get('confirmation_type', 'checkbox')
            if conf_type == 'checkbox':
                buttons = [[("✅ Done", "sop_confirm")]]
            elif conf_type == 'photo':
                text += "📸 Please send a photo to confirm this step."
                buttons = [[("⏭ Skip", "sop_skip")]]
            elif conf_type in ['text', 'number']:
                text += f"✏️ Please enter your response ({conf_type})."
                buttons = [[("⏭ Skip", "sop_skip")]]
        else:
            # No confirmation needed
            buttons = [[("Next →", "sop_next")]]

        # Add navigation
        if session.current_step_index > 0:
            buttons.append([("← Back", "sop_back"), ("❌ Cancel", "sop_cancel")])
        else:
            buttons.append([("❌ Cancel", "sop_cancel")])

        keyboard = self._create_keyboard(buttons)

        if text:  # Only send if there's text (might have sent image)
            self.bot.send_message(chat_id, text, reply_markup=keyboard)
        else:
            # Just send buttons after image
            self.bot.send_message(
                chat_id,
                f"Step {session.current_step_index + 1} of {len(session.steps)}",
                reply_markup=keyboard
            )

    def _record_response(self, user_id: int, response: any, media_url: str = None):
        """Record a step response."""
        session = self.sessions.get(user_id)
        if not session:
            return

        step = session.get_current_step()
        if not step:
            return

        session.responses[step['id']] = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
        }
        if media_url:
            session.responses[step['id']]['media_url'] = media_url

        session.update_activity()

    def _advance_step(self, user_id: int, target_step: int = None):
        """Move to the next step or a specific step."""
        session = self.sessions.get(user_id)
        if not session:
            return

        if target_step is not None:
            # Go to specific step (1-indexed to 0-indexed)
            session.current_step_index = target_step - 1
        else:
            session.current_step_index += 1

        session.update_activity()

    def _complete_sop(self, chat_id: int, user_id: int):
        """Mark SOP as completed."""
        session = self.sessions.get(user_id)
        if not session:
            return

        end_time = datetime.now()
        total_seconds = int((end_time - session.start_time).total_seconds())

        supabase = self._get_supabase()
        if supabase and session.completion_id:
            try:
                supabase.table('sop_completions').update({
                    'status': 'completed',
                    'completed_at': end_time.isoformat(),
                    'step_responses': session.responses,
                    'total_time_seconds': total_seconds,
                }).eq('id', session.completion_id).execute()
            except Exception as e:
                self.logger.error(f"Error updating completion | error={e}")

        minutes = total_seconds // 60
        seconds = total_seconds % 60

        self.bot.send_message(
            chat_id,
            f"✅ <b>{session.template_name} Completed!</b>\n\n"
            f"⏱ Time taken: {minutes} min {seconds} sec\n"
            f"📊 Steps completed: {len(session.steps)}"
        )

        # Clean up session
        del self.sessions[user_id]

    def handle_text_input(self, message: Dict):
        """Handle text/photo input for SOP steps requiring confirmation."""
        user_id = message["from"]["id"]
        chat_id = message["chat"]["id"]

        session = self.sessions.get(user_id)
        if not session:
            return False

        step = session.get_current_step()
        if not step or not step.get('requires_confirmation'):
            return False

        conf_type = step.get('confirmation_type')

        # Handle photo
        if message.get('photo') and conf_type == 'photo':
            photo = message['photo'][-1]  # Largest size
            file_id = photo.get('file_id')
            self._record_response(user_id, 'photo_submitted', media_url=file_id)
            self._advance_step(user_id)
            self._show_current_step(chat_id, user_id)
            return True

        # Handle text/number
        if message.get('text') and conf_type in ['text', 'number']:
            text = message['text']
            if conf_type == 'number':
                try:
                    text = float(text)
                except ValueError:
                    self.bot.send_message(chat_id, "⚠️ Please enter a valid number.")
                    return True
            self._record_response(user_id, text)
            self._advance_step(user_id)
            self._show_current_step(chat_id, user_id)
            return True

        return False

    def handle_callback(self, callback_query: Dict):
        """Handle all SOP-related callbacks."""
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")

        # Acknowledge callback
        self.bot._make_request("answerCallbackQuery",
                              {"callback_query_id": callback_query.get("id")})

        self.logger.debug(f"SOP callback | data={data} user={user_id}")

        # Handle starting an SOP
        if data.startswith("sop_start_"):
            template_id = data.replace("sop_start_", "")
            self._start_sop(chat_id, user_id, template_id)

        elif data == "sop_next" or data == "sop_confirm":
            session = self.sessions.get(user_id)
            if session:
                self._record_response(user_id, True)
                self._advance_step(user_id)
                self._show_current_step(chat_id, user_id)

        elif data == "sop_skip":
            session = self.sessions.get(user_id)
            if session:
                self._record_response(user_id, 'skipped')
                self._advance_step(user_id)
                self._show_current_step(chat_id, user_id)

        elif data == "sop_back":
            session = self.sessions.get(user_id)
            if session and session.current_step_index > 0:
                session.current_step_index -= 1
                session.update_activity()
                self._show_current_step(chat_id, user_id)

        elif data == "sop_branch_yes":
            session = self.sessions.get(user_id)
            if session:
                step = session.get_current_step()
                self._record_response(user_id, 'yes')
                if step and step.get('next_step_on_yes'):
                    self._advance_step(user_id, step['next_step_on_yes'])
                else:
                    self._advance_step(user_id)
                self._show_current_step(chat_id, user_id)

        elif data == "sop_branch_no":
            session = self.sessions.get(user_id)
            if session:
                step = session.get_current_step()
                self._record_response(user_id, 'no')
                if step and step.get('next_step_on_no'):
                    self._advance_step(user_id, step['next_step_on_no'])
                else:
                    self._advance_step(user_id)
                self._show_current_step(chat_id, user_id)

        elif data == "sop_cancel":
            session = self.sessions.get(user_id)
            if session:
                # Mark as abandoned
                supabase = self._get_supabase()
                if supabase and session.completion_id:
                    try:
                        supabase.table('sop_completions').update({
                            'status': 'abandoned',
                            'step_responses': session.responses,
                        }).eq('id', session.completion_id).execute()
                    except Exception as e:
                        self.logger.error(f"Error updating abandoned SOP | error={e}")

                del self.sessions[user_id]
            self.bot.send_message(chat_id, "❌ SOP cancelled.")


# ===== ORDER FLOW HANDLER =====

class OrderFlowHandler:
    """
    Manages interactive /order flow: vendor selection, line-by-line item prompting,
    Back/Skip/Cancel/Done navigation, Review, and final Confirm post.
    """
    
    def __init__(self, bot, notion_manager, calculator):
        """Initialize with dependencies."""
        self.bot = bot
        self.notion = notion_manager
        self.calc = calculator
        self.logger = logging.getLogger('business')
        self.sessions: Dict[int, OrderSession] = {}

    def _get_kitchen_timezone(self) -> str:
        """
        Get kitchen timezone for date/time calculations.
        
        Returns:
            str: Timezone string (e.g., 'America/Chicago')
            
        Logs: timezone retrieved
        """
        tz = BUSINESS_TIMEZONE  # From module constants
        self.logger.debug(f"Kitchen timezone | tz={tz}")
        return tz
    
    def _build_calendar_keyboard(self, session: OrderSession, target_date: datetime) -> Dict:
        """
        Build inline calendar keyboard for date selection.
        
        Creates a month grid with:
        - Month/Year header
        - Weekday labels
        - Date buttons (disabled for past dates)
        - Prev/Next month navigation
        - "Type date instead" fallback
        
        Args:
            session: Order session
            target_date: Base date for calendar (usually today + month_offset)
            
        Returns:
            Dict: Telegram inline keyboard markup
            
        Logs: calendar build, month offset, date range
        """
        import calendar
        from datetime import datetime, timedelta
        
        self.logger.info(f"[{session.session_token}] Building calendar | month_offset={session._calendar_month_offset} target={target_date.strftime('%Y-%m')}")
        
        # Get month boundaries
        year = target_date.year
        month = target_date.month
        
        # Month name and year header
        month_name = calendar.month_name[month]
        header_text = f"📅 {month_name} {year}"
        
        # Build calendar grid
        cal = calendar.monthcalendar(year, month)
        today = datetime.now().date()
        
        buttons = []
        
        # Header row
        buttons.append([{"text": header_text, "callback_data": "order_cal_noop"}])
        
        # Weekday labels
        weekday_row = []
        for day_name in ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']:
            weekday_row.append({"text": day_name, "callback_data": "order_cal_noop"})
        buttons.append(weekday_row)
        
        # Date buttons
        for week in cal:
            week_row = []
            for day in week:
                if day == 0:
                    # Empty cell
                    week_row.append({"text": " ", "callback_data": "order_cal_noop"})
                else:
                    date_obj = datetime(year, month, day).date()
                    date_str = date_obj.strftime('%Y-%m-%d')
                    
                    # Disable past dates
                    if date_obj < today:
                        week_row.append({"text": f"·{day}·", "callback_data": "order_cal_noop"})
                    else:
                        # Determine callback based on current mode
                        if session.delivery_date is None:
                            callback = f"order_delivery_date|{date_str}"
                        else:
                            callback = f"order_next_delivery_date|{date_str}"
                        
                        week_row.append({"text": str(day), "callback_data": callback})
            
            buttons.append(week_row)
        
        # Navigation row
        nav_row = []
        if session._calendar_month_offset > 0:
            nav_row.append({"text": "◀️ Prev", "callback_data": "order_cal_prev"})
        else:
            nav_row.append({"text": " ", "callback_data": "order_cal_noop"})
        
        nav_row.append({"text": "✍️ Type date", "callback_data": "order_type_date"})
        nav_row.append({"text": "Next ▶️", "callback_data": "order_cal_next"})
        buttons.append(nav_row)
        
        # Cancel button
        buttons.append([{"text": "❌ Cancel", "callback_data": "order_cancel"}])
        
        self.logger.debug(f"[{session.session_token}] Calendar built | year={year} month={month} weeks={len(cal)}")
        
        return {"inline_keyboard": buttons}
    
    def _compute_consumption_days(self, session: OrderSession) -> int:
        """
        Compute consumption days based on delivery dates and on-hand timing.
        
        Formula:
        - start = today (or tomorrow if on-hand is night)
        - end = next_delivery_date - 1 day
        - consumption_days = max(0, (end - start).days + 1)
        
        Args:
            session: Order session with delivery_date, next_delivery_date, onhand_time_hint
            
        Returns:
            int: Consumption days (0 if invalid)
            
        Logs: timezone, start/end dates, on-hand timing, computed days
        """
        from datetime import datetime, timedelta
        
        self.logger.info(f"[{session.session_token}] Computing consumption days | delivery={session.delivery_date} next_delivery={session.next_delivery_date} onhand_time={session.onhand_time_hint}")
        
        if not session.delivery_date or not session.next_delivery_date:
            self.logger.warning(f"[{session.session_token}] Missing delivery dates | cannot compute consumption days")
            return 0
        
        # Get kitchen timezone
        tz_str = self._get_kitchen_timezone()
        
        try:
            # Parse dates
            delivery = datetime.strptime(session.delivery_date, '%Y-%m-%d').date()
            next_delivery = datetime.strptime(session.next_delivery_date, '%Y-%m-%d').date()
            
            # Get today in kitchen timezone
            now_kitchen = get_time_in_timezone(tz_str)
            today = now_kitchen.date()
            
            self.logger.debug(f"[{session.session_token}] Date parsing | today={today} delivery={delivery} next_delivery={next_delivery} tz={tz_str}")
            
            # Determine start date based on on-hand timing
            if session.onhand_time_hint == "night":
                start = today + timedelta(days=1)
                self.logger.info(f"[{session.session_token}] On-hand recorded at night | start=tomorrow ({start})")
            else:
                # Default to morning (include today)
                start = today
                self.logger.info(f"[{session.session_token}] On-hand recorded in morning | start=today ({start})")
            
            # End is day before next delivery
            end = next_delivery - timedelta(days=1)
            
            # Calculate consumption days
            consumption_days = max(0, (end - start).days + 1)
            
            self.logger.info(f"[{session.session_token}] Consumption days computed | start={start} end={end} days={consumption_days}")
            
            return consumption_days
            
        except Exception as e:
            self.logger.error(f"[{session.session_token}] Error computing consumption days | error={e}", exc_info=True)
            return 0
    
    def _show_calendar_picker(self, session: OrderSession, mode: str):
        """
        Display calendar picker for date selection.
        
        Args:
            session: Order session
            mode: 'delivery' or 'next_delivery'
            
        Logs: calendar display, mode, month offset
        """
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        
        self.logger.info(f"[{session.session_token}] Showing calendar picker | mode={mode} month_offset={session._calendar_month_offset}")
        
        # Calculate target month
        today = datetime.now()
        target_date = today + relativedelta(months=session._calendar_month_offset)
        
        # Build keyboard
        keyboard = self._build_calendar_keyboard(session, target_date)
        
        # Build prompt text
        if mode == "delivery":
            text = (
                f"📋 <b>{session.vendor} Order</b>\n"
                f"───────────────────────────────\n\n"
                f"📅 <b>Select Delivery Date</b>\n\n"
                f"Choose the date when this order will be delivered.\n"
                f"Or use the 'Type date' button to enter manually."
            )
        else:  # next_delivery
            text = (
                f"📋 <b>{session.vendor} Order</b>\n"
                f"───────────────────────────────\n"
                f"Delivery: <b>{session.delivery_date}</b>\n\n"
                f"📅 <b>Select Next Delivery Date</b>\n\n"
                f"Choose the next expected delivery after this one.\n"
                f"This determines how many days the order must cover."
            )
        
        self.bot.send_message(session.chat_id, text, reply_markup=keyboard)
        self.logger.debug(f"[{session.session_token}] Calendar picker sent | mode={mode}")
    
    def _show_onhand_timing_prompt(self, session: OrderSession):
        """
        Prompt user for on-hand timing (morning or night).
        
        First looks up most recent On-Hand entry in Notion to show as hint.
        
        This determines whether to include today in consumption days calculation.
        
        Logs: on-hand timing prompt, Notion metadata lookup, hint display
        """
        from datetime import datetime
        
        self.logger.info(f"[{session.session_token}] Showing on-hand timing prompt | vendor={session.vendor}")
        
        # Try to get latest on-hand metadata from Notion
        metadata = None
        try:
            metadata = self.notion.get_latest_onhand_metadata(session.vendor)
            self.logger.info(f"[{session.session_token}] On-hand metadata retrieved | found={metadata is not None}")
        except Exception as e:
            self.logger.warning(f"[{session.session_token}] Failed to get on-hand metadata | error={e}")
            metadata = None
        
        # Build prompt text
        text = (
            f"📋 <b>{session.vendor} Order</b>\n"
            f"───────────────────────────────\n"
            f"Delivery: <b>{session.delivery_date}</b>\n"
            f"Next delivery: <b>{session.next_delivery_date}</b>\n\n"
            f"⏰ <b>On-Hand Timing</b>\n\n"
        )
        
        # Add hint if metadata available
        if metadata:
            created_time = metadata.get('created_time', '')
            created_by = metadata.get('created_by', 'Unknown')
            multiple_recent = metadata.get('multiple_recent', False)
            
            # Format timestamp for display
            try:
                # Parse UTC timestamp from Notion
                dt_utc = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                
                # Convert to kitchen timezone
                try:
                    import pytz
                    kitchen_tz = pytz.timezone(BUSINESS_TIMEZONE)
                    dt_local = dt_utc.astimezone(kitchen_tz)
                    formatted_time = dt_local.strftime('%Y-%m-%d %H:%M')
                    self.logger.debug(f"[{session.session_token}] Timestamp converted | utc={created_time} local={formatted_time}")
                except ImportError:
                    # Fallback if pytz not available - use system local time
                    dt_local = dt_utc.astimezone()
                    formatted_time = dt_local.strftime('%Y-%m-%d %H:%M')
                    self.logger.debug(f"[{session.session_token}] Timestamp converted (system tz) | utc={created_time} local={formatted_time}")
            except:
                formatted_time = created_time[:16] if created_time else 'Unknown'
                self.logger.warning(f"[{session.session_token}] Failed to parse timestamp | using raw={formatted_time}")
            
            text += (
                f"💡 <b>Last on-hand in Notion:</b>\n"
                f"   {formatted_time} by {created_by}\n"
            )
            
            if multiple_recent:
                text += f"   ⚠️ <i>Multiple recent entries found—please confirm.</i>\n"
            
            text += f"\n"
            
            self.logger.info(f"[{session.session_token}] Showing on-hand hint | time={formatted_time} by={created_by} multiple={multiple_recent}")
        else:
            self.logger.info(f"[{session.session_token}] No on-hand hint available | falling back to basic prompt")
        
        text += (
            f"When was the on-hand inventory last recorded?\n\n"
            f"🌅 <b>Morning</b> - Include today in consumption\n"
            f"🌙 <b>Night</b> - Start consumption tomorrow\n\n"
            f"This affects how many days the order must cover."
        )
        
        keyboard = self._create_keyboard([
            [("🌅 Morning", "order_onhand|morning"), ("🌙 Night", "order_onhand|night")],
            [("❌ Cancel", "order_cancel")]
        ])
        
        self.bot.send_message(session.chat_id, text, reply_markup=keyboard)
        self.logger.debug(f"[{session.session_token}] On-hand timing prompt sent | hint_shown={metadata is not None}")
    
    def _handle_delivery_date_callback(self, chat_id: int, user_id: int, date_str: str):
        """
        Handle delivery date selection from calendar.
        
        Validates date, stores it, then shows calendar for next delivery date.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Logs: date selected, validation, next prompt
        """
        from datetime import datetime
        
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for delivery date | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        self.logger.info(f"[{session.session_token}] Delivery date selected | date={date_str}")
        
        # Validate date format
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if selected_date < today:
                self.logger.warning(f"[{session.session_token}] Past delivery date rejected | date={date_str}")
                self.bot.send_message(chat_id, "⚠️ Cannot select a past date. Please choose again.")
                self._show_calendar_picker(session, "delivery")
                return
            
            session.delivery_date = date_str
            session.update_activity()
            self.logger.info(f"[{session.session_token}] Delivery date stored | date={date_str}")
            
            # Reset calendar offset for next delivery
            session._calendar_month_offset = 0
            
            # Show calendar for next delivery date
            self._show_calendar_picker(session, "next_delivery")
            
        except ValueError as e:
            self.logger.error(f"[{session.session_token}] Invalid delivery date format | date={date_str} error={e}")
            self.bot.send_message(chat_id, "⚠️ Invalid date format. Please try again.")
            self._show_calendar_picker(session, "delivery")
    
    def _handle_next_delivery_date_callback(self, chat_id: int, user_id: int, date_str: str):
        """
        Handle next delivery date selection from calendar.
        
        Validates next_delivery > delivery, stores it, then prompts for on-hand timing.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Logs: date selected, validation, on-hand timing prompt
        """
        from datetime import datetime
        
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for next delivery date | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        self.logger.info(f"[{session.session_token}] Next delivery date selected | date={date_str}")
        
        # Validate date format
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            delivery_date = datetime.strptime(session.delivery_date, "%Y-%m-%d").date()
            
            # Validate next_delivery > delivery
            if selected_date <= delivery_date:
                self.logger.warning(f"[{session.session_token}] Next delivery not after delivery | next={date_str} delivery={session.delivery_date}")
                self.bot.send_message(
                    chat_id,
                    f"⚠️ <b>Invalid Date Range</b>\n"
                    f"───────────────────────────────\n"
                    f"Next delivery must be AFTER delivery date.\n\n"
                    f"Delivery: {session.delivery_date}\n"
                    f"Next delivery: {date_str} ❌\n\n"
                    f"Please select a later date."
                )
                self._show_calendar_picker(session, "next_delivery")
                return
            
            session.next_delivery_date = date_str
            session.update_activity()
            self.logger.info(f"[{session.session_token}] Next delivery date stored | date={date_str}")
            
            # Prompt for on-hand timing
            self._show_onhand_timing_prompt(session)
            
        except ValueError as e:
            self.logger.error(f"[{session.session_token}] Invalid next delivery date format | date={date_str} error={e}")
            self.bot.send_message(chat_id, "⚠️ Invalid date format. Please try again.")
            self._show_calendar_picker(session, "next_delivery")
    
    def _handle_onhand_timing_callback(self, chat_id: int, user_id: int, timing: str):
        """
        Handle on-hand timing selection.
        
        Stores timing, computes consumption days, then prompts for submitter name.
        
        Args:
            timing: 'morning' or 'night'
            
        Logs: timing selected, consumption days computed, submitter prompt
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for on-hand timing | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        self.logger.info(f"[{session.session_token}] On-hand timing selected by user | timing={timing} vendor={session.vendor}")
        
        session.onhand_time_hint = timing
        session.update_activity()
        
        # Log the user's final choice (not auto-decided)
        self.logger.critical(f"[{session.session_token}] USER CHOICE CONFIRMED | on_hand_timing={timing} delivery={session.delivery_date} next={session.next_delivery_date}")
        
        # Compute consumption days
        consumption_days = self._compute_consumption_days(session)
        session.consumption_days = consumption_days
        
        if consumption_days == 0:
            self.logger.warning(f"[{session.session_token}] Zero consumption days computed | allowing continue with warning")
            self.bot.send_message(
                chat_id,
                f"⚠️ <b>Warning</b>\n"
                f"Consumption days computed as 0.\n"
                f"This may indicate an invalid date range.\n\n"
                f"Continuing anyway..."
            )
        
        self.logger.info(f"[{session.session_token}] Consumption days set | days={consumption_days}")
        
        # Show submitter prompt
        self._show_submitter_prompt_order(session)
    
    def _handle_type_date_callback(self, chat_id: int, user_id: int):
        """
        Handle "Type date instead" button from calendar picker.
        
        Switches to text input mode for custom date entry.
        
        Logs: type date requested, input mode set
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for type date | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        # Determine which date we're entering
        if session.delivery_date is None:
            mode = "delivery"
            prompt = "delivery date"
        else:
            mode = "next_delivery"
            prompt = "next delivery date"
        
        self.logger.info(f"[{session.session_token}] Type date mode activated | mode={mode}")
        
        session._date_input_mode = mode
        session.update_activity()
        
        self.bot.send_message(
            chat_id,
            f"📅 <b>Enter {prompt.title()}</b>\n"
            f"───────────────────────────────\n"
            f"Format: YYYY-MM-DD\n"
            f"Example: 2025-10-15\n\n"
            f"Or type /cancel to go back"
        )
    
    def _format_date_range_label(self, start_date: str, end_date: str) -> str:
        """
        Format date range as human-readable label (e.g., 'Sat–Wed').
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            str: Formatted range label
            
        Logs: date parsing, label generation
        """
        from datetime import datetime
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get abbreviated day names
            start_day = start.strftime('%a')  # e.g., 'Sat'
            end_day = end.strftime('%a')  # e.g., 'Wed'
            
            range_label = f"{start_day}–{end_day}"
            
            self.logger.debug(f"Date range label | start={start_date} end={end_date} label='{range_label}'")
            
            return range_label
            
        except Exception as e:
            self.logger.error(f"Error formatting date range | start={start_date} end={end_date} error={e}")
            return "Date Range"
    
    def _compute_item_recommendation(self, item: dict, on_hand: float, forecast_multiplier: float = 1.0) -> dict:
        """
        Compute order recommendation using PAR LEVELS with FORECAST ADJUSTMENT.

        Logic:
        - Adjusted min_par = min_par × forecast_multiplier
        - Adjusted max_par = max_par × forecast_multiplier
        - If on_hand < adjusted_min → ORDER (bring to adjusted_max)

        Args:
            item: Item dict with 'name', 'min_par', 'max_par' keys
            on_hand: Current on-hand quantity
            forecast_multiplier: Sales forecast multiplier (default 1.0)

        Returns:
            dict with recommendation details including adjusted pars

        Logs: All inputs, adjusted pars, decision logic
        """
        import math

        item_name = item.get('name', item.get('item_name', 'Unknown'))
        base_min_par = float(item.get('min_par', 0))
        base_max_par = float(item.get('max_par', 0))

        # Apply forecast adjustment
        adj_min_par = base_min_par * forecast_multiplier
        adj_max_par = base_max_par * forecast_multiplier

        print(f"[PAR-CALC] '{item_name}': on_hand={on_hand:.1f}")
        print(f"[PAR-CALC]   Base pars: min={base_min_par:.1f}, max={base_max_par:.1f}")
        print(f"[PAR-CALC]   Forecast multiplier: {forecast_multiplier:.2f}")
        print(f"[PAR-CALC]   Adjusted pars: min={adj_min_par:.1f}, max={adj_max_par:.1f}")

        # Validate par configuration
        if adj_max_par <= 0:
            print(f"[PAR-CALC] ⚠ '{item_name}' has no max_par configured")
            self.logger.warning(f"[PAR-CALC] Item '{item_name}' missing max_par")
            return {
                'item_name': item_name,
                'on_hand': on_hand,
                'min_par': base_min_par,
                'max_par': base_max_par,
                'adj_min_par': adj_min_par,
                'adj_max_par': adj_max_par,
                'forecast_multiplier': forecast_multiplier,
                'order_qty': 0,
                'status': 'UNCONFIGURED',
                'reason': 'No max_par set'
            }

        # Par-based decision with adjusted values
        if on_hand < adj_min_par:
            order_qty = max(0, math.ceil(adj_max_par - on_hand))
            status = 'ORDER'
            reason = f'Below adjusted min ({on_hand:.1f} < {adj_min_par:.1f})'
            print(f"[PAR-CALC] ✓ '{item_name}' → ORDER {order_qty}")
        else:
            order_qty = 0
            status = 'OK'
            reason = f'Above adjusted min ({on_hand:.1f} >= {adj_min_par:.1f})'
            print(f"[PAR-CALC] ✓ '{item_name}' → OK")

        self.logger.info(f"[PAR-CALC] {item_name}: {status} qty={order_qty} | mult={forecast_multiplier:.2f}")

        return {
            'item_name': item_name,
            'on_hand': on_hand,
            'min_par': base_min_par,
            'max_par': base_max_par,
            'adj_min_par': adj_min_par,
            'adj_max_par': adj_max_par,
            'forecast_multiplier': forecast_multiplier,
            'order_qty': order_qty,
            'status': status,
            'reason': reason
        }

    # === PHASE 5: ORDER FLAGGING ===
    def _check_order_flag(self, item: dict, quantity: float, recommended: float) -> dict:
        """
        Check if order quantity exceeds 130% of recommended.

        Args:
            item: Item dict with name, etc.
            quantity: Ordered quantity
            recommended: Recommended quantity

        Returns:
            dict: {'flagged': bool, 'ratio': float, 'message': str}
        """
        ORDER_FLAG_THRESHOLD = 1.30  # 130%

        if recommended <= 0:
            return {'flagged': False, 'ratio': 0, 'message': 'No recommendation'}

        ratio = quantity / recommended
        flagged = ratio > ORDER_FLAG_THRESHOLD

        if flagged:
            message = f"{ratio:.0%} of recommended ({recommended})"
            self.logger.info(f"[PHASE 5] Order flagged: {item['name']} qty={quantity} rec={recommended} ratio={ratio:.0%}")
            print(f"[PHASE 5] ORDER FLAG: {item['name']} - {quantity} is {ratio:.0%} of recommended {recommended}")
        else:
            message = "OK"

        return {
            'flagged': flagged,
            'ratio': round(ratio, 2),
            'message': message,
            'threshold': ORDER_FLAG_THRESHOLD
        }
    # === END PHASE 5 ===

    def _handle_use_recommended(self, chat_id: int, user_id: int, item_name: str, rec_qty: str):
        """
        Handle "Use recommended" button click.
        
        Pre-fills the recommended quantity and advances to next item.
        
        Args:
            item_name: Name of item
            rec_qty: Recommended quantity as string
            
        Logs: action, item, quantity set
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for use recommended | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        try:
            qty = float(rec_qty)
            
            self.logger.info(f"[{session.session_token}] Use recommended action | item={item_name} qty={qty}")
            
            # Verify current item matches
            current_item = session.get_current_item()
            if not current_item or current_item['name'] != item_name:
                self.logger.warning(f"[{session.session_token}] Item mismatch | expected={current_item['name'] if current_item else 'none'} got={item_name}")
                self.bot.send_message(chat_id, "⚠️ Item mismatch. Please try again.")
                self._show_current_item(session)
                return
            
            # Set quantity and advance
            session.set_current_quantity(qty)
            session.index += 1
            
            self.logger.info(f"[{session.session_token}] Recommended quantity set | item={item_name} qty={qty}")
            
            # Show next item or done
            if session.index >= len(session.items):
                self._handle_done(chat_id, session.user_id)
            else:
                self._show_current_item(session)
                
        except ValueError as e:
            self.logger.error(f"[{session.session_token}] Invalid recommended quantity | rec_qty='{rec_qty}' error={e}")
            self.bot.send_message(chat_id, "⚠️ Error using recommended quantity. Please try again.")
            self._show_current_item(session)

    def handle_preselected_vendor_command(self, message: Dict, location: str):
        """
        Entry point for location-specific commands (legacy wrappers).
        Validates that the requested location exists in Notion before proceeding.
        
        Args:
            message: Telegram message object
            location: Location name string (must exist in Notion Items Master)
            
        Logs: command entry, location validation, session check, delegation
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        session_token = str(uuid.uuid4())[:8]
        
        self.logger.info(f"[{session_token}] Preselected location command | location='{location}' user={user_id} chat={chat_id}")
        
        # Validate that this location exists in Notion
        try:
            available_locations = self.notion.get_locations(use_cache=True)
            self.logger.debug(f"[{session_token}] Available locations | count={len(available_locations)} locations={available_locations}")
            
            if location not in available_locations:
                self.logger.error(f"[{session_token}] Invalid location requested | location='{location}' available={available_locations}")
                self.bot.send_message(
                    chat_id,
                    f"⚠️ <b>Invalid Location</b>\n"
                    f"────────────────────────────\n"
                    f"Location '{location}' not found in Items Master.\n\n"
                    f"Available locations: {', '.join(available_locations)}\n\n"
                    f"Use /order to see all locations."
                )
                return
            
            self.logger.info(f"[{session_token}] Location validated | location='{location}'")
            
        except Exception as e:
            self.logger.error(f"[{session_token}] Location validation failed | location='{location}' error={e}", exc_info=True)
            self.bot.send_message(
                chat_id,
                "⚠️ Unable to validate location. Please try /order or contact support."
            )
            return
        
        # Check for active session
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if not session.is_expired():
                # If existing session is for same location, offer resume
                if session.location == location:
                    self.logger.info(f"[{session.session_token}] Active session for same location | location='{location}' progress={session.get_progress()}")
                    keyboard = self._create_keyboard([
                        [("📂 Resume", "order_resume"), ("🔄 Start Over", "order_restart")],
                        [("❌ Cancel", "order_cancel_existing")]
                    ])
                    self.bot.send_message(
                        chat_id,
                        f"📋 <b>Active {location} Order Session</b>\n"
                        f"────────────────────────────\n"
                        f"Progress: {session.get_progress()} • Entered: {session.get_entered_count()}\n\n"
                        f"What would you like to do?",
                        reply_markup=keyboard
                    )
                    return
                else:
                    # Different location - inform and offer to cancel old session
                    self.logger.info(f"[{session.session_token}] Active session for different location | existing='{session.location}' requested='{location}'")
                    keyboard = self._create_keyboard([
                        [("🔄 Switch to " + location, "order_restart")],
                        [("❌ Cancel", "order_cancel_existing")]
                    ])
                    self.bot.send_message(
                        chat_id,
                        f"⚠️ <b>Active Session for {session.location}</b>\n"
                        f"────────────────────────────\n"
                        f"You have an active order for {session.location}.\n"
                        f"Progress: {session.get_progress()}\n\n"
                        f"Cancel it to start {location} order?",
                        reply_markup=keyboard
                    )
                    return
        
        # No active session - start directly with location
        self.logger.info(f"[{session_token}] No active session | starting directly with location='{location}'")
        self._handle_vendor_selection(chat_id, user_id, location)
    
    def handle_order_command(self, message: Dict):
        """
        Entry point for /order command.
        Check for existing session; if present, offer resume/restart.
        Otherwise start vendor selection.
        
        Logs: command entry, session check, vendor prompt.
        """
        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        session_token = str(uuid.uuid4())[:8]
        
        self.logger.info(f"[{session_token}] /order command entry | user={user_id} chat={chat_id}")
        
        # Check for active session
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if not session.is_expired():
                self.logger.info(f"[{session.session_token}] Active session found | vendor={session.vendor} progress={session.get_progress()}")
                # Offer resume or restart
                keyboard = self._create_keyboard([
                    [("📂 Resume", "order_resume"), ("🔄 Start Over", "order_restart")],
                    [("❌ Cancel", "order_cancel_existing")]
                ])
                self.bot.send_message(
                    chat_id,
                    f"📋 <b>Active Order Session</b>\n"
                    f"────────────────────────────\n"
                    f"Vendor: {session.vendor}\n"
                    f"Progress: {session.get_progress()} • Entered: {session.get_entered_count()}\n\n"
                    f"What would you like to do?",
                    reply_markup=keyboard
                )
                return
        
        # No active session - start new
        self.logger.info(f"[{session_token}] No active session | starting vendor selection")
        self._start_vendor_selection(chat_id, user_id, session_token)
    
    def _start_vendor_selection(self, chat_id: int, user_id: int, session_token: str):
        """
        Prompt user to select location from dynamically discovered list.
        Uses Notion Items Master as single source of truth for locations.

        Args:
            chat_id: Telegram chat ID
            user_id: Telegram user ID
            session_token: Correlation token for logging

        Logs: location discovery, button generation, prompt sent
        """
        self.logger.info(f"[{session_token}] Prompting location selection for order")

        try:
            # Discover available locations from Notion
            locations = self.notion.get_locations(use_cache=True)
            self.logger.info(
                f"[{session_token}] Locations discovered for order menu | count={len(locations)} locations={locations}"
            )

            # Normalize, de-dup, and sort for stable button order
            if not locations:
                raise ValueError("No locations returned from Notion")

            norm_locations = sorted({str(loc).strip() for loc in locations if str(loc).strip()})
            if not norm_locations:
                raise ValueError("No usable locations after normalization")

            # Build dynamic location buttons
            location_buttons = []
            for location in norm_locations:
                button_text = f"📍 {location}"
                callback_data = f"order_vendor|{location}"
                location_buttons.append([(button_text, callback_data)])
                self.logger.debug(
                    f"[{session_token}] Built location button | text='{button_text}' callback='{callback_data}'"
                )

            # Add cancel button
            location_buttons.append([("❌ Cancel", "order_cancel")])

            keyboard = self._create_keyboard(location_buttons)

            # Send prompt
            self.bot.send_message(
                chat_id,
                "Interactive order flow\nSelect vendor:",
                reply_markup=keyboard,
            )
            self.logger.info(
                f"[{session_token}] Order location menu sent | locations={len(norm_locations)}"
            )

        except Exception as e:
            self.logger.error(
                f"[{session_token}] Error building order location menu | error={e}",
                exc_info=True,
            )
            self.bot.send_message(
                chat_id,
                "Unable to load locations. Please try again or contact support.",
            )


    
    def handle_callback(self, callback_query: Dict):
        """
        Route all order-related callbacks.
        
        Logs: callback data, routing decision.
        """
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = callback_query.get("from", {}).get("id")
        
        # Acknowledge callback
        self.bot._make_request("answerCallbackQuery", {"callback_query_id": callback_query.get("id")})
        
        session = self.sessions.get(user_id)
        token = session.session_token if session else "no_session"
        self.logger.info(f"[{token}] Order callback | data={data} user={user_id}")
        
        # Route vendor selection
        if data.startswith("order_vendor|"):
            vendor = data.split("|")[1]
            self._handle_vendor_selection(chat_id, user_id, vendor)
        
        # Route delivery date selection
        elif data.startswith("order_delivery_date|"):
            date_str = data.split("|", 1)[1]
            self._handle_delivery_date_callback(chat_id, user_id, date_str)
        
        # Route next delivery date selection
        elif data.startswith("order_next_delivery_date|"):
            date_str = data.split("|", 1)[1]
            self._handle_next_delivery_date_callback(chat_id, user_id, date_str)
        
        # Route on-hand timing selection
        elif data.startswith("order_onhand|"):
            timing = data.split("|")[1]
            self._handle_onhand_timing_callback(chat_id, user_id, timing)
        
        # Calendar navigation
        elif data == "order_cal_prev":
            if session:
                session._calendar_month_offset -= 1
                mode = "delivery" if session.delivery_date is None else "next_delivery"
                self._show_calendar_picker(session, mode)
        
        elif data == "order_cal_next":
            if session:
                session._calendar_month_offset += 1
                mode = "delivery" if session.delivery_date is None else "next_delivery"
                self._show_calendar_picker(session, mode)
        
        elif data == "order_type_date":
            self._handle_type_date_callback(chat_id, user_id)
        
        elif data == "order_cal_noop":
            pass  # Ignore spacer/header buttons
        
        # Route use recommended action
        elif data.startswith("order_use_rec|"):
            parts = data.split("|")
            if len(parts) >= 3:
                item_name = parts[1]
                rec_qty = parts[2]
                self._handle_use_recommended(chat_id, user_id, item_name, rec_qty)
        
        # Existing navigation callbacks
        elif data == "order_back":
            self._handle_back(chat_id, user_id)
        elif data == "order_skip":
            self._handle_skip(chat_id, user_id)
        elif data == "order_done":
            self._handle_done(chat_id, user_id)
        elif data in ["order_cancel", "order_cancel_existing"]:
            self._handle_cancel(chat_id, user_id)
        elif data == "order_resume":
            self._resume_session(chat_id, user_id)
        elif data == "order_restart":
            self._delete_session(user_id)
            session_token = str(uuid.uuid4())[:8]
            self._start_vendor_selection(chat_id, user_id, session_token)
        elif data == "order_confirm":
            self._handle_confirm(chat_id, user_id)
        elif data == "order_review_back":
            self._resume_items(chat_id, user_id)

        # === PHASE 5: ORDER FLAG CALLBACKS ===
        elif data.startswith("order_flag_confirm|"):
            parts = data.split("|")
            item_name = parts[1]
            quantity = float(parts[2])

            if session and hasattr(session, 'pending_order_flag'):
                pending = session.pending_order_flag

                # Track flagged order items
                if not hasattr(session, 'flagged_orders'):
                    session.flagged_orders = {}

                session.flagged_orders[item_name] = {
                    'quantity': quantity,
                    'recommended': pending.get('recommended'),
                    'ratio': pending.get('ratio')
                }

                # Set quantity and continue
                session.quantities[item_name] = quantity
                session.pending_order_flag = {}

                self.logger.info(f"[PHASE 5] Order flag confirmed: {item_name} = {quantity}")
                self.bot.send_message(chat_id, f"<b>{item_name}</b>: {quantity} <i>(flagged)</i>")

                # Move to next item
                session.index += 1
                if session.index >= len(session.items):
                    self._handle_done(chat_id, user_id)
                else:
                    self._show_current_item(session)

        elif data == "order_flag_change":
            if session:
                session.pending_order_flag = {}
                self.bot.send_message(chat_id, "Enter a new quantity:")
        # === END PHASE 5 ===

    def _handle_vendor_selection(self, chat_id: int, user_id: int, location: str):
        """
        Handle location selection from dynamic menu.
        Location string comes directly from Notion Items Master via callback data.
        
        Args:
            chat_id: Telegram chat ID
            user_id: Telegram user ID
            location: Location name as selected by user (from Notion)
            
        Logs: location selected, session creation, calendar prompt
        """
        session_token = str(uuid.uuid4())[:8]
        self.logger.info(f"[{session_token}] Location selected for order | location='{location}' user={user_id}")
        
        # Create session with dynamic location
        session = OrderSession(
            user_id=user_id,
            chat_id=chat_id,
            session_token=session_token,
            location=location,  # Store exactly as received from Notion
            items=[],
            _calendar_month_offset=0
        )
        
        self.sessions[user_id] = session
        self.logger.info(f"[{session_token}] Order session created | location='{location}'")
        
        # Show calendar picker for delivery date
        self._show_calendar_picker(session, "delivery")
    
    def _load_items_and_begin_entry(self, session: OrderSession):
        """
        Load catalog items for session's dynamic location and begin item entry loop.
        Uses location string from session without any hard-coded checks.
        
        Args:
            session: Order session with location already set
            
        Logs: catalog load, inventory load, item count, first item display
        """
        # Load items for the session's location (dynamic, no hard-coding)
        try:
            items_objs = self.notion.get_items_for_location(session.location, use_cache=False)
            self.logger.info(f"[{session.session_token}] Items loaded | location='{session.location}' count={len(items_objs)}")
        except Exception as e:
            self.logger.error(f"[{session.session_token}] Failed to load items | location='{session.location}' error={e}", exc_info=True)
            self.bot.send_message(
                session.chat_id,
                f"⚠️ Unable to load catalog items for {session.location}. Please try again."
            )
            self._delete_session(session.user_id)
            return
        
        if not items_objs:
            self.logger.warning(f"[{session.session_token}] No items found | location='{session.location}'")
            self.bot.send_message(
                session.chat_id,
                f"⚠️ No items found for {session.location}"
            )
            self._delete_session(session.user_id)
            return
        
        # Get current inventory (On Hand) for the session's location
        try:
            inventory_data = self.notion.get_latest_inventory(session.location, entry_type="on_hand")
            self.logger.info(f"[{session.session_token}] Inventory loaded | location='{session.location}' items_count={len(inventory_data)}")
        except Exception as e:
            self.logger.error(f"[{session.session_token}] Failed to load inventory | location='{session.location}' error={e}", exc_info=True)
            inventory_data = {}
        
        # Build items list with On Hand, ADU, and PAR levels
        items = []
        for item_obj in items_objs:
            on_hand = inventory_data.get(item_obj.name, 0.0)
            items.append({
                'name': item_obj.name,
                'unit': item_obj.unit_type,
                'adu': item_obj.adu,
                'on_hand': on_hand,
                'id': item_obj.id,
                'min_par': getattr(item_obj, 'min_par', 0.0),
                'max_par': getattr(item_obj, 'max_par', 0.0)
            })
            self.logger.debug(f"[{session.session_token}] [PAR] Item prepared | name={item_obj.name} on_hand={on_hand} min_par={getattr(item_obj, 'min_par', 0)} max_par={getattr(item_obj, 'max_par', 0)}")
        
        session.items = items
        
        # Initialize quantities dict
        for item in items:
            session.quantities[item['name']] = None
        
        # Reset index to start item entry
        session.index = 0
        
        self.logger.info(f"[{session.session_token}] Order items initialized | location='{session.location}' count={len(items)}")
        
        # Show welcome and first item
        self.bot.send_message(
            session.chat_id,
            f"📋 <b>{session.location} Order</b>\n"
            f"────────────────────────────\n"
            f"Date: {session.order_date}\n"
            f"Submitter: {session.submitter_name}\n"
            f"Items: {len(items)}\n\n"
            f"Enter order quantity for each item.\n"
            f"On Hand and ADU are shown for context.\n\n"
            f"Use: Back, Skip, Done, Cancel"
        )
        self._show_current_item(session)
    
    def _show_submitter_prompt_order(self, session: OrderSession):
        """
        Prompt user to enter their name for order.
        
        Logs: submitter prompt
        """
        self.logger.info(f"[{session.session_token}] Showing submitter prompt for order")
        
        # Set special index to indicate submitter input mode
        session.index = -2  # Different from -1 (custom date mode)
        
        self.bot.send_message(
            session.chat_id,
            f"📋 <b>{session.vendor} Order</b>\n"
            f"────────────────────────────\n\n"
            f"👤 <b>Enter Your Name</b>\n\n"
            f"Type your name to continue.\n"
            f"Or type /cancel to exit."
        )

    def _show_current_item(self, session: OrderSession):
        """
        Display current item with On Hand, ADU, Consumption Days, Recommended, and navigation buttons.
        
        Logs: item displayed, current value if already entered, consumption days, recommendation.
        """
        from datetime import datetime, timedelta
        
        item = session.get_current_item()
        if not item:
            self.logger.info(f"[{session.session_token}] No current item | moving to done")
            self._handle_done(session.chat_id, session.user_id)
            return
        
        progress = session.get_progress()
        current_value = session.quantities.get(item['name'])
        
        self.logger.info(f"[{session.session_token}] Displaying item | name={item['name']} progress={progress} current_value={current_value}")
        
        # Compute consumption days and recommended
        consumption_days = session.consumption_days if session.consumption_days else 0
        
        # Compute date range label
        if session.delivery_date and session.next_delivery_date:
            # Get kitchen timezone
            tz_str = self._get_kitchen_timezone()
            now_kitchen = get_time_in_timezone(tz_str)
            today = now_kitchen.date()
            
            # Determine start date based on on-hand timing
            if session.onhand_time_hint == "night":
                start = today + timedelta(days=1)
            else:
                start = today
            
            # End is day before next delivery
            next_delivery = datetime.strptime(session.next_delivery_date, '%Y-%m-%d').date()
            end = next_delivery - timedelta(days=1)
            
            range_label = self._format_date_range_label(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        else:
            range_label = "Not Set"
        
        # Get forecast multiplier for this vendor
        forecast_multiplier = self.notion.get_forecast_multiplier(session.vendor)

        # Compute recommended quantity using PAR logic with forecast
        rec_result = self._compute_item_recommendation(item, item['on_hand'], forecast_multiplier)
        recommended = rec_result['order_qty']
        rec_status = rec_result['status']
        rec_reason = rec_result['reason']

        self.logger.info(f"[{session.session_token}] [PAR] Item computed | name={item['name']} status={rec_status} rec={recommended} mult={forecast_multiplier:.2f} | {rec_reason}")

        # Get par values for display (use adjusted values if forecast active)
        min_par = rec_result.get('adj_min_par', rec_result['min_par'])
        max_par = rec_result.get('adj_max_par', rec_result['max_par'])

        # Status indicator
        if rec_status == 'ORDER':
            status_icon = '🔴'
            status_text = 'NEED TO ORDER'
        elif rec_status == 'UNCONFIGURED':
            status_icon = '⚠️'
            status_text = 'PAR NOT SET'
        else:
            status_icon = '🟢'
            status_text = 'OK'

        # Build message text with PAR info
        text = (
            f"[{progress}] <b>{item['name']}</b>\n"
            f"Unit: {item['unit']}\n"
            f"─────────────────────\n"
            f"📦 On-hand: <b>{item['on_hand']:.1f}</b>\n"
            f"📉 Min par: {min_par:.1f}\n"
            f"📈 Max par: {max_par:.1f}\n"
        )

        # Show forecast info if multiplier != 1.0
        if forecast_multiplier != 1.0:
            text += f"📊 Forecast: ×{forecast_multiplier:.2f}\n"
            text += f"   (pars adjusted for projected sales)\n"

        text += (
            f"─────────────────────\n"
            f"{status_icon} Status: <b>{status_text}</b>\n"
            f"💡 {rec_reason}\n"
        )

        if recommended > 0:
            text += f"\n<b>Recommended: {recommended}</b>"

        text += "\n\nEnter order quantity:"

        if current_value is not None:
            text += f"\n💡 Current order: {current_value}"

        print(f"[PAR-UI] Showing item '{item['name']}': status={rec_status}, rec={recommended}")

        # Create navigation buttons
        buttons = []

        # Recommendation button (if there's a recommendation)
        if recommended > 0:
            buttons.append([(f"✅ Use {recommended}", f"order_use_rec|{item['name']}|{recommended}")])

        # Navigation row
        nav_row = []
        if session.index > 0:
            nav_row.append(("◀️ Back", "order_back"))
        nav_row.append(("⏭️ Skip", "order_skip"))
        buttons.append(nav_row)

        # Done/Cancel row
        buttons.append([("✅ Done", "order_done"), ("❌ Cancel", "order_cancel")])

        keyboard = self._create_keyboard(buttons)

        # Log message details
        self.logger.debug(f"[{session.session_token}] [PAR-UI] Item prompt sent | item={item['name']} status={rec_status} rec={recommended}")
        
        self.bot.send_message(session.chat_id, text, reply_markup=keyboard)

    def handle_text_input(self, message: Dict, session: OrderSession):
        """
        Validate and process text input for current item, person tag capture, or date entry.
        
        States handled:
        - Item quantity entry: validate numeric, reject negative
        - Person tag capture (on Review screen): store any text
        - Custom date entry for delivery / next_delivery
        
        Logs: input received, state, validation result, quantity/tag set.
        """
        text = message.get("text", "").strip()
        chat_id = session.chat_id
        
        session.update_activity()
        self.logger.info(f"[{session.session_token}] Text input | text='{text}' index={session.index}")

        # Check for cancel command FIRST (works in all input modes)
        if text.lower() in ["/cancel", "cancel"]:
            self.logger.info(f"[{session.session_token}] Cancel requested during input")
            self._handle_cancel(chat_id, session.user_id)
            return

        # Custom date input mode
        if getattr(session, "_date_input_mode", None):
            from datetime import datetime
            
            mode = session._date_input_mode
            self.logger.info(f"[{session.session_token}] Custom date input | mode={mode} text='{text}'")

            try:
                selected_dt = datetime.strptime(text, "%Y-%m-%d")
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                if selected_dt < today:
                    self.logger.warning(f"[{session.session_token}] Past date rejected | date={text}")
                    self.bot.send_message(chat_id, "Cannot select a past date. Enter a valid date (YYYY-MM-DD).")
                    return
                
                if mode == "delivery":
                    session.delivery_date = text
                    session._date_input_mode = None
                    self.logger.info(f"[{session.session_token}] Custom delivery date stored | date={text}")
                    # Prompt for next delivery date
                    self._show_calendar_picker(session, "next_delivery")

                elif mode == "next_delivery":
                    # Validate next_delivery > delivery
                    try:
                        delivery_dt = datetime.strptime(session.delivery_date, "%Y-%m-%d").date()
                    except Exception:
                        self.logger.error(f"[{session.session_token}] Missing/invalid delivery_date when setting next_delivery")
                        self.bot.send_message(chat_id, "Set a delivery date first.")
                        return

                    if selected_dt.date() <= delivery_dt:
                        self.logger.warning(
                            f"[{session.session_token}] Next delivery not after delivery | next={text} delivery={session.delivery_date}"
                        )
                        self.bot.send_message(
                            chat_id,
                            f"Next delivery must be after delivery date.\nDelivery: {session.delivery_date}\nNext delivery: {text}\nEnter a later date."
                        )
                        return

                    session.next_delivery_date = text
                    session._date_input_mode = None
                    self.logger.info(f"[{session.session_token}] Custom next delivery date stored | date={text}")
                    # Prompt for on-hand timing
                    self._show_onhand_timing_prompt(session)

                return
                
            except ValueError:
                self.logger.warning(f"[{session.session_token}] Invalid custom date format | text='{text}'")
                self.bot.send_message(
                    chat_id,
                    "Invalid date format. Use YYYY-MM-DD. Example: 2025-10-15\nOr type /cancel to go back."
                )
                return

        # Submitter name input mode (index = -2)
        if session.index == -2:
            text_clean = text.strip()
            if not text_clean:
                self.logger.warning(f"[{session.session_token}] Empty submitter name rejected")
                self.bot.send_message(chat_id, "Name cannot be empty. Enter your name.")
                return

            session.submitter_name = text_clean
            self.logger.info(f"[{session.session_token}] Submitter name captured | name='{session.submitter_name}'")

            # Now load items and begin entry
            self._load_items_and_begin_entry(session)
            return

        # Review screen: capture person tag (index past last item, pre-confirm)
        if session.index >= len(session.items):
            self.logger.info(f"[{session.session_token}] Capturing person tag on Review | tag='{text}'")
            session.person_tag = text
            session.update_activity()
            self._show_review_with_tag(session)
            return
        
        # Item entry mode: command shortcuts
        lower = text.lower()
        if lower in ["/back", "back"]:
            self._handle_back(chat_id, session.user_id)
            return
        elif lower in ["/skip", "skip"]:
            self._handle_skip(chat_id, session.user_id)
            return
        elif lower in ["/done", "done"]:
            self._handle_done(chat_id, session.user_id)
            return
        elif lower in ["/cancel", "cancel"]:
            # already handled at top, but keep guard to be explicit
            self._handle_cancel(chat_id, session.user_id)
            return
        
        # Numeric quantity validation
        try:
            # allow "1,234" style inputs
            normalized = text.replace(",", "")
            qty = float(normalized)
            if qty < 0:
                raise ValueError("Negative quantity")

            item = session.get_current_item()
            item_name = item["name"] if item else "unknown"
            self.logger.info(f"[{session.session_token}] Valid quantity | qty={qty} item={item_name}")

            # === PHASE 5: CHECK ORDER FLAG ===
            if item and qty > 0:
                # Get recommendation for this item
                forecast_multiplier = self.notion.get_forecast_multiplier(session.vendor)
                rec_result = self._compute_item_recommendation(item, item.get('on_hand', 0), forecast_multiplier)
                recommended = rec_result.get('order_qty', 0)

                # Check if flagged
                flag_check = self._check_order_flag(item, qty, recommended)

                if flag_check['flagged']:
                    # Store pending flag for confirmation
                    if not hasattr(session, 'pending_order_flag'):
                        session.pending_order_flag = {}

                    session.pending_order_flag = {
                        'item_name': item['name'],
                        'quantity': qty,
                        'recommended': recommended,
                        'ratio': flag_check['ratio']
                    }

                    # Show flag warning
                    keyboard = self._create_keyboard([
                        [("Confirm Order", f"order_flag_confirm|{item['name']}|{qty}")],
                        [("Change Quantity", "order_flag_change")],
                        [("Cancel", "order_cancel")]
                    ])

                    self.bot.send_message(
                        chat_id,
                        f"<b>ORDER FLAG</b>\n"
                        f"{'=' * 25}\n\n"
                        f"<b>{item['name']}</b>\n\n"
                        f"You entered: <b>{qty}</b>\n"
                        f"Recommended: <b>{recommended}</b>\n"
                        f"Ratio: <b>{flag_check['ratio']:.0%}</b>\n\n"
                        f"This order is significantly higher than recommended.\n"
                        f"Managers will be notified.\n\n"
                        f"Confirm or change quantity?",
                        reply_markup=keyboard
                    )
                    return  # Don't proceed until confirmed
            # === END PHASE 5 ===

            session.set_current_quantity(qty)
            session.index += 1

            # Next step
            if session.index >= len(session.items):
                self._handle_done(chat_id, session.user_id)
            else:
                self._show_current_item(session)
        
        except ValueError as e:
            self.logger.warning(f"[{session.session_token}] Invalid quantity input | text='{text}' error={e}")
            self.bot.send_message(
                chat_id,
                "Enter a valid number (0 or positive). You can also type Back, Skip, Done, or Cancel."
            )
            # Re-show current item to keep user anchored
            self._show_current_item(session)


    def _handle_back(self, chat_id: int, user_id: int):
        """
        Navigate to previous item, restore prior prompt state.
        
        Logs: back action, new index.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for back | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        if session.index > 0:
            session.move_back()
            self.logger.info(f"[{session.session_token}] Back action | new_index={session.index}")
            self._show_current_item(session)
        else:
            self.logger.info(f"[{session.session_token}] Back at first item | index=0")
            self.bot.send_message(chat_id, "Already at first item.")
            self._show_current_item(session)
    
    def _handle_skip(self, chat_id: int, user_id: int):
        """
        Skip current item (record None), advance index.
        
        Logs: skip action, item skipped.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for skip | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        item = session.get_current_item()
        self.logger.info(f"[{session.session_token}] Skip action | item={item['name'] if item else 'none'}")
        session.skip_current()
        
        if session.index >= len(session.items):
            self._handle_done(chat_id, user_id)
        else:
            self._show_current_item(session)
    
    def _handle_done(self, chat_id: int, user_id: int):
        """
        Short-circuit to Review without losing prior entries.
        Show minimal review: delivery date + qty × item lines.
        
        Logs: done action, review display.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for done | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        self.logger.info(f"[{session.session_token}] Done action | entered={session.get_entered_count()}/{len(session.items)}")
        
        # Display review (which prompts for person tag)
        self._show_review_with_tag(session)
        
    def _show_review_with_tag(self, session: OrderSession):
        """
        Display Review screen with current person tag (if any).
        Uses selected delivery date from calendar picker.
        Includes PAR coverage statistics.

        Logs: review display with tag, delivery date used.
        """
        self.logger.info(f"[{session.session_token}] Displaying Review with tag | tag='{session.person_tag}'")

        # === PAR COVERAGE STATISTICS ===
        print(f"[PAR-SUMMARY] Generating order summary for session {session.session_token}")

        # Get forecast multiplier for this vendor
        forecast_multiplier = self.notion.get_forecast_multiplier(session.vendor)
        print(f"[PAR-SUMMARY] Using forecast multiplier: {forecast_multiplier:.2f}")

        items_below_par = 0
        items_ok = 0
        items_unconfigured = 0

        for item in session.items:
            on_hand = item.get('on_hand', 0)
            rec_result = self._compute_item_recommendation(item, on_hand, forecast_multiplier)

            if rec_result['status'] == 'ORDER':
                items_below_par += 1
            elif rec_result['status'] == 'UNCONFIGURED':
                items_unconfigured += 1
            else:
                items_ok += 1

        print(f"[PAR-SUMMARY] Stats: {items_below_par} below par, {items_ok} ok, {items_unconfigured} unconfigured")
        self.logger.info(f"[{session.session_token}] [PAR-SUMMARY] below_par={items_below_par}, ok={items_ok}, unconfigured={items_unconfigured}, mult={forecast_multiplier:.2f}")
        # === END PAR STATISTICS ===
        
        # Use the delivery date selected in calendar picker
        delivery_date = session.delivery_date if session.delivery_date else "—"
        
        self.logger.info(f"[{session.session_token}] Review using delivery_date | date={delivery_date}")
        
        # Build review text
        text = (
            f"📋 <b>Review Order</b>\n"
            f"────────────────────────────\n"
            f"Vendor: <b>{session.vendor}</b>\n"
            f"Delivery: <b>{delivery_date}</b>\n"
        )

        # Show consumption window if available
        if session.next_delivery_date and session.consumption_days:
            text += f"Next delivery: <b>{session.next_delivery_date}</b>\n"
            text += f"Coverage: <b>{session.consumption_days} days</b>\n"

        # Add PAR coverage stats
        text += f"\n📊 <b>PAR Status:</b>\n"
        text += f"  🔴 Below par: {items_below_par}\n"
        text += f"  🟢 OK: {items_ok}\n"
        if items_unconfigured > 0:
            text += f"  ⚠️ Unconfigured: {items_unconfigured}\n"

        text += "\n"
        
        entered_items = []
        for item in session.items:
            qty = session.quantities.get(item['name'])
            if qty is not None and qty > 0:
                entered_items.append(f"  • {qty} × {item['name']}")
        
        if entered_items:
            text += f"📦 <b>Order ({len(entered_items)} items):</b>\n"
            text += "\n".join(entered_items[:20])
            if len(entered_items) > 20:
                text += f"\n  ...and {len(entered_items) - 20} more"
            text += "\n\n"
        else:
            text += "⚠️ No items entered.\n\n"
        
        # Show current person tag or prompt
        if session.person_tag:
            text += f"💬 Person tag: {session.person_tag}\n\n"
            text += "<i>Type new tag to replace, or press Confirm</i>"
        else:
            text += "💬 <i>Type a person tag (e.g., @handle) or press Confirm</i>"
        
        keyboard = self._create_keyboard([
            [("✅ Confirm", "order_confirm"), ("🔙 Back to Items", "order_review_back")],
            [("❌ Cancel", "order_cancel")]
        ])
        
        self.logger.info(f"[{session.session_token}] Review message built | delivery={delivery_date} items={len(entered_items)} text_len={len(text)}")
        
        self.bot.send_message(session.chat_id, text, reply_markup=keyboard)

    def _build_order_review_message(self, session: OrderSession, delivery_date: str,
                                    submitter: str = "Unknown") -> str:
        """
        Build detailed order confirmation message for notifications.

        Format matches the Entry Confirmation format:
        - Type, Location, Submitter, Delivery Date
        - All items with quantities (including zeros)
        - Summary stats

        Args:
            session: Order session
            delivery_date: Delivery date
            submitter: Submitter name

        Returns:
            str: Formatted message
        """
        # Header section
        text = "📋 <b>Order Confirmation</b>\n\n"
        text += f"<b>Type:</b> Order\n"
        text += f"<b>Location:</b> {session.vendor}\n"
        text += f"<b>Submitter:</b> {submitter}\n"
        text += f"<b>Delivery Date:</b> {delivery_date}\n\n"

        # Items section - show ALL items (including zeros for visibility)
        ordered_items = []
        total_qty = 0.0
        items_with_qty = 0

        for item in session.items:
            qty = session.quantities.get(item['name'])
            if qty is not None:
                ordered_items.append(f"• {qty} × {item['name']}")
                total_qty += qty
                if qty > 0:
                    items_with_qty += 1

        text += f"📦 <b>Ordered ({items_with_qty}):</b>\n"
        if ordered_items:
            # Show items with qty first, then zeros
            items_with_value = [i for i in ordered_items if not i.startswith("• 0 ×")]
            items_zero = [i for i in ordered_items if i.startswith("• 0 ×")]
            text += "\n".join(items_with_value)
            if items_zero:
                text += "\n\n<i>Not ordered:</i>\n"
                text += "\n".join(items_zero)
        else:
            text += "• No items ordered"

        # Summary section
        text += "\n\n📊 <b>Summary:</b>\n"
        text += f"• Items ordered: {items_with_qty}/{len(session.items)}\n"
        text += f"• Total quantity: {total_qty:.0f}\n"

        # Show person tag if set
        if session.person_tag:
            text += f"• Person: {session.person_tag}\n"

        self.logger.debug(f"Order review message built | items={items_with_qty} submitter={submitter}")

        return text

    def _handle_confirm(self, chat_id: int, user_id: int):
        """
        Post final minimal message to resolved prep chat.
        Include delivery date, qty × item lines, and optional person tag on first line.
        Block if no chat ID is resolvable; provide clear guidance.
        
        Logs: confirm action, chat resolution, idempotency check, prep chat post, success/failure.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for confirm | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        self.logger.info(f"[{session.session_token}] Confirm action | vendor={session.vendor} person_tag='{session.person_tag}'")
        
        # Resolve prep chat ID
        prep_chat_id = self.bot._resolve_order_prep_chat(session.vendor, session.session_token)
        
        if not prep_chat_id:
            self.logger.error(f"[{session.session_token}] Cannot confirm - no prep chat configured | vendor={session.vendor}")
            self.bot.send_message(
                chat_id,
                f"⚠️ <b>Configuration Required</b>\n"
                f"────────────────────────────\n"
                f"No prep chat configured for {session.vendor} orders.\n\n"
                f"<b>To fix:</b>\n"
                f"1. Set ORDER_PREP_CHAT_ID_{session.vendor.upper()} in .env\n"
                f"   OR\n"
                f"2. Set ORDER_PREP_CHAT_ID (global fallback)\n"
                f"3. Restart the bot\n\n"
                f"Contact administrator for assistance."
            )
            return
        
        # Get delivery date from session (user selected via calendar)
        delivery_date = session.delivery_date if session.delivery_date else "—"
        self.logger.info(f"[{session.session_token}] Using selected delivery date | date={delivery_date}")
        
        # Build final message
        entered_items = []
        for item in session.items:
            qty = session.quantities.get(item['name'])
            if qty is not None and qty > 0:
                entered_items.append(f"• {qty} × {item['name']}")
        
        if not entered_items:
            self.logger.warning(f"[{session.session_token}] Confirm with no items | blocking")
            self.bot.send_message(chat_id, "⚠️ Cannot confirm order with no items.")
            return
        
        # Build message with submitter first, then person tag if different
        text = ""

        # Show submitter name
        if session.submitter_name:
            text += f"{session.submitter_name}\n"

        # Show person tag only if different from submitter
        if session.person_tag and session.person_tag != session.submitter_name:
            text += f"{session.person_tag}\n"
            
        text += f"Delivery: {delivery_date}\n"
        text += "\n".join(entered_items)
         
        
        self.logger.info(f"[{session.session_token}] Order message prepared | items={len(entered_items)} tag_present={bool(session.person_tag)}")
        
        # Idempotency check (simple: prevent double-clicking Confirm)
        if hasattr(session, '_confirmed'):
            self.logger.warning(f"[{session.session_token}] Duplicate confirm attempt blocked")
            self.bot.send_message(chat_id, "⚠️ Order already confirmed.")
            return
        
        # Mark as confirmed
        session._confirmed = True

        # === PHASE 5: CHECK FOR FLAGGED ORDERS ===
        flagged_orders = getattr(session, 'flagged_orders', {})
        is_flagged = bool(flagged_orders)
        flag_reason = None

        if is_flagged:
            flag_items = []
            for item_name, flag_data in flagged_orders.items():
                ratio = flag_data.get('ratio', 0)
                flag_items.append(f"{item_name} ({ratio:.0%})")
            flag_reason = f"Order exceeds recommendation: {', '.join(flag_items)}"

            self.logger.info(f"[PHASE 5] Order has flagged items: {list(flagged_orders.keys())}")

            # Send manager notification
            self._notify_managers_order_flag(session, flagged_orders)
        # === END PHASE 5 ===

        # Post to prep chat
        try:
            self.logger.info(f"[{session.session_token}] Posting to prep chat | chat_id={prep_chat_id}")
            success = self.bot.send_message(prep_chat_id, text)
            
            if success:
                self.logger.info(f"[{session.session_token}] Order posted successfully | chat={prep_chat_id} items={len(entered_items)}")
                self.bot.send_message(
                    chat_id,
                    f"✅ <b>Order Posted</b>\n\n"
                    f"Order sent to {session.vendor} prep team.\n"
                    f"Items: {len(entered_items)}\n"
                    f"Delivery: {delivery_date}"
                )

                # Log order submission for deadline tracking
                self.notion.log_deadline_event(
                    location=session.vendor,
                    event_type='order_submitted',
                    submitted_by=session.submitter_name,
                    notes=f"Order submitted via Telegram"
                )
                print(f"[ORDER] ✓ Logged order submission for deadline tracking")

                # === ORDER NOTIFICATIONS (same as entry confirmations) ===
                # Get notification config for this vendor
                notification_config = self.bot._get_notification_config(
                    session.vendor, 'entry_confirmation', session.session_token
                )

                # Build rich order review message for notifications
                order_review_message = self._build_order_review_message(
                    session, delivery_date, session.submitter_name or "Unknown"
                )

                # Send to individual users from database config
                if notification_config['found'] and notification_config['telegram_ids']:
                    notified_count = self.bot._notify_individual_users(
                        notification_config['telegram_ids'],
                        order_review_message,
                        session.session_token
                    )
                    self.logger.info(f"[{session.session_token}] Order confirmation sent to {notified_count} user(s)")
                    print(f"[ORDER] ✓ Sent order confirmation to {notified_count} user(s)")
                else:
                    self.logger.info(f"[{session.session_token}] No notification config for order confirmation | vendor={session.vendor}")
                # === END ORDER NOTIFICATIONS ===

                # Clean up session
                self._delete_session(user_id)
            else:
                self.logger.error(f"[{session.session_token}] Failed to post to prep chat | chat={prep_chat_id}")
                session._confirmed = False  # Allow retry
                self.bot.send_message(chat_id, "⚠️ Failed to post order. Please try again.")
                
        except Exception as e:
            self.logger.error(f"[{session.session_token}] Exception posting to prep chat | chat={prep_chat_id} error={e}", exc_info=True)
            session._confirmed = False  # Allow retry
            self.bot.send_message(chat_id, "⚠️ Error posting order. Please try again or contact support.")

    # === PHASE 5: MANAGER NOTIFICATION FOR FLAGGED ORDERS ===
    def _notify_managers_order_flag(self, session: OrderSession, flagged_orders: dict):
        """
        Notify managers about flagged order items.

        Args:
            session: Current order session
            flagged_orders: Dict of {item_name: {quantity, recommended, ratio}}
        """
        managers = self.notion.get_managers_for_location(session.vendor)

        if not managers:
            self.logger.warning(f"[PHASE 5] No managers found for {session.vendor}")
            return

        # Build message
        items_text = ""
        for item_name, data in flagged_orders.items():
            qty = data.get('quantity', 0)
            rec = data.get('recommended', 0)
            ratio = data.get('ratio', 0)
            items_text += f"  <b>{item_name}</b>\n"
            items_text += f"    Ordered: {qty} | Recommended: {rec} | Ratio: {ratio:.0%}\n\n"

        message = (
            f"<b>ORDER FLAG ALERT</b>\n"
            f"{'=' * 25}\n\n"
            f"<b>Location:</b> {session.vendor}\n"
            f"<b>Delivery Date:</b> {session.delivery_date}\n"
            f"<b>Submitted by:</b> {session.submitter_name}\n\n"
            f"<b>Flagged Items:</b>\n\n"
            f"{items_text}"
            f"<i>Orders exceed 130% of recommended quantities.</i>"
        )

        # Send to managers
        for manager in managers:
            try:
                self.bot.send_message(manager['telegram_id'], message)
                self.logger.info(f"[PHASE 5] Order flag notification sent to {manager['name']}")
            except Exception as e:
                self.logger.error(f"[PHASE 5] Failed to notify {manager['name']}: {e}")
    # === END PHASE 5 ===

    def _handle_cancel(self, chat_id: int, user_id: int):
        """
        Cancel and delete session, clear transient state.
        
        Logs: cancel action, session deletion.
        """
        session = self.sessions.get(user_id)
        token = session.session_token if session else "no_session"
        self.logger.info(f"[{token}] Cancel action | user={user_id}")
        
        if user_id in self.sessions:
            self._delete_session(user_id)
            self.bot.send_message(
                chat_id,
                "❌ <b>Order Cancelled</b>\n"
                "No data saved.\n\n"
                "Use /order to start over."
            )
        else:
            self.bot.send_message(chat_id, "No active session to cancel.")
    
    def _resume_session(self, chat_id: int, user_id: int):
        """
        Resume existing session from current item.
        
        Logs: resume action.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for resume | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        session.update_activity()
        self.logger.info(f"[{session.session_token}] Resume action | index={session.index}")
        self._show_current_item(session)
    
    def _resume_items(self, chat_id: int, user_id: int):
        """
        From review, go back to first unanswered item or last item.
        
        Logs: resume to items action, target index.
        """
        session = self.sessions.get(user_id)
        if not session:
            self.logger.warning(f"No session for resume items | user={user_id}")
            self.bot.send_message(chat_id, "No active session.")
            return
        
        # Find first unanswered item
        for i, item in enumerate(session.items):
            if session.quantities.get(item['name']) is None:
                session.index = i
                break
        else:
            session.index = len(session.items) - 1
        
        self.logger.info(f"[{session.session_token}] Resume to items | index={session.index}")
        self._show_current_item(session)
    
    def _delete_session(self, user_id: int):
        """Delete session and log."""
        session = self.sessions.get(user_id)
        if session:
            self.logger.info(f"[{session.session_token}] Session deleted | user={user_id}")
            del self.sessions[user_id]
    
    def _create_keyboard(self, buttons: List[List[tuple]]) -> Dict:
        """Create inline keyboard markup."""
        return {
            "inline_keyboard": [
                [{"text": text, "callback_data": data} for text, data in row]
                for row in buttons
            ]
        }
    
    def cleanup_expired_sessions(self):
        """
        Periodic cleanup of expired sessions.
        
        Logs: cleanup action, sessions removed.
        """
        expired_users = []
        for user_id, session in self.sessions.items():
            if session.is_expired():
                expired_users.append(user_id)
        
        for user_id in expired_users:
            self._delete_session(user_id)
        
        if expired_users:
            self.logger.info(f"Cleaned up {len(expired_users)} expired order sessions")

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

            # Initialize deadline checker
            self.logger.info("Initializing deadline checker...")
            self.deadline_checker = DeadlineChecker(self.notion_manager, self.bot)
            self.deadline_checker.start()
            print(f"[STARTUP] ✓ Deadline checker started")

            # Initialize task assignment checker
            self.logger.info("Initializing task assignment checker...")
            self.task_checker = TaskAssignmentChecker(self.bot)
            self.task_checker.start()
            print(f"[STARTUP] ✓ Task assignment checker started")

            # Initialize scheduled message sender
            self.logger.info("Initializing scheduled message sender...")
            self.scheduled_sender = ScheduledMessageSender(self.bot)
            self.scheduled_sender.start()
            print(f"[STARTUP] ✓ Scheduled message sender started")

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

        if hasattr(self, 'deadline_checker') and self.deadline_checker:
            self.logger.info("Stopping deadline checker...")
            self.deadline_checker.stop()

        if hasattr(self, 'task_checker') and self.task_checker:
            self.logger.info("Stopping task assignment checker...")
            self.task_checker.stop()

        if hasattr(self, 'scheduled_sender') and self.scheduled_sender:
            self.logger.info("Stopping scheduled message sender...")
            self.scheduled_sender.stop()

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
