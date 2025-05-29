import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    _instance: Optional['Logger'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        """Initialize logger with file and console handlers"""
        self.logger = logging.getLogger('InvoiceProcessor')
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'invoice_processing_{timestamp}.log')

        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler with simpler formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls):
        """Get singleton logger instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.logger

    @staticmethod
    def set_debug_mode(enabled: bool):
        """Set debug mode for console output"""
        logger = Logger.get_logger()
        console_handler = next(
            (handler for handler in logger.handlers
             if isinstance(handler, logging.StreamHandler) and
             not isinstance(handler, logging.FileHandler)),
            None
        )
        if console_handler:
            console_handler.setLevel(logging.DEBUG if enabled else logging.INFO)
