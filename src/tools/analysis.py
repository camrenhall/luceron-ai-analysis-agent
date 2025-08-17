"""
Document analysis tools - analysis now handled by AWS Step Functions.
"""

import json
import logging
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)