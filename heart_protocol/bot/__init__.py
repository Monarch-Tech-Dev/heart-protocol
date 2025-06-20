"""
Monarch Bot - Heart Protocol's Caring AI Companion

The Monarch Bot embodies Heart Protocol's values of love, care, and healing.
Like a monarch butterfly's transformation, it helps guide and support users
through their healing journeys with wisdom, compassion, and hope.

Core Philosophy: "Every interaction is an opportunity to serve love."
"""

from .persona import MonarchPersona, PersonalityTrait, CommunicationStyle
from .response_generator import ResponseGenerator, ResponseType, ResponseContext
from .interaction_handler import InteractionHandler, InteractionType
from .caring_intelligence import CaringIntelligence, CaringResponse
from .bot_core import MonarchBot

__all__ = [
    'MonarchBot',
    'MonarchPersona',
    'PersonalityTrait',
    'CommunicationStyle',
    'ResponseGenerator',
    'ResponseType',
    'ResponseContext',
    'InteractionHandler',
    'InteractionType',
    'CaringIntelligence',
    'CaringResponse'
]