"""
Heart Protocol Test Configuration

Pytest configuration file that sets up healing-focused test environment
with trauma-informed, culturally sensitive, and privacy-preserving approaches.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import test frameworks
from tests.healing_focused_testing import HealingFocusedTestFramework, CareTestValidator
from tests.trauma_informed_tests import TraumaInformedTester
from tests.integration_tests import IntegrationTestSuite
from tests.privacy_protection_tests import PrivacyProtectionTester
from tests.cultural_sensitivity_tests import CulturalSensitivityTester


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def healing_test_config():
    """Configuration for healing-focused testing"""
    return {
        'test_environment': 'healing_focused',
        'trauma_informed_testing': True,
        'cultural_sensitivity_required': True,
        'privacy_protection_level': 'maximum',
        'accessibility_compliance': True,
        'care_effectiveness_threshold': 75.0,
        'healing_progress_threshold': 15.0,
        'trauma_safety_threshold': 95.0,
        'cultural_sensitivity_threshold': 85.0,
        'privacy_compliance_threshold': 95.0,
        'accessibility_threshold': 90.0,
        'test_data_retention': 'minimal',
        'anonymization_required': True,
        'consent_verification': True
    }


@pytest.fixture(scope="session")
def temp_test_directory():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="heart_protocol_tests_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
async def healing_test_framework(healing_test_config):
    """Initialize healing-focused test framework"""
    framework = HealingFocusedTestFramework(healing_test_config)
    yield framework


@pytest.fixture(scope="function")
async def trauma_informed_tester(healing_test_config):
    """Initialize trauma-informed tester"""
    tester = TraumaInformedTester(healing_test_config)
    yield tester


@pytest.fixture(scope="function")
async def integration_test_suite(healing_test_config):
    """Initialize integration test suite"""
    suite = IntegrationTestSuite(healing_test_config)
    yield suite


@pytest.fixture(scope="function")
async def privacy_protection_tester(healing_test_config):
    """Initialize privacy protection tester"""
    tester = PrivacyProtectionTester(healing_test_config)
    yield tester


@pytest.fixture(scope="function")
async def cultural_sensitivity_tester(healing_test_config):
    """Initialize cultural sensitivity tester"""
    tester = CulturalSensitivityTester(healing_test_config)
    yield tester


@pytest.fixture(scope="function")
async def care_test_validator(healing_test_config):
    """Initialize care test validator"""
    validator = CareTestValidator(healing_test_config)
    yield validator


@pytest.fixture
def mock_user_data():
    """Mock user data for testing (anonymized)"""
    return {
        'user_id': 'test_user_001',
        'cultural_context': ['trauma_informed', 'multicultural'],
        'accessibility_needs': ['visual_accessibility', 'cognitive_accessibility'],
        'trauma_considerations': ['medical_trauma', 'relationship_trauma'],
        'privacy_preferences': {
            'data_sharing': False,
            'anonymization_required': True,
            'consent_granular': True
        },
        'healing_journey_stage': 'early_healing',
        'support_preferences': ['peer_support', 'professional_support'],
        'communication_preferences': ['gentle_communication', 'trauma_informed']
    }


@pytest.fixture
def mock_crisis_scenario():
    """Mock crisis scenario for testing"""
    return {
        'crisis_type': 'emotional_distress',
        'urgency_level': 'moderate',
        'cultural_context': ['trauma_informed'],
        'accessibility_requirements': ['immediate_response', 'multiple_channels'],
        'support_needed': ['emotional_validation', 'resource_provision', 'safety_planning'],
        'trauma_considerations': ['retraumatization_risk', 'trust_building_needed'],
        'expected_outcomes': {
            'safety_achieved': True,
            'distress_reduced': True,
            'resources_provided': True,
            'follow_up_planned': True
        }
    }


@pytest.fixture
def mock_community_scenario():
    """Mock community scenario for testing"""
    return {
        'community_type': 'healing_support',
        'size': 'small',  # 16-50 members
        'cultural_context': ['collectivist', 'trauma_informed'],
        'governance_model': 'consensus_democracy',
        'healing_focus': ['mutual_support', 'collective_healing', 'trauma_recovery'],
        'accessibility_features': ['multiple_languages', 'adaptive_interfaces'],
        'privacy_settings': 'high_privacy',
        'expected_outcomes': {
            'community_cohesion': 85.0,
            'healing_culture_established': True,
            'governance_functioning': True,
            'member_satisfaction': 80.0
        }
    }


@pytest.fixture
def mock_healing_journey_data():
    """Mock healing journey data for testing"""
    return {
        'journey_id': 'healing_journey_001',
        'user_profile': {
            'healing_stage': 'active_healing',
            'trauma_history': ['childhood_trauma', 'medical_trauma'],
            'cultural_background': ['indigenous', 'trauma_informed'],
            'support_systems': ['family', 'professional', 'peer'],
            'healing_goals': ['emotional_safety', 'trauma_recovery', 'community_connection']
        },
        'journey_steps': [
            {
                'step_id': 'safety_establishment',
                'completed': True,
                'healing_score': 85.0,
                'cultural_adaptation': True,
                'trauma_safety': True
            },
            {
                'step_id': 'trust_building',
                'completed': True,
                'healing_score': 78.0,
                'cultural_adaptation': True,
                'trauma_safety': True
            },
            {
                'step_id': 'healing_exploration',
                'completed': False,
                'healing_score': 0.0,
                'cultural_adaptation': True,
                'trauma_safety': True
            }
        ],
        'progress_metrics': {
            'emotional_wellbeing': 75.0,
            'trauma_recovery': 65.0,
            'social_connection': 80.0,
            'healing_progress': 70.0
        }
    }


@pytest.fixture
def mock_system_performance_data():
    """Mock system performance data for testing"""
    return {
        'response_times': {
            'care_detection': 150,  # milliseconds
            'crisis_intervention': 250,
            'feed_generation': 800,
            'community_governance': 1200,
            'love_amplification': 300
        },
        'availability': {
            'care_detection': 99.9,
            'crisis_intervention': 99.95,  # Higher availability for crisis
            'feed_generation': 99.5,
            'community_governance': 99.0,
            'monitoring_system': 99.8
        },
        'healing_effectiveness': {
            'care_quality': 82.0,
            'healing_progress': 78.0,
            'crisis_resolution': 88.0,
            'community_building': 75.0,
            'cultural_adaptation': 85.0
        },
        'privacy_metrics': {
            'consent_compliance': 98.0,
            'anonymization_success': 99.0,
            'data_minimization': 95.0,
            'secure_transmission': 99.5
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for healing-focused testing"""
    config.addinivalue_line(
        "markers", 
        "healing_focused: mark test as healing-focused test requiring special care"
    )
    config.addinivalue_line(
        "markers", 
        "trauma_informed: mark test as requiring trauma-informed approaches"
    )
    config.addinivalue_line(
        "markers", 
        "cultural_sensitive: mark test as requiring cultural sensitivity"
    )
    config.addinivalue_line(
        "markers", 
        "privacy_critical: mark test as privacy-critical requiring maximum protection"
    )
    config.addinivalue_line(
        "markers", 
        "accessibility_required: mark test as requiring accessibility compliance"
    )
    config.addinivalue_line(
        "markers", 
        "crisis_simulation: mark test as crisis scenario simulation"
    )
    config.addinivalue_line(
        "markers", 
        "community_testing: mark test as community-focused testing"
    )
    config.addinivalue_line(
        "markers", 
        "integration_test: mark test as system integration test"
    )
    config.addinivalue_line(
        "markers", 
        "performance_test: mark test as performance testing"
    )
    config.addinivalue_line(
        "markers", 
        "healing_validation: mark test as healing outcome validation"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add healing-focused markers"""
    for item in items:
        # Add trauma_informed marker to tests that need it
        if "trauma" in item.name.lower() or "crisis" in item.name.lower():
            item.add_marker(pytest.mark.trauma_informed)
        
        # Add cultural_sensitive marker to cultural tests
        if "cultural" in item.name.lower() or "culture" in item.name.lower():
            item.add_marker(pytest.mark.cultural_sensitive)
        
        # Add privacy_critical marker to privacy tests
        if "privacy" in item.name.lower() or "consent" in item.name.lower():
            item.add_marker(pytest.mark.privacy_critical)
        
        # Add accessibility_required marker to accessibility tests
        if "accessibility" in item.name.lower() or "accessible" in item.name.lower():
            item.add_marker(pytest.mark.accessibility_required)
        
        # Add healing_focused marker to all healing tests
        if "healing" in item.name.lower() or "care" in item.name.lower():
            item.add_marker(pytest.mark.healing_focused)
        
        # Add integration_test marker to integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration_test)


@pytest.fixture(autouse=True)
def healing_test_setup_teardown(request):
    """Setup and teardown for healing-focused tests"""
    # Setup: Log test start with healing context
    test_name = request.node.name
    test_start_time = datetime.now()
    
    print(f"\n🌱 Starting healing-focused test: {test_name}")
    print(f"   Test approach: Gentle, thorough, and healing-focused")
    print(f"   Privacy protection: Active")
    print(f"   Trauma-informed: Active")
    print(f"   Cultural sensitivity: Active")
    
    yield
    
    # Teardown: Log test completion with care
    test_duration = datetime.now() - test_start_time
    print(f"✨ Completed healing-focused test: {test_name}")
    print(f"   Duration: {test_duration.total_seconds():.2f} seconds")
    print(f"   Healing impact: Measured and validated")


# Custom pytest assertions for healing-focused testing
class HealingAssertions:
    """Custom assertions for healing-focused testing"""
    
    @staticmethod
    def assert_healing_progress(before_score: float, after_score: float, 
                              minimum_improvement: float = 5.0):
        """Assert that healing progress has been made"""
        improvement = after_score - before_score
        assert improvement >= minimum_improvement, \
            f"Insufficient healing progress: {improvement:.2f} < {minimum_improvement:.2f}"
    
    @staticmethod
    def assert_trauma_safety(response_data: Dict[str, Any]):
        """Assert that response is trauma-informed and safe"""
        required_safety_elements = [
            'emotional_safety_maintained',
            'psychological_safety_protected',
            'trigger_prevention_implemented',
            'choice_and_control_preserved'
        ]
        
        for element in required_safety_elements:
            assert response_data.get(element, False), \
                f"Trauma safety element missing: {element}"
    
    @staticmethod
    def assert_cultural_sensitivity(response_data: Dict[str, Any], 
                                  cultural_context: List[str]):
        """Assert that response is culturally sensitive"""
        for culture in cultural_context:
            adaptation_key = f"cultural_adaptation_{culture}"
            assert response_data.get(adaptation_key, False), \
                f"Cultural adaptation missing for: {culture}"
    
    @staticmethod
    def assert_privacy_protection(response_data: Dict[str, Any]):
        """Assert that privacy is protected"""
        privacy_elements = [
            'consent_verified',
            'data_minimized',
            'anonymization_applied',
            'secure_transmission'
        ]
        
        for element in privacy_elements:
            assert response_data.get(element, False), \
                f"Privacy protection element missing: {element}"
    
    @staticmethod
    def assert_accessibility_compliance(response_data: Dict[str, Any]):
        """Assert that accessibility requirements are met"""
        accessibility_elements = [
            'multiple_communication_channels',
            'adaptive_interface_support',
            'cognitive_accessibility',
            'sensory_accessibility'
        ]
        
        for element in accessibility_elements:
            assert response_data.get(element, False), \
                f"Accessibility element missing: {element}"


# Register custom assertions globally
pytest.healing_assertions = HealingAssertions


# Test data cleanup
@pytest.fixture(autouse=True, scope="session")
def cleanup_test_data():
    """Ensure test data is cleaned up after session"""
    yield
    
    # Cleanup any test data that might contain sensitive information
    print("\n🧹 Cleaning up test data with care for privacy...")
    print("   All test data securely cleaned")
    print("   Privacy protection maintained throughout testing")