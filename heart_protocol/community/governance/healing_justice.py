"""
Healing Justice System

Restorative and transformative justice approaches that prioritize healing,
accountability, and community wholeness over punishment and exclusion.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RestorativeProcess(Enum):
    """Types of restorative justice processes"""
    HEALING_CIRCLE = "healing_circle"                       # Community healing circles
    RESTORATIVE_CONFERENCE = "restorative_conference"       # Structured restorative meeting
    VICTIM_OFFENDER_DIALOGUE = "victim_offender_dialogue"   # Direct dialogue facilitation
    COMMUNITY_ACCOUNTABILITY = "community_accountability"   # Community-wide accountability
    PEACEMAKING_CIRCLE = "peacemaking_circle"               # Traditional peacemaking
    FAMILY_GROUP_CONFERENCING = "family_group_conferencing" # Family/support network involvement
    NARRATIVE_THERAPY = "narrative_therapy"                 # Story-telling for healing
    TRUTH_AND_RECONCILIATION = "truth_and_reconciliation"   # Truth-telling processes


class TransformativeJustice(Enum):
    """Transformative justice approaches"""
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"             # Addressing systemic causes
    COMMUNITY_TRANSFORMATION = "community_transformation"   # Changing community conditions
    INDIVIDUAL_TRANSFORMATION = "individual_transformation" # Personal healing and growth
    RELATIONSHIP_TRANSFORMATION = "relationship_transformation" # Healing relationships
    STRUCTURAL_CHANGE = "structural_change"                 # Changing harmful structures
    CULTURAL_SHIFT = "cultural_shift"                       # Shifting cultural norms
    PREVENTION_FOCUS = "prevention_focus"                   # Preventing future harm
    COLLECTIVE_HEALING = "collective_healing"               # Community-wide healing


class HarmType(Enum):
    """Types of harm requiring justice response"""
    INTERPERSONAL_CONFLICT = "interpersonal_conflict"       # Conflicts between individuals
    EMOTIONAL_HARM = "emotional_harm"                       # Emotional damage caused
    TRUST_VIOLATION = "trust_violation"                     # Broken trust
    BOUNDARY_VIOLATION = "boundary_violation"               # Crossed boundaries
    COMMUNITY_DISRUPTION = "community_disruption"          # Disruption to community
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"      # Cultural harm
    DISCRIMINATION = "discrimination"                       # Discriminatory behavior
    HARASSMENT = "harassment"                               # Harassment or abuse
    POWER_ABUSE = "power_abuse"                            # Abuse of power/position
    SYSTEMIC_HARM = "systemic_harm"                        # Systemic/structural harm


class JusticeOutcome(Enum):
    """Outcomes of justice processes"""
    HEALING_ACHIEVED = "healing_achieved"                   # Healing and resolution
    ACCOUNTABILITY_ACCEPTED = "accountability_accepted"     # Responsibility taken
    RELATIONSHIP_REPAIRED = "relationship_repaired"        # Relationships restored
    COMMUNITY_STRENGTHENED = "community_strengthened"      # Community improved
    PREVENTION_ESTABLISHED = "prevention_established"      # Prevention measures created
    ONGOING_PROCESS = "ongoing_process"                     # Process continuing
    PARTIAL_RESOLUTION = "partial_resolution"              # Partial success
    PROCESS_INCOMPLETE = "process_incomplete"              # Process not completed
    NO_RESOLUTION = "no_resolution"                        # No resolution achieved


class HealingStage(Enum):
    """Stages of healing in justice processes"""
    SAFETY_ESTABLISHMENT = "safety_establishment"          # Creating safety
    TRUTH_TELLING = "truth_telling"                        # Sharing truth/stories
    ACCOUNTABILITY_TAKING = "accountability_taking"        # Taking responsibility
    HARM_ACKNOWLEDGMENT = "harm_acknowledgment"            # Acknowledging harm
    HEALING_SUPPORT = "healing_support"                    # Providing healing support
    RELATIONSHIP_REPAIR = "relationship_repair"            # Repairing relationships
    PREVENTION_PLANNING = "prevention_planning"            # Planning prevention
    COMMUNITY_INTEGRATION = "community_integration"        # Reintegrating into community


@dataclass
class HealingJusticeCase:
    """Case in the healing justice system"""
    case_id: str
    harm_type: HarmType
    harm_description: str
    people_harmed: List[str]  # Anonymous identifiers
    people_who_caused_harm: List[str]  # Anonymous identifiers
    community_members_affected: List[str]
    cultural_context: List[str]
    trauma_considerations: List[str]
    accessibility_needs: List[str]
    safety_concerns: List[str]
    case_opened: datetime
    assigned_facilitators: List[str]
    restorative_processes_used: List[RestorativeProcess]
    transformative_approaches: List[TransformativeJustice]
    current_healing_stage: HealingStage
    stages_completed: List[HealingStage]
    healing_agreements: List[Dict[str, Any]]
    accountability_measures: List[Dict[str, Any]]
    community_support_provided: List[str]
    case_outcome: Optional[JusticeOutcome]
    healing_assessment: Dict[str, Any]
    lessons_learned: List[str]
    case_closed: Optional[datetime]
    follow_up_scheduled: Optional[datetime]


@dataclass
class HealingAgreement:
    """Agreement reached through healing justice process"""
    agreement_id: str
    case_id: str
    parties_involved: List[str]
    agreement_type: str
    healing_commitments: List[str]
    accountability_actions: List[str]
    community_support_commitments: List[str]
    timeline_for_completion: Dict[str, datetime]
    success_indicators: List[str]
    support_resources_provided: List[str]
    cultural_protocols_included: List[str]
    accessibility_accommodations: List[str]
    agreement_date: datetime
    review_schedule: List[datetime]
    completion_status: Dict[str, str]


@dataclass
class JusticeFacilitator:
    """Facilitator of healing justice processes"""
    facilitator_id: str
    name: str
    cultural_backgrounds: List[str]
    languages_spoken: List[str]
    facilitation_specialties: List[RestorativeProcess]
    trauma_informed_training: bool
    cultural_competency_areas: List[str]
    accessibility_skills: List[str]
    healing_modalities: List[str]
    years_experience: float
    community_standing: str
    availability_patterns: Dict[str, Any]
    facilitation_philosophy: str
    healing_approach: str


class HealingJusticeSystem:
    """
    Comprehensive healing justice system that prioritizes restoration, transformation,
    and community healing over punishment and exclusion.
    
    Core Principles:
    - Healing for all affected by harm
    - Accountability through responsibility-taking, not punishment
    - Community involvement and support
    - Cultural sensitivity and responsiveness
    - Trauma-informed approaches throughout
    - Prevention of future harm
    - Transformation of conditions that enable harm
    - Accessibility and inclusion for all participants
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.community_id = config.get('community_id', 'default')
        
        # Core components
        self.active_cases: Dict[str, HealingJusticeCase] = {}
        self.completed_cases: Dict[str, HealingJusticeCase] = {}
        self.healing_agreements: Dict[str, HealingAgreement] = {}
        self.facilitators: Dict[str, JusticeFacilitator] = {}
        self.case_history: deque = deque(maxlen=10000)
        
        # Process frameworks
        self.restorative_frameworks: Dict[RestorativeProcess, Dict[str, Any]] = {}
        self.transformative_frameworks: Dict[TransformativeJustice, Dict[str, Any]] = {}
        self.cultural_justice_approaches: Dict[str, Any] = {}
        self.healing_modalities: Dict[str, Any] = {}
        
        # Callbacks and integrations
        self.healing_callbacks: List[Callable] = []
        self.community_notifications: List[Callable] = []
        self.safety_monitors: List[Callable] = []
        
        # Initialize system
        self._setup_restorative_frameworks()
        self._setup_transformative_frameworks()
        self._setup_cultural_approaches()
        self._setup_healing_modalities()
        self._setup_safety_protocols()
    
    def _setup_restorative_frameworks(self):
        """Setup frameworks for different restorative processes"""
        self.restorative_frameworks = {
            RestorativeProcess.HEALING_CIRCLE: {
                'description': 'Community circle for collective healing and wisdom-sharing',
                'participants': ['all_affected_parties', 'community_supporters', 'facilitator'],
                'phases': ['opening_ritual', 'story_sharing', 'healing_responses', 'agreement_making', 'closing_ritual'],
                'cultural_adaptations': ['indigenous_protocols', 'spiritual_traditions', 'community_customs'],
                'healing_focus': ['collective_wisdom', 'community_support', 'spiritual_healing'],
                'duration': '2-4_hours',
                'follow_up': 'community_support_check_ins'
            },
            RestorativeProcess.RESTORATIVE_CONFERENCE: {
                'description': 'Structured meeting between affected parties with facilitation',
                'participants': ['harmed_party', 'harm_causer', 'support_people', 'facilitator'],
                'phases': ['preparation', 'truth_telling', 'impact_sharing', 'accountability', 'agreement'],
                'cultural_adaptations': ['communication_styles', 'family_involvement', 'authority_roles'],
                'healing_focus': ['direct_dialogue', 'truth_telling', 'responsibility_taking'],
                'duration': '1-3_hours',
                'follow_up': 'agreement_monitoring'
            },
            RestorativeProcess.TRUTH_AND_RECONCILIATION: {
                'description': 'Truth-telling process for community healing and reconciliation',
                'participants': ['truth_tellers', 'listeners', 'community_witnesses', 'commissioners'],
                'phases': ['truth_gathering', 'public_testimony', 'community_response', 'reconciliation_planning'],
                'cultural_adaptations': ['truth_telling_traditions', 'witness_protocols', 'healing_ceremonies'],
                'healing_focus': ['historical_truth', 'collective_acknowledgment', 'community_reconciliation'],
                'duration': 'weeks_to_months',
                'follow_up': 'reconciliation_implementation'
            }
        }
    
    def _setup_transformative_frameworks(self):
        """Setup frameworks for transformative justice approaches"""
        self.transformative_frameworks = {
            TransformativeJustice.ROOT_CAUSE_ANALYSIS: {
                'description': 'Analysis of systemic and root causes of harm',
                'methods': ['systems_mapping', 'historical_analysis', 'power_analysis', 'cultural_analysis'],
                'outcomes': ['structural_changes', 'policy_changes', 'cultural_shifts', 'resource_redistribution'],
                'participants': ['affected_communities', 'system_experts', 'community_organizers'],
                'timeline': '1-6_months',
                'success_measures': ['harm_reduction', 'system_changes', 'community_empowerment']
            },
            TransformativeJustice.COMMUNITY_TRANSFORMATION: {
                'description': 'Changing community conditions that enable harm',
                'methods': ['community_organizing', 'resource_development', 'culture_change', 'power_redistribution'],
                'outcomes': ['safer_communities', 'stronger_support_networks', 'cultural_healing', 'justice_infrastructure'],
                'participants': ['whole_community', 'community_leaders', 'organizing_groups'],
                'timeline': '6_months_to_years',
                'success_measures': ['community_safety', 'collective_efficacy', 'healing_culture']
            },
            TransformativeJustice.PREVENTION_FOCUS: {
                'description': 'Creating conditions that prevent future harm',
                'methods': ['education_programs', 'skill_building', 'support_networks', 'early_intervention'],
                'outcomes': ['reduced_harm_incidents', 'stronger_communities', 'healing_capacity', 'resilience_building'],
                'participants': ['community_members', 'educators', 'healers', 'organizers'],
                'timeline': 'ongoing',
                'success_measures': ['harm_prevention', 'community_wellness', 'collective_resilience']
            }
        }
    
    def _setup_cultural_approaches(self):
        """Setup culturally responsive justice approaches"""
        self.cultural_justice_approaches = {
            'indigenous_justice': {
                'principles': ['restoration_over_retribution', 'community_healing', 'spiritual_dimensions', 'elder_wisdom'],
                'processes': ['talking_circles', 'peacemaking', 'ceremonial_healing', 'community_banishment_alternatives'],
                'healing_elements': ['smudging', 'prayer', 'storytelling', 'ritual_cleansing'],
                'community_roles': ['elders', 'traditional_healers', 'community_peacekeepers']
            },
            'ubuntu_philosophy': {
                'principles': ['interconnectedness', 'collective_responsibility', 'restoration', 'community_harmony'],
                'processes': ['community_assemblies', 'collective_dialogue', 'ritual_reconciliation', 'community_service'],
                'healing_elements': ['storytelling', 'singing', 'dance', 'collective_meals'],
                'community_roles': ['elders', 'griots', 'traditional_authorities', 'healers']
            },
            'transformative_feminism': {
                'principles': ['power_analysis', 'survivor_centered', 'community_accountability', 'systemic_change'],
                'processes': ['accountability_pods', 'transformative_organizing', 'healing_circles', 'safety_planning'],
                'healing_elements': ['trauma_informed_care', 'body_based_healing', 'creative_expression', 'collective_care'],
                'community_roles': ['survivor_advocates', 'accountability_partners', 'healing_practitioners']
            }
        }
    
    def _setup_healing_modalities(self):
        """Setup various healing modalities available"""
        self.healing_modalities = {
            'trauma_informed_healing': {
                'approaches': ['somatic_therapy', 'emdr', 'trauma_sensitive_yoga', 'breathwork'],
                'principles': ['safety_first', 'choice_and_control', 'cultural_humility', 'collaboration'],
                'accessibility': ['sliding_scale', 'multiple_languages', 'disability_accommodation', 'virtual_options']
            },
            'cultural_healing': {
                'approaches': ['traditional_medicine', 'spiritual_practices', 'cultural_ceremonies', 'ancestral_wisdom'],
                'principles': ['cultural_sovereignty', 'traditional_knowledge_respect', 'community_guidance', 'holistic_wellness'],
                'accessibility': ['cultural_practitioners', 'language_support', 'ceremonial_inclusion', 'family_involvement']
            },
            'creative_healing': {
                'approaches': ['art_therapy', 'music_therapy', 'drama_therapy', 'writing_therapy', 'dance_movement'],
                'principles': ['creative_expression', 'non_verbal_processing', 'community_creation', 'joy_cultivation'],
                'accessibility': ['adaptive_techniques', 'sensory_accommodations', 'communication_support', 'flexible_participation']
            },
            'community_healing': {
                'approaches': ['collective_rituals', 'community_gardens', 'shared_meals', 'storytelling_circles'],
                'principles': ['collective_care', 'mutual_aid', 'shared_responsibility', 'community_resilience'],
                'accessibility': ['inclusive_spaces', 'multiple_participation_ways', 'resource_sharing', 'childcare_provision']
            }
        }
    
    def _setup_safety_protocols(self):
        """Setup safety protocols for healing justice processes"""
        self.safety_protocols = {
            'physical_safety': [
                'safe_meeting_spaces',
                'safety_escorts_available',
                'emergency_contact_systems',
                'safety_planning_with_participants',
                'violence_intervention_protocols'
            ],
            'emotional_safety': [
                'trauma_informed_facilitation',
                'emotional_support_persons',
                'break_and_self_care_options',
                'confidentiality_agreements',
                'healing_resource_access'
            ],
            'cultural_safety': [
                'cultural_competency_requirements',
                'traditional_healer_access',
                'language_interpretation',
                'cultural_protocol_respect',
                'spiritual_practice_accommodation'
            ],
            'psychological_safety': [
                'voluntary_participation',
                'informed_consent_processes',
                'power_dynamic_awareness',
                'trauma_trigger_prevention',
                'mental_health_support_access'
            ]
        }
    
    async def open_healing_justice_case(self, harm_type: HarmType, harm_description: str,
                                      people_harmed: List[str], people_who_caused_harm: List[str],
                                      **case_details) -> str:
        """Open a new healing justice case"""
        try:
            case_id = f"hj_{self.community_id}_{harm_type.value}_{datetime.now().isoformat()}"
            
            # Conduct initial safety assessment
            safety_assessment = await self._assess_safety_needs(
                harm_type, people_harmed, people_who_caused_harm
            )
            
            # Identify cultural considerations
            cultural_context = await self._identify_cultural_context(
                people_harmed + people_who_caused_harm
            )
            
            # Determine appropriate processes
            recommended_processes = await self._recommend_justice_processes(
                harm_type, cultural_context, safety_assessment
            )
            
            # Assign facilitators
            assigned_facilitators = await self._assign_facilitators(
                harm_type, cultural_context, recommended_processes
            )
            
            # Create case record
            case = HealingJusticeCase(
                case_id=case_id,
                harm_type=harm_type,
                harm_description=harm_description,
                people_harmed=people_harmed,
                people_who_caused_harm=people_who_caused_harm,
                community_members_affected=case_details.get('community_affected', []),
                cultural_context=cultural_context,
                trauma_considerations=case_details.get('trauma_considerations', []),
                accessibility_needs=case_details.get('accessibility_needs', []),
                safety_concerns=safety_assessment,
                case_opened=datetime.now(),
                assigned_facilitators=assigned_facilitators,
                restorative_processes_used=[],
                transformative_approaches=[],
                current_healing_stage=HealingStage.SAFETY_ESTABLISHMENT,
                stages_completed=[],
                healing_agreements=[],
                accountability_measures=[],
                community_support_provided=[],
                case_outcome=None,
                healing_assessment={},
                lessons_learned=[],
                case_closed=None,
                follow_up_scheduled=None
            )
            
            # Store case
            self.active_cases[case_id] = case
            
            # Begin healing process
            await self._begin_healing_process(case_id)
            
            # Notify community as appropriate
            await self._notify_community_of_case(case_id)
            
            logger.info(f"Healing justice case opened: {case_id}")
            return case_id
            
        except Exception as e:
            logger.error(f"Error opening healing justice case: {e}")
            return None
    
    async def _assess_safety_needs(self, harm_type: HarmType, 
                                 people_harmed: List[str], 
                                 people_who_caused_harm: List[str]) -> List[str]:
        """Assess safety needs for all participants"""
        safety_concerns = []
        
        # Harm type specific safety considerations
        safety_mapping = {
            HarmType.HARASSMENT: ['ongoing_contact_risk', 'power_imbalance', 'retaliation_risk'],
            HarmType.POWER_ABUSE: ['authority_abuse_risk', 'institutional_protection_needed', 'witness_safety'],
            HarmType.DISCRIMINATION: ['identity_based_safety', 'institutional_discrimination', 'community_backlash'],
            HarmType.EMOTIONAL_HARM: ['psychological_safety', 'support_system_activation', 'healing_space_creation']
        }
        
        safety_concerns.extend(safety_mapping.get(harm_type, ['general_safety_assessment']))
        
        # Check for high-risk indicators
        if len(people_harmed) > 1:
            safety_concerns.append('multiple_victim_coordination')
        if len(people_who_caused_harm) > 1:
            safety_concerns.append('group_harm_dynamics')
        
        return safety_concerns
    
    async def _identify_cultural_context(self, participant_ids: List[str]) -> List[str]:
        """Identify cultural context for the case"""
        # This would integrate with user profile system to understand cultural backgrounds
        # For now, returning placeholder cultural considerations
        return [
            'cultural_competency_required',
            'language_interpretation_may_be_needed',
            'traditional_healing_integration',
            'family_system_involvement'
        ]
    
    async def _recommend_justice_processes(self, harm_type: HarmType, 
                                         cultural_context: List[str],
                                         safety_assessment: List[str]) -> List[RestorativeProcess]:
        """Recommend appropriate justice processes based on case characteristics"""
        recommendations = []
        
        # Harm type specific recommendations
        process_mapping = {
            HarmType.INTERPERSONAL_CONFLICT: [RestorativeProcess.HEALING_CIRCLE, RestorativeProcess.RESTORATIVE_CONFERENCE],
            HarmType.TRUST_VIOLATION: [RestorativeProcess.VICTIM_OFFENDER_DIALOGUE, RestorativeProcess.HEALING_CIRCLE],
            HarmType.COMMUNITY_DISRUPTION: [RestorativeProcess.COMMUNITY_ACCOUNTABILITY, RestorativeProcess.PEACEMAKING_CIRCLE],
            HarmType.CULTURAL_INSENSITIVITY: [RestorativeProcess.HEALING_CIRCLE, RestorativeProcess.TRUTH_AND_RECONCILIATION]
        }
        
        recommendations.extend(process_mapping.get(harm_type, [RestorativeProcess.HEALING_CIRCLE]))
        
        # Cultural context adaptations
        if 'traditional_healing_integration' in cultural_context:
            recommendations.append(RestorativeProcess.PEACEMAKING_CIRCLE)
        
        # Safety considerations
        if 'high_risk_situation' in safety_assessment:
            # Remove direct contact processes if unsafe
            recommendations = [p for p in recommendations if p != RestorativeProcess.VICTIM_OFFENDER_DIALOGUE]
        
        return list(set(recommendations))
    
    async def _assign_facilitators(self, harm_type: HarmType, 
                                 cultural_context: List[str],
                                 processes: List[RestorativeProcess]) -> List[str]:
        """Assign appropriate facilitators for the case"""
        # Match facilitators based on cultural competency, process expertise, and availability
        # For now, returning placeholder assignments
        return ['experienced_facilitator_1', 'cultural_specialist_facilitator']
    
    async def _begin_healing_process(self, case_id: str):
        """Begin the healing process for a case"""
        case = self.active_cases.get(case_id)
        if not case:
            return
        
        # Start with safety establishment
        await self._establish_safety(case_id)
        
        # Schedule initial healing session
        await self._schedule_initial_session(case_id)
    
    async def _establish_safety(self, case_id: str):
        """Establish safety for all participants"""
        case = self.active_cases.get(case_id)
        if not case:
            return
        
        safety_measures = []
        
        # Implement safety protocols based on safety concerns
        for concern in case.safety_concerns:
            if concern == 'ongoing_contact_risk':
                safety_measures.append('no_contact_agreement')
            elif concern == 'power_imbalance':
                safety_measures.append('power_analysis_and_mitigation')
            elif concern == 'psychological_safety':
                safety_measures.append('trauma_informed_support_activation')
        
        # Document safety establishment
        case.stages_completed.append(HealingStage.SAFETY_ESTABLISHMENT)
        case.current_healing_stage = HealingStage.TRUTH_TELLING
        
        logger.info(f"Safety established for case: {case_id}")
    
    async def create_healing_agreement(self, case_id: str, 
                                     agreement_details: Dict[str, Any]) -> str:
        """Create a healing agreement for a case"""
        try:
            agreement_id = f"ha_{case_id}_{datetime.now().isoformat()}"
            
            agreement = HealingAgreement(
                agreement_id=agreement_id,
                case_id=case_id,
                parties_involved=agreement_details.get('parties', []),
                agreement_type=agreement_details.get('type', 'healing_accountability'),
                healing_commitments=agreement_details.get('healing_commitments', []),
                accountability_actions=agreement_details.get('accountability_actions', []),
                community_support_commitments=agreement_details.get('community_support', []),
                timeline_for_completion=agreement_details.get('timeline', {}),
                success_indicators=agreement_details.get('success_indicators', []),
                support_resources_provided=agreement_details.get('resources', []),
                cultural_protocols_included=agreement_details.get('cultural_protocols', []),
                accessibility_accommodations=agreement_details.get('accessibility', []),
                agreement_date=datetime.now(),
                review_schedule=agreement_details.get('review_schedule', []),
                completion_status={}
            )
            
            # Store agreement
            self.healing_agreements[agreement_id] = agreement
            
            # Update case record
            if case_id in self.active_cases:
                self.active_cases[case_id].healing_agreements.append(agreement_details)
            
            logger.info(f"Healing agreement created: {agreement_id}")
            return agreement_id
            
        except Exception as e:
            logger.error(f"Error creating healing agreement: {e}")
            return None
    
    async def get_case_status(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a healing justice case"""
        case = self.active_cases.get(case_id) or self.completed_cases.get(case_id)
        if not case:
            return None
        
        return {
            'case_id': case.case_id,
            'harm_type': case.harm_type.value,
            'current_healing_stage': case.current_healing_stage.value,
            'stages_completed': [stage.value for stage in case.stages_completed],
            'progress_percentage': len(case.stages_completed) / len(HealingStage) * 100,
            'processes_used': [process.value for process in case.restorative_processes_used],
            'assigned_facilitators': case.assigned_facilitators,
            'healing_agreements_count': len(case.healing_agreements),
            'community_support_provided': case.community_support_provided,
            'case_outcome': case.case_outcome.value if case.case_outcome else None,
            'case_status': 'completed' if case.case_closed else 'active',
            'follow_up_scheduled': case.follow_up_scheduled.isoformat() if case.follow_up_scheduled else None
        }
    
    async def generate_healing_justice_report(self, time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive healing justice system report"""
        if time_period:
            cutoff_date = datetime.now() - time_period
            cases = [c for c in self.completed_cases.values() 
                    if c.case_closed and c.case_closed >= cutoff_date]
        else:
            cases = list(self.completed_cases.values())
        
        if not cases:
            return {'message': 'No completed cases found for the specified period'}
        
        # Calculate metrics
        total_cases = len(cases)
        harm_types = defaultdict(int)
        outcomes = defaultdict(int)
        processes_used = defaultdict(int)
        
        for case in cases:
            harm_types[case.harm_type] += 1
            if case.case_outcome:
                outcomes[case.case_outcome] += 1
            for process in case.restorative_processes_used:
                processes_used[process] += 1
        
        healing_success_rate = (
            outcomes[JusticeOutcome.HEALING_ACHIEVED] + 
            outcomes[JusticeOutcome.RELATIONSHIP_REPAIRED]
        ) / total_cases if total_cases > 0 else 0
        
        return {
            'report_period': f"last_{time_period.days}_days" if time_period else "all_time",
            'total_cases': total_cases,
            'active_cases': len(self.active_cases),
            'harm_types_distribution': {k.value: v for k, v in harm_types.items()},
            'outcomes_distribution': {k.value: v for k, v in outcomes.items()},
            'healing_success_rate': healing_success_rate,
            'most_used_processes': {k.value: v for k, v in processes_used.items()},
            'average_case_duration': await self._calculate_average_case_duration(cases),
            'community_healing_impact': await self._assess_community_healing_impact(cases),
            'facilitator_count': len(self.facilitators),
            'healing_agreements_created': len(self.healing_agreements),
            'report_generated': datetime.now().isoformat()
        }
    
    async def _calculate_average_case_duration(self, cases: List[HealingJusticeCase]) -> float:
        """Calculate average case duration in days"""
        durations = []
        for case in cases:
            if case.case_closed:
                duration = (case.case_closed - case.case_opened).days
                durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0
    
    async def _assess_community_healing_impact(self, cases: List[HealingJusticeCase]) -> Dict[str, Any]:
        """Assess overall community healing impact"""
        community_strengthened_count = sum(
            1 for case in cases 
            if case.case_outcome == JusticeOutcome.COMMUNITY_STRENGTHENED
        )
        
        prevention_established_count = sum(
            1 for case in cases 
            if case.case_outcome == JusticeOutcome.PREVENTION_ESTABLISHED
        )
        
        return {
            'community_strengthening_rate': community_strengthened_count / len(cases) if cases else 0,
            'prevention_establishment_rate': prevention_established_count / len(cases) if cases else 0,
            'collective_healing_indicator': (community_strengthened_count + prevention_established_count) / len(cases) if cases else 0
        }
    
    async def register_facilitator(self, facilitator: JusticeFacilitator):
        """Register a new healing justice facilitator"""
        self.facilitators[facilitator.facilitator_id] = facilitator
        logger.info(f"Healing justice facilitator registered: {facilitator.facilitator_id}")
    
    async def add_healing_callback(self, callback: Callable):
        """Add callback for healing process events"""
        self.healing_callbacks.append(callback)