"""
Governance Framework

Core framework for healing-focused, democratic community governance that
prioritizes wellbeing, inclusion, and collective wisdom in decision-making.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class GovernanceModel(Enum):
    """Models of community governance"""
    CONSENSUS_DEMOCRACY = "consensus_democracy"                 # Consensus-based decisions
    DELIBERATIVE_DEMOCRACY = "deliberative_democracy"          # Deliberation-focused
    PARTICIPATORY_DEMOCRACY = "participatory_democracy"        # High participation emphasis
    REPRESENTATIVE_DEMOCRACY = "representative_democracy"       # Elected representatives
    COUNCIL_GOVERNANCE = "council_governance"                   # Council-based decisions
    ROTATING_LEADERSHIP = "rotating_leadership"                 # Rotating leadership model
    DISTRIBUTED_GOVERNANCE = "distributed_governance"          # Distributed decision-making
    HYBRID_GOVERNANCE = "hybrid_governance"                     # Mixed approaches


class DecisionType(Enum):
    """Types of decisions requiring governance"""
    POLICY_DECISION = "policy_decision"                         # Community policies
    RESOURCE_ALLOCATION = "resource_allocation"                 # Resource distribution
    CONFLICT_RESOLUTION = "conflict_resolution"                 # Resolving conflicts
    COMMUNITY_STANDARDS = "community_standards"                 # Behavioral standards
    PLATFORM_CHANGES = "platform_changes"                      # Technical changes
    CRISIS_RESPONSE = "crisis_response"                         # Emergency decisions
    MEMBERSHIP_DECISIONS = "membership_decisions"               # Membership changes
    CULTURAL_GUIDELINES = "cultural_guidelines"                 # Cultural norms
    HEALING_PROTOCOLS = "healing_protocols"                     # Healing procedures
    JUSTICE_PROCESSES = "justice_processes"                     # Justice and accountability


class GovernancePhase(Enum):
    """Phases of governance processes"""
    ISSUE_IDENTIFICATION = "issue_identification"               # Identifying issues
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"          # Engaging stakeholders
    INFORMATION_GATHERING = "information_gathering"            # Gathering information
    DELIBERATION = "deliberation"                              # Community discussion
    DECISION_MAKING = "decision_making"                        # Making decisions
    IMPLEMENTATION = "implementation"                           # Implementing decisions
    EVALUATION = "evaluation"                                  # Evaluating outcomes
    REFLECTION = "reflection"                                  # Reflecting and learning


class ParticipationLevel(Enum):
    """Levels of community participation"""
    ACTIVE_LEADERSHIP = "active_leadership"                     # Leading initiatives
    REGULAR_PARTICIPATION = "regular_participation"            # Regular involvement
    OCCASIONAL_ENGAGEMENT = "occasional_engagement"            # Periodic participation
    INFORMED_OBSERVER = "informed_observer"                     # Staying informed
    MINIMAL_ENGAGEMENT = "minimal_engagement"                   # Minimal involvement
    DISENGAGED = "disengaged"                                  # Not participating


@dataclass
class GovernanceDecision:
    """Record of a governance decision"""
    decision_id: str
    decision_type: DecisionType
    title: str
    description: str
    initiated_by: str
    initiated_at: datetime
    stakeholders: List[str]
    decision_process: GovernanceModel
    phases_completed: List[GovernancePhase]
    current_phase: GovernancePhase
    participation_stats: Dict[str, Any]
    cultural_considerations: List[str]
    healing_impact_assessment: Dict[str, Any]
    accessibility_accommodations: List[str]
    decision_outcome: Optional[str]
    implementation_plan: Optional[Dict[str, Any]]
    evaluation_criteria: List[str]
    lessons_learned: List[str]
    completed_at: Optional[datetime]
    effectiveness_score: Optional[float]


@dataclass
class GovernanceMetrics:
    """Metrics for governance effectiveness"""
    period_start: datetime
    period_end: datetime
    total_decisions: int
    decisions_by_type: Dict[DecisionType, int]
    participation_rates: Dict[ParticipationLevel, float]
    average_decision_time: float
    consensus_achievement_rate: float
    stakeholder_satisfaction: float
    implementation_success_rate: float
    healing_impact_positive: float
    cultural_sensitivity_score: float
    accessibility_compliance: float
    democratic_legitimacy_score: float
    community_cohesion_impact: float


@dataclass
class CommunityStakeholder:
    """Community stakeholder in governance"""
    stakeholder_id: str
    stakeholder_type: str
    participation_level: ParticipationLevel
    cultural_background: List[str]
    accessibility_needs: List[str]
    expertise_areas: List[str]
    governance_preferences: Dict[str, Any]
    decision_history: List[str]
    influence_network: List[str]
    healing_priorities: List[str]
    communication_preferences: Dict[str, Any]
    availability_patterns: Dict[str, Any]


class GovernanceFramework:
    """
    Comprehensive framework for healing-focused community governance that
    emphasizes democratic participation, cultural sensitivity, and collective wellbeing.
    
    Core Principles:
    - Democratic participation with accessibility for all
    - Healing-centered approaches to decision-making
    - Cultural sensitivity and adaptation
    - Transparency and accountability
    - Collective wisdom and shared power
    - Restorative rather than punitive justice
    - Continuous learning and adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.community_id = config.get('community_id', 'default')
        self.governance_model = GovernanceModel(config.get('governance_model', 'consensus_democracy'))
        
        # Core components
        self.active_decisions: Dict[str, GovernanceDecision] = {}
        self.completed_decisions: Dict[str, GovernanceDecision] = {}
        self.stakeholders: Dict[str, CommunityStakeholder] = {}
        self.governance_history: deque = deque(maxlen=10000)
        self.cultural_adaptations: Dict[str, Any] = {}
        self.healing_protocols: Dict[str, Any] = {}
        
        # Decision-making engines
        self.decision_engines: Dict[GovernanceModel, Callable] = {}
        self.participation_facilitators: List[Callable] = []
        self.healing_assessors: List[Callable] = []
        self.cultural_adaptors: List[Callable] = []
        
        # Initialize framework
        self._setup_governance_models()
        self._setup_cultural_adaptations()
        self._setup_healing_protocols()
        self._setup_accessibility_support()
    
    def _setup_governance_models(self):
        """Setup different governance models available"""
        self.governance_model_configs = {
            GovernanceModel.CONSENSUS_DEMOCRACY: {
                'decision_threshold': 0.9,  # 90% agreement needed
                'discussion_phases': ['information_gathering', 'deliberation', 'consensus_building'],
                'facilitation_required': True,
                'cultural_sensitivity': 'high',
                'healing_integration': 'high'
            },
            GovernanceModel.DELIBERATIVE_DEMOCRACY: {
                'decision_threshold': 0.75,  # 75% majority
                'discussion_phases': ['stakeholder_engagement', 'information_gathering', 'deliberation', 'voting'],
                'facilitation_required': True,
                'cultural_sensitivity': 'high',
                'healing_integration': 'medium'
            },
            GovernanceModel.PARTICIPATORY_DEMOCRACY: {
                'decision_threshold': 0.6,  # 60% majority
                'discussion_phases': ['broad_engagement', 'collaborative_design', 'community_vote'],
                'facilitation_required': False,
                'cultural_sensitivity': 'medium',
                'healing_integration': 'medium'
            },
            GovernanceModel.COUNCIL_GOVERNANCE: {
                'decision_threshold': 0.67,  # Council supermajority
                'discussion_phases': ['council_deliberation', 'community_input', 'council_decision'],
                'facilitation_required': True,
                'cultural_sensitivity': 'high',
                'healing_integration': 'high'
            }
        }
    
    def _setup_cultural_adaptations(self):
        """Setup cultural adaptations for governance"""
        self.cultural_adaptations = {
            'collectivist_cultures': {
                'decision_approach': 'harmony_seeking',
                'conflict_style': 'indirect_mediation',
                'leadership_model': 'elder_guidance',
                'participation_style': 'group_consensus',
                'communication_norms': 'respectful_deference'
            },
            'individualist_cultures': {
                'decision_approach': 'rights_based',
                'conflict_style': 'direct_confrontation',
                'leadership_model': 'elected_representation',
                'participation_style': 'individual_voice',
                'communication_norms': 'open_debate'
            },
            'hierarchical_cultures': {
                'decision_approach': 'authority_respect',
                'conflict_style': 'structured_process',
                'leadership_model': 'respected_authority',
                'participation_style': 'role_based',
                'communication_norms': 'formal_protocols'
            },
            'egalitarian_cultures': {
                'decision_approach': 'equal_participation',
                'conflict_style': 'peer_mediation',
                'leadership_model': 'rotating_leadership',
                'participation_style': 'equal_voice',
                'communication_norms': 'informal_dialogue'
            }
        }
    
    def _setup_healing_protocols(self):
        """Setup healing-centered governance protocols"""
        self.healing_protocols = {
            'decision_healing_assessment': {
                'healing_impact_questions': [
                    'How will this decision affect community wellbeing?',
                    'What healing opportunities does this create?',
                    'Who might be harmed by this decision?',
                    'How can we minimize harm and maximize healing?',
                    'What support do affected members need?'
                ],
                'healing_criteria': [
                    'psychological_safety_enhancement',
                    'trauma_sensitivity',
                    'cultural_healing_support',
                    'relationship_strengthening',
                    'collective_resilience_building'
                ]
            },
            'conflict_transformation': {
                'healing_approaches': [
                    'restorative_circles',
                    'truth_and_reconciliation',
                    'healing_dialogues',
                    'community_healing_rituals',
                    'transformative_mediation'
                ],
                'healing_outcomes': [
                    'relationship_repair',
                    'understanding_deepening',
                    'trust_rebuilding',
                    'community_strengthening',
                    'individual_healing'
                ]
            },
            'decision_implementation_healing': {
                'healing_support_measures': [
                    'transition_support_for_affected_members',
                    'healing_spaces_for_processing_change',
                    'cultural_sensitivity_in_implementation',
                    'trauma_informed_change_management',
                    'community_care_during_transitions'
                ]
            }
        }
    
    def _setup_accessibility_support(self):
        """Setup accessibility support for governance participation"""
        self.accessibility_support = {
            'communication_accessibility': [
                'multiple_language_support',
                'sign_language_interpretation',
                'audio_description_services',
                'easy_read_versions',
                'visual_communication_aids'
            ],
            'participation_accessibility': [
                'flexible_meeting_times',
                'multiple_participation_channels',
                'async_participation_options',
                'mobility_accommodation',
                'neurodiversity_support'
            ],
            'decision_accessibility': [
                'plain_language_explanations',
                'visual_decision_summaries',
                'multiple_feedback_methods',
                'extended_consideration_time',
                'supported_decision_making'
            ]
        }
    
    async def initiate_governance_decision(self, decision_type: DecisionType,
                                         title: str, description: str,
                                         initiated_by: str,
                                         stakeholders: Optional[List[str]] = None) -> str:
        """Initiate a new governance decision process"""
        try:
            decision_id = f"gov_{self.community_id}_{decision_type.value}_{datetime.now().isoformat()}"
            
            # Identify stakeholders if not provided
            if not stakeholders:
                stakeholders = await self._identify_stakeholders(decision_type, description)
            
            # Assess cultural considerations
            cultural_considerations = await self._assess_cultural_considerations(stakeholders)
            
            # Conduct healing impact assessment
            healing_impact = await self._assess_healing_impact(decision_type, description)
            
            # Determine accessibility accommodations needed
            accessibility_accommodations = await self._determine_accessibility_needs(stakeholders)
            
            # Create decision record
            decision = GovernanceDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                title=title,
                description=description,
                initiated_by=initiated_by,
                initiated_at=datetime.now(),
                stakeholders=stakeholders,
                decision_process=self.governance_model,
                phases_completed=[],
                current_phase=GovernancePhase.ISSUE_IDENTIFICATION,
                participation_stats={},
                cultural_considerations=cultural_considerations,
                healing_impact_assessment=healing_impact,
                accessibility_accommodations=accessibility_accommodations,
                decision_outcome=None,
                implementation_plan=None,
                evaluation_criteria=await self._define_evaluation_criteria(decision_type),
                lessons_learned=[],
                completed_at=None,
                effectiveness_score=None
            )
            
            # Store decision
            self.active_decisions[decision_id] = decision
            
            # Begin governance process
            await self._advance_governance_phase(decision_id)
            
            logger.info(f"Governance decision initiated: {decision_id}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Error initiating governance decision: {e}")
            return None
    
    async def _identify_stakeholders(self, decision_type: DecisionType, 
                                   description: str) -> List[str]:
        """Identify relevant stakeholders for a decision"""
        stakeholder_mapping = {
            DecisionType.POLICY_DECISION: ['all_members', 'community_leaders', 'affected_groups'],
            DecisionType.RESOURCE_ALLOCATION: ['resource_contributors', 'resource_recipients', 'community_treasurer'],
            DecisionType.CONFLICT_RESOLUTION: ['conflict_parties', 'mediators', 'community_healers'],
            DecisionType.COMMUNITY_STANDARDS: ['all_members', 'cultural_representatives', 'accessibility_advocates'],
            DecisionType.CRISIS_RESPONSE: ['crisis_responders', 'affected_individuals', 'community_leaders'],
            DecisionType.HEALING_PROTOCOLS: ['healing_practitioners', 'trauma_survivors', 'cultural_healers']
        }
        
        base_stakeholders = stakeholder_mapping.get(decision_type, ['all_members'])
        
        # Add stakeholders based on content analysis
        content_stakeholders = await self._analyze_content_for_stakeholders(description)
        
        return list(set(base_stakeholders + content_stakeholders))
    
    async def _assess_cultural_considerations(self, stakeholders: List[str]) -> List[str]:
        """Assess cultural considerations for the decision process"""
        considerations = []
        
        # Analyze stakeholder cultural backgrounds
        cultural_groups = set()
        for stakeholder_id in stakeholders:
            stakeholder = self.stakeholders.get(stakeholder_id)
            if stakeholder:
                cultural_groups.update(stakeholder.cultural_background)
        
        # Generate cultural considerations
        for culture in cultural_groups:
            if culture in self.cultural_adaptations:
                adaptation = self.cultural_adaptations[culture]
                considerations.extend([
                    f"respect_{culture}_{adaptation['decision_approach']}",
                    f"accommodate_{culture}_{adaptation['communication_norms']}",
                    f"include_{culture}_{adaptation['participation_style']}"
                ])
        
        return list(set(considerations))
    
    async def _assess_healing_impact(self, decision_type: DecisionType, 
                                   description: str) -> Dict[str, Any]:
        """Assess the healing impact of the proposed decision"""
        healing_assessment = {
            'potential_healing_benefits': [],
            'potential_healing_risks': [],
            'healing_support_needed': [],
            'healing_measurement_criteria': []
        }
        
        # Decision type specific healing considerations
        healing_considerations = {
            DecisionType.CONFLICT_RESOLUTION: {
                'benefits': ['relationship_repair', 'trust_building', 'community_healing'],
                'risks': ['re_traumatization', 'exclusion', 'power_imbalance'],
                'support': ['healing_circles', 'trauma_informed_mediation', 'aftercare_support']
            },
            DecisionType.COMMUNITY_STANDARDS: {
                'benefits': ['safety_enhancement', 'inclusion_promotion', 'clarity_provision'],
                'risks': ['exclusion_risk', 'cultural_insensitivity', 'enforcement_trauma'],
                'support': ['education_support', 'cultural_adaptation', 'gentle_accountability']
            },
            DecisionType.HEALING_PROTOCOLS: {
                'benefits': ['healing_access', 'trauma_recovery', 'collective_resilience'],
                'risks': ['inappropriate_interventions', 'cultural_mismatch', 'retraumatization'],
                'support': ['cultural_healing_integration', 'trauma_expertise', 'ongoing_evaluation']
            }
        }
        
        if decision_type in healing_considerations:
            considerations = healing_considerations[decision_type]
            healing_assessment['potential_healing_benefits'] = considerations['benefits']
            healing_assessment['potential_healing_risks'] = considerations['risks']
            healing_assessment['healing_support_needed'] = considerations['support']
        
        return healing_assessment
    
    async def _determine_accessibility_needs(self, stakeholders: List[str]) -> List[str]:
        """Determine accessibility accommodations needed for stakeholders"""
        accommodations = set()
        
        for stakeholder_id in stakeholders:
            stakeholder = self.stakeholders.get(stakeholder_id)
            if stakeholder:
                accommodations.update(stakeholder.accessibility_needs)
        
        # Add standard accessibility accommodations
        accommodations.update([
            'multiple_communication_channels',
            'flexible_participation_timing',
            'plain_language_summaries',
            'visual_and_audio_options'
        ])
        
        return list(accommodations)
    
    async def _define_evaluation_criteria(self, decision_type: DecisionType) -> List[str]:
        """Define criteria for evaluating decision effectiveness"""
        base_criteria = [
            'stakeholder_satisfaction',
            'implementation_success',
            'healing_impact_positive',
            'cultural_sensitivity_maintained',
            'accessibility_achieved'
        ]
        
        type_specific_criteria = {
            DecisionType.CONFLICT_RESOLUTION: [
                'relationship_repair_achieved',
                'trust_rebuilding_progress',
                'recurrence_prevention'
            ],
            DecisionType.RESOURCE_ALLOCATION: [
                'equitable_distribution_achieved',
                'resource_efficiency',
                'community_benefit_maximized'
            ],
            DecisionType.HEALING_PROTOCOLS: [
                'healing_outcomes_improved',
                'trauma_reduction_achieved',
                'cultural_healing_integration'
            ]
        }
        
        return base_criteria + type_specific_criteria.get(decision_type, [])
    
    async def _advance_governance_phase(self, decision_id: str):
        """Advance the governance process to the next phase"""
        decision = self.active_decisions.get(decision_id)
        if not decision:
            return
        
        current_phase = decision.current_phase
        
        # Define phase progression
        phase_progression = {
            GovernancePhase.ISSUE_IDENTIFICATION: GovernancePhase.STAKEHOLDER_ENGAGEMENT,
            GovernancePhase.STAKEHOLDER_ENGAGEMENT: GovernancePhase.INFORMATION_GATHERING,
            GovernancePhase.INFORMATION_GATHERING: GovernancePhase.DELIBERATION,
            GovernancePhase.DELIBERATION: GovernancePhase.DECISION_MAKING,
            GovernancePhase.DECISION_MAKING: GovernancePhase.IMPLEMENTATION,
            GovernancePhase.IMPLEMENTATION: GovernancePhase.EVALUATION,
            GovernancePhase.EVALUATION: GovernancePhase.REFLECTION
        }
        
        if current_phase in phase_progression:
            # Mark current phase as completed
            decision.phases_completed.append(current_phase)
            
            # Advance to next phase
            decision.current_phase = phase_progression[current_phase]
            
            # Trigger phase-specific actions
            await self._execute_phase_actions(decision_id, decision.current_phase)
    
    async def _execute_phase_actions(self, decision_id: str, phase: GovernancePhase):
        """Execute actions specific to the current governance phase"""
        phase_actions = {
            GovernancePhase.STAKEHOLDER_ENGAGEMENT: self._engage_stakeholders,
            GovernancePhase.INFORMATION_GATHERING: self._gather_information,
            GovernancePhase.DELIBERATION: self._facilitate_deliberation,
            GovernancePhase.DECISION_MAKING: self._facilitate_decision_making,
            GovernancePhase.IMPLEMENTATION: self._support_implementation,
            GovernancePhase.EVALUATION: self._evaluate_outcomes,
            GovernancePhase.REFLECTION: self._facilitate_reflection
        }
        
        if phase in phase_actions:
            await phase_actions[phase](decision_id)
    
    async def _engage_stakeholders(self, decision_id: str):
        """Engage stakeholders in the governance process"""
        decision = self.active_decisions.get(decision_id)
        if not decision:
            return
        
        engagement_activities = []
        
        # Create culturally appropriate engagement strategies
        for stakeholder_id in decision.stakeholders:
            stakeholder = self.stakeholders.get(stakeholder_id)
            if stakeholder:
                # Customize engagement based on cultural background
                engagement_strategy = await self._create_cultural_engagement_strategy(
                    stakeholder, decision
                )
                engagement_activities.append(engagement_strategy)
        
        # Track engagement activities
        decision.participation_stats['engagement_activities'] = engagement_activities
        
        logger.info(f"Stakeholder engagement initiated for decision: {decision_id}")
    
    async def _create_cultural_engagement_strategy(self, stakeholder: CommunityStakeholder,
                                                 decision: GovernanceDecision) -> Dict[str, Any]:
        """Create culturally appropriate engagement strategy for stakeholder"""
        cultural_backgrounds = stakeholder.cultural_background
        
        strategy = {
            'stakeholder_id': stakeholder.stakeholder_id,
            'communication_methods': [],
            'participation_approaches': [],
            'cultural_adaptations': [],
            'accessibility_accommodations': stakeholder.accessibility_needs
        }
        
        # Apply cultural adaptations
        for culture in cultural_backgrounds:
            if culture in self.cultural_adaptations:
                adaptation = self.cultural_adaptations[culture]
                strategy['communication_methods'].append(adaptation['communication_norms'])
                strategy['participation_approaches'].append(adaptation['participation_style'])
                strategy['cultural_adaptations'].append(f"respect_{culture}_values")
        
        return strategy
    
    async def get_governance_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a governance decision"""
        decision = self.active_decisions.get(decision_id) or self.completed_decisions.get(decision_id)
        if not decision:
            return None
        
        return {
            'decision_id': decision.decision_id,
            'title': decision.title,
            'decision_type': decision.decision_type.value,
            'current_phase': decision.current_phase.value,
            'phases_completed': [p.value for p in decision.phases_completed],
            'progress_percentage': len(decision.phases_completed) / 8 * 100,  # 8 total phases
            'stakeholders_count': len(decision.stakeholders),
            'cultural_considerations': decision.cultural_considerations,
            'healing_impact_assessment': decision.healing_impact_assessment,
            'accessibility_accommodations': decision.accessibility_accommodations,
            'decision_outcome': decision.decision_outcome,
            'completed': decision.completed_at is not None,
            'effectiveness_score': decision.effectiveness_score
        }
    
    async def generate_governance_report(self, time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive governance effectiveness report"""
        if time_period:
            cutoff_date = datetime.now() - time_period
            decisions = [d for d in self.completed_decisions.values() 
                        if d.completed_at and d.completed_at >= cutoff_date]
        else:
            decisions = list(self.completed_decisions.values())
        
        if not decisions:
            return {'message': 'No completed decisions found for the specified period'}
        
        # Calculate metrics
        total_decisions = len(decisions)
        decisions_by_type = defaultdict(int)
        effectiveness_scores = []
        healing_impacts = []
        
        for decision in decisions:
            decisions_by_type[decision.decision_type] += 1
            if decision.effectiveness_score:
                effectiveness_scores.append(decision.effectiveness_score)
            if decision.healing_impact_assessment.get('healing_impact_score'):
                healing_impacts.append(decision.healing_impact_assessment['healing_impact_score'])
        
        return {
            'report_period': f"last_{time_period.days}_days" if time_period else "all_time",
            'total_decisions': total_decisions,
            'decisions_by_type': {k.value: v for k, v in decisions_by_type.items()},
            'average_effectiveness_score': sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0,
            'average_healing_impact': sum(healing_impacts) / len(healing_impacts) if healing_impacts else 0,
            'governance_model': self.governance_model.value,
            'stakeholder_count': len(self.stakeholders),
            'cultural_adaptations_active': len(self.cultural_adaptations),
            'accessibility_accommodations_provided': sum(len(d.accessibility_accommodations) for d in decisions),
            'report_generated': datetime.now().isoformat()
        }
    
    async def register_stakeholder(self, stakeholder: CommunityStakeholder):
        """Register a new community stakeholder"""
        self.stakeholders[stakeholder.stakeholder_id] = stakeholder
        logger.info(f"Stakeholder registered: {stakeholder.stakeholder_id}")
    
    async def update_governance_model(self, new_model: GovernanceModel):
        """Update the community's governance model"""
        old_model = self.governance_model
        self.governance_model = new_model
        
        # Log the change
        logger.info(f"Governance model updated from {old_model.value} to {new_model.value}")
        
        # This could trigger a governance decision about the change itself
        await self.initiate_governance_decision(
            DecisionType.PLATFORM_CHANGES,
            f"Governance Model Change to {new_model.value}",
            f"Community governance model changed from {old_model.value} to {new_model.value}",
            "system"
        )