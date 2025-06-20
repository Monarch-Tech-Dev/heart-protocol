"""
Knowledge Synthesis for Community Wisdom Feed

Synthesizes validated wisdom insights into comprehensive knowledge collections,
creating coherent wisdom guides that serve collective healing and growth.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging

from .wisdom_curation import WisdomInsight, WisdomCategory, WisdomType
from .insight_validation import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class SynthesisMethod(Enum):
    """Methods for synthesizing knowledge"""
    THEMATIC_CLUSTERING = "thematic_clustering"         # Group by themes/topics
    EXPERIENTIAL_JOURNEY = "experiential_journey"       # Organize by healing journey stages
    PRACTICAL_PATHWAY = "practical_pathway"             # Step-by-step practical guides
    WISDOM_LINEAGE = "wisdom_lineage"                   # Connect related insights over time
    CRISIS_RESOURCE_COMPILATION = "crisis_resource_compilation"  # Emergency wisdom collection
    CULTURAL_SYNTHESIS = "cultural_synthesis"           # Organize by cultural contexts
    TRAUMA_INFORMED_GUIDE = "trauma_informed_guide"     # Trauma-specific wisdom organization


class SynthesisScope(Enum):
    """Scope of knowledge synthesis"""
    INDIVIDUAL_TOPIC = "individual_topic"               # Single specific topic
    CATEGORY_COMPREHENSIVE = "category_comprehensive"   # Entire wisdom category
    CROSS_CATEGORY = "cross_category"                   # Multiple related categories
    JOURNEY_STAGE = "journey_stage"                     # Specific healing stage
    CRISIS_SUPPORT = "crisis_support"                   # Crisis intervention wisdom
    DAILY_PRACTICE = "daily_practice"                   # Daily healing practices
    COMMUNITY_COLLECTION = "community_collection"       # Community-generated themes


@dataclass
class SynthesizedWisdom:
    """Represents a synthesized wisdom collection"""
    synthesis_id: str
    title: str
    description: str
    category: WisdomCategory
    synthesis_method: SynthesisMethod
    scope: SynthesisScope
    source_insights: List[str]  # insight_ids
    synthesized_content: str
    key_themes: List[str]
    practical_applications: List[str]
    cultural_considerations: List[str]
    trauma_informed_notes: List[str]
    safety_guidelines: List[str]
    target_audience: List[str]
    prerequisites: List[str]
    related_synthesis: List[str]  # other synthesis_ids
    evidence_strength: float  # 0.0 to 1.0
    community_validation: float  # 0.0 to 1.0
    created_at: datetime
    last_updated: datetime
    usage_metrics: Dict[str, int]
    effectiveness_data: Optional[Dict[str, Any]]


class KnowledgeSynthesizer:
    """
    Synthesizes validated wisdom insights into comprehensive knowledge guides.
    
    Core Principles:
    - Honor the wisdom of lived experience
    - Create coherent, usable knowledge from diverse insights
    - Maintain trauma-informed approaches throughout
    - Respect cultural diversity and context
    - Prioritize safety and harm reduction
    - Foster collective learning and growth
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Synthesis algorithms and templates
        self.synthesis_templates = self._initialize_synthesis_templates()
        self.clustering_algorithms = self._initialize_clustering_algorithms()
        self.narrative_frameworks = self._initialize_narrative_frameworks()
        self.safety_guidelines = self._initialize_safety_guidelines()
        
        # Synthesized knowledge storage
        self.synthesized_wisdom = {}      # synthesis_id -> SynthesizedWisdom
        self.category_syntheses = {}      # category -> List[synthesis_ids]
        self.theme_collections = {}       # theme -> List[synthesis_ids]
        
        # Synthesis metrics
        self.synthesis_metrics = {
            'syntheses_created': 0,
            'insights_synthesized': 0,
            'themes_identified': 0,
            'community_validated_syntheses': 0,
            'cross_category_syntheses': 0
        }
        
        logger.info("Knowledge Synthesizer initialized")
    
    def _initialize_synthesis_templates(self) -> Dict[SynthesisMethod, Dict[str, Any]]:
        """Initialize templates for different synthesis methods"""
        
        return {
            SynthesisMethod.THEMATIC_CLUSTERING: {
                'structure': [
                    'introduction_and_context',
                    'core_themes_overview',
                    'detailed_wisdom_by_theme',
                    'practical_applications',
                    'safety_considerations',
                    'cultural_adaptations',
                    'further_resources'
                ],
                'narrative_style': 'organized_collection',
                'focus': 'thematic_coherence'
            },
            
            SynthesisMethod.EXPERIENTIAL_JOURNEY: {
                'structure': [
                    'journey_overview',
                    'early_stage_wisdom',
                    'middle_stage_insights',
                    'advanced_stage_understanding',
                    'integration_wisdom',
                    'supporting_others',
                    'ongoing_growth'
                ],
                'narrative_style': 'journey_narrative',
                'focus': 'progression_and_growth'
            },
            
            SynthesisMethod.PRACTICAL_PATHWAY: {
                'structure': [
                    'pathway_introduction',
                    'foundation_building',
                    'core_practices',
                    'advanced_techniques',
                    'troubleshooting_challenges',
                    'maintenance_and_sustainability',
                    'adaptation_guidance'
                ],
                'narrative_style': 'step_by_step_guide',
                'focus': 'actionable_implementation'
            },
            
            SynthesisMethod.CRISIS_RESOURCE_COMPILATION: {
                'structure': [
                    'immediate_safety_first',
                    'crisis_stabilization_techniques',
                    'emergency_coping_strategies',
                    'seeking_help_guidance',
                    'post_crisis_recovery',
                    'prevention_strategies',
                    'professional_resources'
                ],
                'narrative_style': 'emergency_resource_guide',
                'focus': 'crisis_intervention_and_safety'
            },
            
            SynthesisMethod.TRAUMA_INFORMED_GUIDE: {
                'structure': [
                    'trauma_informed_foundation',
                    'safety_and_choice_principles',
                    'body_awareness_wisdom',
                    'emotional_regulation_insights',
                    'relationship_healing_guidance',
                    'meaning_making_support',
                    'post_traumatic_growth_wisdom'
                ],
                'narrative_style': 'trauma_informed_narrative',
                'focus': 'trauma_healing_and_growth'
            }
        }
    
    def _initialize_clustering_algorithms(self) -> Dict[str, Any]:
        """Initialize algorithms for clustering related insights"""
        
        return {
            'semantic_similarity': {
                'keywords_weight': 0.3,
                'context_weight': 0.2,
                'category_weight': 0.2,
                'applicability_weight': 0.15,
                'outcome_weight': 0.15
            },
            'experiential_similarity': {
                'journey_stage_weight': 0.4,
                'challenge_type_weight': 0.3,
                'solution_approach_weight': 0.2,
                'outcome_pattern_weight': 0.1
            },
            'practical_similarity': {
                'action_type_weight': 0.4,
                'implementation_complexity_weight': 0.2,
                'resource_requirements_weight': 0.2,
                'time_commitment_weight': 0.1,
                'skill_level_weight': 0.1
            }
        }
    
    def _initialize_narrative_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize frameworks for creating coherent narratives"""
        
        return {
            'organized_collection': {
                'opening': "This collection brings together wisdom from our community about {topic}. "
                          "Each insight has been shared by someone with lived experience and validated "
                          "for safety and helpfulness.",
                'theme_introduction': "The following insights center around {theme}:",
                'wisdom_presentation': "Community members have found:",
                'safety_note': "Remember: Everyone's journey is unique. What works for one person may "
                              "need adaptation for another. Trust your instincts and seek professional "
                              "support when needed.",
                'closing': "This wisdom collection continues to grow as our community shares their "
                          "experiences and insights. Each piece of wisdom represents courage, growth, "
                          "and the generosity of sharing to help others."
            },
            
            'journey_narrative': {
                'opening': "Every healing journey is unique, yet many of us walk similar paths and "
                          "face similar challenges. This guide weaves together wisdom from community "
                          "members at different stages of their journey.",
                'stage_introduction': "During the {stage} stage, community members have discovered:",
                'transition': "As the journey progresses, wisdom often shifts to focus on:",
                'integration': "Many have found that integrating these insights happens gradually:",
                'closing': "Remember that healing is not linear. You may find yourself revisiting "
                          "earlier stages, and that's completely normal. Each stage brings its own "
                          "gifts and wisdom."
            },
            
            'step_by_step_guide': {
                'opening': "This practical guide synthesizes community wisdom into actionable steps "
                          "for {topic}. Each step builds on the previous ones, but feel free to "
                          "adapt the pace to your needs.",
                'step_introduction': "Step {number}: {title}",
                'practice_note': "Community members suggest:",
                'troubleshooting': "If you're finding this challenging:",
                'closing': "Remember: This is a practice, not a performance. Be gentle with yourself "
                          "as you implement these approaches, and celebrate small progress."
            }
        }
    
    def _initialize_safety_guidelines(self) -> Dict[WisdomCategory, List[str]]:
        """Initialize safety guidelines for different wisdom categories"""
        
        return {
            WisdomCategory.CRISIS_NAVIGATION: [
                "If you're in immediate danger, contact emergency services first",
                "These strategies are for additional support, not replacement of professional crisis intervention",
                "Trust your instincts about what feels safe and helpful",
                "Have a safety plan and support person identified",
                "If something doesn't feel right, stop and seek professional guidance"
            ],
            
            WisdomCategory.TRAUMA_RECOVERY: [
                "Trauma healing requires safety first - go at your own pace",
                "These insights are meant to complement, not replace, professional trauma therapy",
                "If you feel overwhelmed, please pause and seek support",
                "Everyone's trauma recovery is unique - adapt these insights to your needs",
                "Professional trauma therapy is strongly recommended for trauma recovery"
            ],
            
            WisdomCategory.EMOTIONAL_REGULATION: [
                "If emotions feel overwhelming, prioritize safety and seek support",
                "These techniques are tools, not requirements - use what helps",
                "Professional support can be invaluable for learning emotional regulation",
                "Be patient with yourself - emotional regulation skills take practice",
                "If you're struggling with persistent emotional difficulties, consider professional help"
            ],
            
            WisdomCategory.RELATIONSHIP_WISDOM: [
                "Healthy relationships should feel safe and respectful",
                "If you're in an abusive relationship, prioritize your safety first",
                "These insights are general guidance - each relationship is unique",
                "Professional couples therapy can provide valuable support",
                "Trust your instincts about what feels healthy and supportive"
            ]
        }
    
    async def synthesize_wisdom_collection(self, insights: List[WisdomInsight],
                                         synthesis_method: SynthesisMethod,
                                         scope: SynthesisScope,
                                         focus_topic: Optional[str] = None) -> SynthesizedWisdom:
        """
        Synthesize a collection of wisdom insights into a coherent guide.
        
        Args:
            insights: List of validated wisdom insights to synthesize
            synthesis_method: Method to use for synthesis
            scope: Scope of the synthesis
            focus_topic: Optional specific topic to focus on
        """
        try:
            # Filter and prepare insights
            validated_insights = await self._filter_validated_insights(insights)
            
            if len(validated_insights) < 2:
                raise ValueError("Need at least 2 validated insights for synthesis")
            
            # Determine category and create synthesis structure
            primary_category = await self._determine_primary_category(validated_insights)
            synthesis_structure = await self._create_synthesis_structure(
                validated_insights, synthesis_method, focus_topic
            )
            
            # Generate synthesized content
            synthesized_content = await self._generate_synthesized_content(
                validated_insights, synthesis_structure, synthesis_method
            )
            
            # Extract themes and applications
            key_themes = await self._extract_key_themes(validated_insights)
            practical_applications = await self._generate_practical_applications(validated_insights)
            
            # Generate safety and cultural considerations
            safety_guidelines = await self._generate_safety_guidelines(primary_category, validated_insights)
            cultural_considerations = await self._generate_cultural_considerations(validated_insights)
            trauma_informed_notes = await self._generate_trauma_informed_notes(validated_insights)
            
            # Determine target audience and prerequisites
            target_audience = await self._determine_target_audience(validated_insights, scope)
            prerequisites = await self._determine_prerequisites(validated_insights, synthesis_method)
            
            # Calculate evidence and validation scores
            evidence_strength = await self._calculate_evidence_strength(validated_insights)
            
            # Create synthesis
            synthesis_id = f"synthesis_{primary_category.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            synthesis = SynthesizedWisdom(
                synthesis_id=synthesis_id,
                title=await self._generate_synthesis_title(primary_category, focus_topic, scope),
                description=await self._generate_synthesis_description(validated_insights, synthesis_method),
                category=primary_category,
                synthesis_method=synthesis_method,
                scope=scope,
                source_insights=[insight.insight_id for insight in validated_insights],
                synthesized_content=synthesized_content,
                key_themes=key_themes,
                practical_applications=practical_applications,
                cultural_considerations=cultural_considerations,
                trauma_informed_notes=trauma_informed_notes,
                safety_guidelines=safety_guidelines,
                target_audience=target_audience,
                prerequisites=prerequisites,
                related_synthesis=[],  # Will be populated later
                evidence_strength=evidence_strength,
                community_validation=0.0,  # Will be populated through validation
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                usage_metrics={'views': 0, 'saves': 0, 'shares': 0, 'helpful_ratings': 0},
                effectiveness_data=None
            )
            
            # Store synthesis
            await self._store_synthesized_wisdom(synthesis)
            
            # Update metrics
            self.synthesis_metrics['syntheses_created'] += 1
            self.synthesis_metrics['insights_synthesized'] += len(validated_insights)
            self.synthesis_metrics['themes_identified'] += len(key_themes)
            
            logger.info(f"Created synthesis {synthesis_id}: {synthesis.title}")
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Error synthesizing wisdom collection: {e}")
            raise
    
    async def _filter_validated_insights(self, insights: List[WisdomInsight]) -> List[WisdomInsight]:
        """Filter insights to only include validated ones"""
        
        # In production, this would check validation status from validation results
        # For now, we filter based on wisdom score
        return [insight for insight in insights if insight.wisdom_score >= 0.6]
    
    async def _determine_primary_category(self, insights: List[WisdomInsight]) -> WisdomCategory:
        """Determine primary category for synthesis"""
        
        category_counts = {}
        for insight in insights:
            category = insight.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Return most common category
        return max(category_counts.items(), key=lambda x: x[1])[0]
    
    async def _create_synthesis_structure(self, insights: List[WisdomInsight],
                                        method: SynthesisMethod,
                                        focus_topic: Optional[str]) -> Dict[str, Any]:
        """Create structure for synthesis based on method"""
        
        template = self.synthesis_templates.get(method, {})
        structure_sections = template.get('structure', [])
        
        # Organize insights by structure sections
        organized_insights = {}
        
        if method == SynthesisMethod.THEMATIC_CLUSTERING:
            organized_insights = await self._organize_by_themes(insights)
        elif method == SynthesisMethod.EXPERIENTIAL_JOURNEY:
            organized_insights = await self._organize_by_journey_stage(insights)
        elif method == SynthesisMethod.PRACTICAL_PATHWAY:
            organized_insights = await self._organize_by_practicality(insights)
        elif method == SynthesisMethod.CRISIS_RESOURCE_COMPILATION:
            organized_insights = await self._organize_by_crisis_relevance(insights)
        elif method == SynthesisMethod.TRAUMA_INFORMED_GUIDE:
            organized_insights = await self._organize_by_trauma_principles(insights)
        
        return {
            'sections': structure_sections,
            'organized_insights': organized_insights,
            'template': template,
            'focus_topic': focus_topic
        }
    
    async def _organize_by_themes(self, insights: List[WisdomInsight]) -> Dict[str, List[WisdomInsight]]:
        """Organize insights by thematic similarity"""
        
        themes = {}
        
        # Simple thematic clustering based on content keywords
        for insight in insights:
            content_words = set(insight.content.lower().split())
            
            # Assign to themes based on content
            assigned_theme = None
            
            # Define theme keywords
            theme_keywords = {
                'coping_techniques': {'breathing', 'grounding', 'calming', 'soothing', 'technique'},
                'emotional_awareness': {'feelings', 'emotions', 'awareness', 'recognize', 'understand'},
                'self_compassion': {'self-compassion', 'gentle', 'kind', 'forgiveness', 'patience'},
                'relationships': {'relationship', 'communication', 'boundaries', 'connection', 'support'},
                'daily_practices': {'daily', 'routine', 'practice', 'habit', 'regular', 'consistent'},
                'crisis_support': {'crisis', 'emergency', 'immediate', 'urgent', 'safety'},
                'growth_mindset': {'growth', 'learning', 'progress', 'development', 'change'}
            }
            
            # Find best matching theme
            best_theme = None
            best_score = 0
            
            for theme, keywords in theme_keywords.items():
                overlap = len(content_words & keywords)
                if overlap > best_score:
                    best_score = overlap
                    best_theme = theme
            
            if best_theme and best_score > 0:
                assigned_theme = best_theme
            else:
                assigned_theme = 'general_wisdom'
            
            if assigned_theme not in themes:
                themes[assigned_theme] = []
            themes[assigned_theme].append(insight)
        
        return themes
    
    async def _organize_by_journey_stage(self, insights: List[WisdomInsight]) -> Dict[str, List[WisdomInsight]]:
        """Organize insights by healing journey stage"""
        
        stages = {
            'early_stage': [],
            'middle_stage': [],
            'advanced_stage': [],
            'integration_stage': []
        }
        
        for insight in insights:
            content_lower = insight.content.lower()
            
            # Classify by journey stage indicators
            if any(word in content_lower for word in ['beginning', 'start', 'first', 'new to', 'just started']):
                stages['early_stage'].append(insight)
            elif any(word in content_lower for word in ['years of', 'experienced', 'advanced', 'deep work']):
                stages['advanced_stage'].append(insight)
            elif any(word in content_lower for word in ['integrate', 'maintain', 'sustain', 'ongoing']):
                stages['integration_stage'].append(insight)
            else:
                stages['middle_stage'].append(insight)  # Default
        
        return stages
    
    async def _organize_by_practicality(self, insights: List[WisdomInsight]) -> Dict[str, List[WisdomInsight]]:
        """Organize insights by practical implementation level"""
        
        levels = {
            'foundation_building': [],
            'core_practices': [],
            'advanced_techniques': [],
            'troubleshooting': [],
            'maintenance': []
        }
        
        for insight in insights:
            content_lower = insight.content.lower()
            
            if any(word in content_lower for word in ['basic', 'foundation', 'start with', 'first step']):
                levels['foundation_building'].append(insight)
            elif any(word in content_lower for word in ['advanced', 'complex', 'sophisticated']):
                levels['advanced_techniques'].append(insight)
            elif any(word in content_lower for word in ['problem', 'difficult', 'challenge', 'when stuck']):
                levels['troubleshooting'].append(insight)
            elif any(word in content_lower for word in ['maintain', 'sustain', 'keep going', 'long term']):
                levels['maintenance'].append(insight)
            else:
                levels['core_practices'].append(insight)  # Default
        
        return levels
    
    async def _organize_by_crisis_relevance(self, insights: List[WisdomInsight]) -> Dict[str, List[WisdomInsight]]:
        """Organize insights by crisis intervention relevance"""
        
        crisis_categories = {
            'immediate_safety': [],
            'stabilization': [],
            'coping_strategies': [],
            'seeking_help': [],
            'recovery': []
        }
        
        for insight in insights:
            content_lower = insight.content.lower()
            
            if any(word in content_lower for word in ['danger', 'safety', 'emergency', 'immediate']):
                crisis_categories['immediate_safety'].append(insight)
            elif any(word in content_lower for word in ['stabilize', 'calm', 'ground', 'center']):
                crisis_categories['stabilization'].append(insight)
            elif any(word in content_lower for word in ['help', 'support', 'call', 'reach out']):
                crisis_categories['seeking_help'].append(insight)
            elif any(word in content_lower for word in ['after', 'recovery', 'healing', 'moving forward']):
                crisis_categories['recovery'].append(insight)
            else:
                crisis_categories['coping_strategies'].append(insight)  # Default
        
        return crisis_categories
    
    async def _organize_by_trauma_principles(self, insights: List[WisdomInsight]) -> Dict[str, List[WisdomInsight]]:
        """Organize insights by trauma-informed principles"""
        
        principles = {
            'safety_and_choice': [],
            'body_awareness': [],
            'emotional_regulation': [],
            'relationship_healing': [],
            'meaning_making': [],
            'post_traumatic_growth': []
        }
        
        for insight in insights:
            content_lower = insight.content.lower()
            
            if any(word in content_lower for word in ['safe', 'choice', 'control', 'agency']):
                principles['safety_and_choice'].append(insight)
            elif any(word in content_lower for word in ['body', 'physical', 'somatic', 'embodied']):
                principles['body_awareness'].append(insight)
            elif any(word in content_lower for word in ['emotions', 'feelings', 'regulate', 'manage']):
                principles['emotional_regulation'].append(insight)
            elif any(word in content_lower for word in ['relationship', 'connection', 'trust', 'attachment']):
                principles['relationship_healing'].append(insight)
            elif any(word in content_lower for word in ['meaning', 'purpose', 'sense', 'why']):
                principles['meaning_making'].append(insight)
            else:
                principles['post_traumatic_growth'].append(insight)  # Default
        
        return principles
    
    async def _generate_synthesized_content(self, insights: List[WisdomInsight],
                                          structure: Dict[str, Any],
                                          method: SynthesisMethod) -> str:
        """Generate the main synthesized content"""
        
        template = structure['template']
        organized_insights = structure['organized_insights']
        narrative_style = template.get('narrative_style', 'organized_collection')
        
        # Get narrative framework
        framework = self.narrative_frameworks.get(narrative_style, {})
        
        # Build content sections
        content_sections = []
        
        # Opening
        opening = framework.get('opening', '').format(
            topic=structure.get('focus_topic', 'healing and growth')
        )
        content_sections.append(opening)
        
        # Process each organized section
        for section_name, section_insights in organized_insights.items():
            if not section_insights:
                continue
            
            # Section introduction
            section_title = section_name.replace('_', ' ').title()
            section_intro = framework.get('theme_introduction', '').format(
                theme=section_title.lower()
            )
            
            content_sections.append(f"\n## {section_title}\n")
            content_sections.append(section_intro)
            
            # Present insights in this section
            wisdom_presentation = framework.get('wisdom_presentation', 'Community members have found:')
            content_sections.append(f"\n{wisdom_presentation}\n")
            
            for insight in section_insights[:5]:  # Limit to top 5 per section
                # Anonymize and present insight
                anonymized_content = await self._anonymize_insight_content(insight)
                content_sections.append(f"• {anonymized_content}")
                
                # Add supporting evidence if available
                if insight.supporting_evidence:
                    evidence = insight.supporting_evidence[0]  # First piece of evidence
                    content_sections.append(f"  _{evidence}_")
        
        # Safety note
        safety_note = framework.get('safety_note', '')
        if safety_note:
            content_sections.append(f"\n## Important Note\n{safety_note}")
        
        # Closing
        closing = framework.get('closing', '')
        if closing:
            content_sections.append(f"\n{closing}")
        
        return '\n'.join(content_sections)
    
    async def _anonymize_insight_content(self, insight: WisdomInsight) -> str:
        """Anonymize insight content for synthesis"""
        
        content = insight.content
        
        # Remove personal identifiers
        anonymized = content.replace(' I ', ' someone ')
        anonymized = anonymized.replace('I ', 'A person ')
        anonymized = anonymized.replace(' my ', ' their ')
        anonymized = anonymized.replace('My ', 'Their ')
        anonymized = anonymized.replace(' me ', ' them ')
        anonymized = anonymized.replace('Me ', 'They ')
        
        return anonymized
    
    async def _extract_key_themes(self, insights: List[WisdomInsight]) -> List[str]:
        """Extract key themes from insights"""
        
        theme_keywords = {}
        
        for insight in insights:
            words = insight.content.lower().split()
            
            # Count significant keywords
            significant_words = [word for word in words 
                               if len(word) > 4 and word not in ['that', 'this', 'with', 'from', 'when', 'what']]
            
            for word in significant_words:
                theme_keywords[word] = theme_keywords.get(word, 0) + 1
        
        # Extract top themes
        sorted_themes = sorted(theme_keywords.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to readable theme names
        top_themes = []
        for theme, count in sorted_themes[:8]:  # Top 8 themes
            if count >= 2:  # Must appear in at least 2 insights
                readable_theme = theme.replace('_', ' ').title()
                top_themes.append(readable_theme)
        
        return top_themes
    
    async def _generate_practical_applications(self, insights: List[WisdomInsight]) -> List[str]:
        """Generate practical applications from insights"""
        
        applications = []
        
        # Extract actionable insights
        for insight in insights:
            content_lower = insight.content.lower()
            
            # Look for action-oriented language
            if any(word in content_lower for word in ['try', 'practice', 'use', 'do', 'start']):
                # Extract the actionable part
                sentences = insight.content.split('.')
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ['try', 'practice', 'use', 'do', 'start']):
                        if len(sentence.strip()) > 10:
                            applications.append(sentence.strip())
                            break
        
        # Deduplicate and limit
        unique_applications = list(set(applications))
        return unique_applications[:10]  # Top 10 applications
    
    async def _generate_safety_guidelines(self, category: WisdomCategory, 
                                        insights: List[WisdomInsight]) -> List[str]:
        """Generate safety guidelines for the synthesis"""
        
        # Start with category-specific guidelines
        guidelines = self.safety_guidelines.get(category, []).copy()
        
        # Add insight-specific safety considerations
        for insight in insights:
            guidelines.extend(insight.care_considerations)
        
        # Add general guidelines
        guidelines.extend([
            "Everyone's healing journey is unique - adapt these insights to your situation",
            "If you're struggling with serious mental health concerns, please seek professional support",
            "Trust your instincts about what feels safe and helpful for you"
        ])
        
        # Remove duplicates and return
        return list(set(guidelines))
    
    async def _generate_cultural_considerations(self, insights: List[WisdomInsight]) -> List[str]:
        """Generate cultural considerations for the synthesis"""
        
        considerations = []
        
        # Check for cultural contexts in insights
        cultural_contexts = set()
        for insight in insights:
            if insight.cultural_context:
                cultural_contexts.add(insight.cultural_context)
        
        if cultural_contexts:
            considerations.append(f"This wisdom comes from diverse cultural contexts: {', '.join(cultural_contexts)}")
        
        # Add general cultural considerations
        considerations.extend([
            "Healing approaches vary across cultures - adapt these insights to your cultural context",
            "Some practices may need modification to align with your cultural or religious values",
            "Community and family structures differ across cultures - consider your context when applying this wisdom"
        ])
        
        return considerations
    
    async def _generate_trauma_informed_notes(self, insights: List[WisdomInsight]) -> List[str]:
        """Generate trauma-informed notes for the synthesis"""
        
        notes = [
            "Go at your own pace - healing cannot be rushed",
            "You have choice and control over how you engage with this wisdom",
            "If something doesn't feel right for you, trust that instinct",
            "Safety is always the first priority in healing work"
        ]
        
        # Add specific notes based on insights
        trauma_informed_count = sum(1 for insight in insights if insight.trauma_informed)
        
        if trauma_informed_count / len(insights) > 0.8:
            notes.append("This collection emphasizes trauma-informed approaches throughout")
        
        return notes
    
    async def _determine_target_audience(self, insights: List[WisdomInsight], 
                                       scope: SynthesisScope) -> List[str]:
        """Determine target audience for the synthesis"""
        
        audience = []
        
        # Based on applicability tags
        all_tags = []
        for insight in insights:
            all_tags.extend(insight.applicability_tags)
        
        # Count tag frequency
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Add most common audiences
        for tag, count in tag_counts.items():
            if count >= len(insights) * 0.3:  # At least 30% of insights apply
                audience.append(tag.replace('_', ' '))
        
        # Add scope-based audience
        if scope == SynthesisScope.CRISIS_SUPPORT:
            audience.append("people in crisis")
        elif scope == SynthesisScope.JOURNEY_STAGE:
            audience.append("people at specific healing stages")
        
        return audience[:5]  # Limit to top 5
    
    async def _determine_prerequisites(self, insights: List[WisdomInsight], 
                                     method: SynthesisMethod) -> List[str]:
        """Determine prerequisites for using the synthesis"""
        
        prerequisites = []
        
        # Method-specific prerequisites
        if method == SynthesisMethod.CRISIS_RESOURCE_COMPILATION:
            prerequisites.extend([
                "Immediate safety assessment",
                "Professional crisis support availability"
            ])
        elif method == SynthesisMethod.TRAUMA_INFORMED_GUIDE:
            prerequisites.extend([
                "Basic safety and stabilization",
                "Trauma-informed therapy support recommended"
            ])
        elif method == SynthesisMethod.PRACTICAL_PATHWAY:
            prerequisites.extend([
                "Readiness to engage in active healing work",
                "Basic emotional regulation skills"
            ])
        
        # Check insight requirements
        context_requirements = []
        for insight in insights:
            context_requirements.extend(insight.context_requirements)
        
        # Add common requirements
        requirement_counts = {}
        for req in context_requirements:
            requirement_counts[req] = requirement_counts.get(req, 0) + 1
        
        for req, count in requirement_counts.items():
            if count >= len(insights) * 0.3:  # Required by 30% of insights
                prerequisites.append(req.replace('_', ' '))
        
        return prerequisites
    
    async def _calculate_evidence_strength(self, insights: List[WisdomInsight]) -> float:
        """Calculate overall evidence strength of the synthesis"""
        
        if not insights:
            return 0.0
        
        # Average wisdom scores
        avg_wisdom_score = sum(insight.wisdom_score for insight in insights) / len(insights)
        
        # Factor in validation scores
        avg_validation_score = sum(insight.validation_score for insight in insights) / len(insights)
        
        # Factor in supporting evidence
        evidence_strength = 0.0
        for insight in insights:
            if insight.supporting_evidence:
                evidence_strength += len(insight.supporting_evidence) * 0.1
        
        evidence_strength = min(1.0, evidence_strength / len(insights))
        
        # Combine factors
        overall_strength = (avg_wisdom_score * 0.4 + 
                          avg_validation_score * 0.4 + 
                          evidence_strength * 0.2)
        
        return min(1.0, overall_strength)
    
    async def _generate_synthesis_title(self, category: WisdomCategory, 
                                      focus_topic: Optional[str],
                                      scope: SynthesisScope) -> str:
        """Generate title for the synthesis"""
        
        category_titles = {
            WisdomCategory.COPING_STRATEGIES: "Community Wisdom on Coping Strategies",
            WisdomCategory.HEALING_INSIGHTS: "Healing Insights from Our Community",
            WisdomCategory.TRAUMA_RECOVERY: "Trauma Recovery Wisdom Collection",
            WisdomCategory.EMOTIONAL_REGULATION: "Emotional Regulation: Community Insights",
            WisdomCategory.CRISIS_NAVIGATION: "Crisis Navigation: Emergency Wisdom Guide",
            WisdomCategory.SELF_CARE_PRACTICES: "Self-Care Wisdom from Lived Experience",
            WisdomCategory.RELATIONSHIP_WISDOM: "Relationship Healing: Community Insights"
        }
        
        base_title = category_titles.get(category, f"{category.value.replace('_', ' ').title()} Wisdom")
        
        if focus_topic:
            base_title = f"{base_title}: {focus_topic.title()}"
        
        # Add scope qualifier
        if scope == SynthesisScope.CRISIS_SUPPORT:
            base_title += " - Crisis Support Edition"
        elif scope == SynthesisScope.JOURNEY_STAGE:
            base_title += " - Journey Stage Guide"
        
        return base_title
    
    async def _generate_synthesis_description(self, insights: List[WisdomInsight], 
                                            method: SynthesisMethod) -> str:
        """Generate description for the synthesis"""
        
        insight_count = len(insights)
        categories = list(set(insight.category for insight in insights))
        
        description = f"This collection synthesizes {insight_count} pieces of wisdom from community members "
        description += f"with lived experience in {', '.join(cat.value.replace('_', ' ') for cat in categories)}. "
        
        method_descriptions = {
            SynthesisMethod.THEMATIC_CLUSTERING: "Organized by key themes and insights.",
            SynthesisMethod.EXPERIENTIAL_JOURNEY: "Structured as a healing journey progression.",
            SynthesisMethod.PRACTICAL_PATHWAY: "Presented as actionable steps and practices.",
            SynthesisMethod.CRISIS_RESOURCE_COMPILATION: "Compiled for crisis support and emergency guidance.",
            SynthesisMethod.TRAUMA_INFORMED_GUIDE: "Structured using trauma-informed healing principles."
        }
        
        description += method_descriptions.get(method, "Carefully curated and validated for safety and helpfulness.")
        
        return description
    
    async def _store_synthesized_wisdom(self, synthesis: SynthesizedWisdom) -> None:
        """Store synthesized wisdom"""
        
        # Store in main collection
        self.synthesized_wisdom[synthesis.synthesis_id] = synthesis
        
        # Store in category collection
        category = synthesis.category
        if category not in self.category_syntheses:
            self.category_syntheses[category] = []
        self.category_syntheses[category].append(synthesis.synthesis_id)
        
        # Store in theme collections
        for theme in synthesis.key_themes:
            theme_key = theme.lower().replace(' ', '_')
            if theme_key not in self.theme_collections:
                self.theme_collections[theme_key] = []
            self.theme_collections[theme_key].append(synthesis.synthesis_id)
        
        logger.debug(f"Stored synthesized wisdom {synthesis.synthesis_id}")
    
    def get_synthesis_by_category(self, category: WisdomCategory) -> List[SynthesizedWisdom]:
        """Get all syntheses for a category"""
        
        synthesis_ids = self.category_syntheses.get(category, [])
        return [self.synthesized_wisdom[id] for id in synthesis_ids if id in self.synthesized_wisdom]
    
    def get_synthesis_metrics(self) -> Dict[str, Any]:
        """Get synthesis metrics"""
        
        return {
            'syntheses_created': self.synthesis_metrics['syntheses_created'],
            'insights_synthesized': self.synthesis_metrics['insights_synthesized'],
            'themes_identified': self.synthesis_metrics['themes_identified'],
            'community_validated_syntheses': self.synthesis_metrics['community_validated_syntheses'],
            'cross_category_syntheses': self.synthesis_metrics['cross_category_syntheses'],
            'total_synthesized_wisdom': len(self.synthesized_wisdom),
            'categories_with_syntheses': len(self.category_syntheses),
            'themes_with_collections': len(self.theme_collections),
            'synthesis_methods_used': len(SynthesisMethod),
            'system_health': {
                'synthesis_templates_loaded': len(self.synthesis_templates) > 0,
                'clustering_algorithms_loaded': len(self.clustering_algorithms) > 0,
                'narrative_frameworks_loaded': len(self.narrative_frameworks) > 0,
                'safety_guidelines_loaded': len(self.safety_guidelines) > 0
            }
        }