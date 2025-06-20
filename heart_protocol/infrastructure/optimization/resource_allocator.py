"""
Wellbeing-Focused Resource Allocator

Resource allocation system that prioritizes user wellbeing and emotional safety
over traditional efficiency metrics, ensuring those who need help most get priority.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import heapq
import json
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Resource allocation strategies based on wellbeing priorities"""
    CRISIS_FIRST = "crisis_first"                    # Crisis users get unlimited resources
    CARE_WEIGHTED = "care_weighted"                  # Weighted by care needs
    HEALING_SUPPORTIVE = "healing_supportive"       # Supports healing journey progress
    COMMUNITY_BALANCED = "community_balanced"       # Balances individual and community needs
    GENTLE_SCALING = "gentle_scaling"               # Gradual resource scaling for sensitive users
    USER_CONTROLLED = "user_controlled"             # User defines resource preferences


class CareQueue(Enum):
    """Priority queues based on care urgency"""
    IMMEDIATE_CRISIS = "immediate_crisis"           # Immediate life-threatening situations
    URGENT_SUPPORT = "urgent_support"              # Urgent support seeking
    HEALING_PRIORITY = "healing_priority"          # Active healing work
    STABLE_SUPPORT = "stable_support"              # Stable users needing support
    COMMUNITY_BUILDING = "community_building"      # Community building activities
    BACKGROUND_PROCESSING = "background_processing" # Non-urgent background tasks


class ResourceType(Enum):
    """Types of system resources"""
    CPU_PROCESSING = "cpu_processing"              # CPU computation time
    MEMORY_ALLOCATION = "memory_allocation"        # Memory allocation
    DATABASE_CONNECTIONS = "database_connections"  # Database connection pool
    API_RATE_LIMITS = "api_rate_limits"           # API request quotas
    CACHE_SPACE = "cache_space"                   # Cache storage allocation
    NETWORK_BANDWIDTH = "network_bandwidth"       # Network bandwidth
    STORAGE_SPACE = "storage_space"               # Storage allocation
    CRISIS_RESPONDERS = "crisis_responders"       # Human crisis responder availability


@dataclass
class ResourceRequest:
    """Request for system resources with care context"""
    request_id: str
    user_id: str
    operation: str
    care_queue: CareQueue
    resource_requirements: Dict[ResourceType, float]
    estimated_duration_ms: int
    user_context: Dict[str, Any]
    healing_priority: int  # 1-10, 10 being highest
    submitted_at: datetime
    max_wait_time_ms: Optional[int] = None
    gentle_loading_preferred: bool = False
    
    def __lt__(self, other):
        """Priority comparison for heap queue"""
        # Higher healing priority = lower value for min-heap
        return (10 - self.healing_priority) < (10 - other.healing_priority)


@dataclass
class ResourceAllocation:
    """Allocated resources for a request"""
    allocation_id: str
    request_id: str
    allocated_resources: Dict[ResourceType, float]
    allocation_started: datetime
    estimated_completion: datetime
    actual_completion: Optional[datetime] = None
    allocation_efficiency: Optional[float] = None
    user_satisfaction: Optional[float] = None
    wellbeing_impact: Optional[float] = None


@dataclass
class CareResourcePool:
    """Resource pool optimized for care delivery"""
    pool_id: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    reserved_for_crisis: float
    gentle_allocation_rate: float
    allocation_history: List[ResourceAllocation] = field(default_factory=list)
    current_allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)


class ResourceAllocator:
    """
    Resource allocation system that prioritizes user wellbeing and emotional safety.
    
    Core Principles:
    - Crisis situations get unlimited resources immediately
    - Allocation decisions consider emotional impact on users
    - Gentle resource scaling for overwhelmed or sensitive users
    - Community needs balanced with individual care
    - User consent and preferences respected in allocation
    - Healing journey progress takes priority over efficiency
    - Fair allocation that prevents resource monopolization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_pools = self._initialize_resource_pools()
        self.request_queues = self._initialize_care_queues()
        self.allocation_history: List[ResourceAllocation] = []
        self.user_allocation_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Allocation state
        self.allocator_active = False
        self.allocation_callbacks: List[Callable] = []
        
    def _initialize_resource_pools(self) -> Dict[ResourceType, CareResourcePool]:
        """Initialize resource pools with care-focused configuration"""
        pools = {}
        
        # CPU Processing Pool
        pools[ResourceType.CPU_PROCESSING] = CareResourcePool(
            pool_id="cpu_care_pool",
            resource_type=ResourceType.CPU_PROCESSING,
            total_capacity=100.0,  # 100% CPU capacity
            available_capacity=100.0,
            reserved_for_crisis=20.0,  # 20% reserved for crisis
            gentle_allocation_rate=0.1  # 10% increments for gentle scaling
        )
        
        # Memory Allocation Pool
        pools[ResourceType.MEMORY_ALLOCATION] = CareResourcePool(
            pool_id="memory_care_pool",
            resource_type=ResourceType.MEMORY_ALLOCATION,
            total_capacity=16384.0,  # 16GB in MB
            available_capacity=16384.0,
            reserved_for_crisis=3276.8,  # 20% reserved for crisis
            gentle_allocation_rate=512.0  # 512MB increments for gentle scaling
        )
        
        # Database Connections Pool
        pools[ResourceType.DATABASE_CONNECTIONS] = CareResourcePool(
            pool_id="db_care_pool",
            resource_type=ResourceType.DATABASE_CONNECTIONS,
            total_capacity=50.0,  # 50 database connections
            available_capacity=50.0,
            reserved_for_crisis=10.0,  # 20% reserved for crisis
            gentle_allocation_rate=2.0  # 2 connection increments
        )
        
        # API Rate Limits Pool
        pools[ResourceType.API_RATE_LIMITS] = CareResourcePool(
            pool_id="api_care_pool",
            resource_type=ResourceType.API_RATE_LIMITS,
            total_capacity=10000.0,  # 10k requests per hour
            available_capacity=10000.0,
            reserved_for_crisis=2000.0,  # 20% reserved for crisis
            gentle_allocation_rate=100.0  # 100 request increments
        )
        
        # Cache Space Pool
        pools[ResourceType.CACHE_SPACE] = CareResourcePool(
            pool_id="cache_care_pool",
            resource_type=ResourceType.CACHE_SPACE,
            total_capacity=8192.0,  # 8GB cache in MB
            available_capacity=8192.0,
            reserved_for_crisis=1638.4,  # 20% reserved for crisis
            gentle_allocation_rate=256.0  # 256MB increments
        )
        
        # Network Bandwidth Pool
        pools[ResourceType.NETWORK_BANDWIDTH] = CareResourcePool(
            pool_id="network_care_pool",
            resource_type=ResourceType.NETWORK_BANDWIDTH,
            total_capacity=1000.0,  # 1Gbps in Mbps
            available_capacity=1000.0,
            reserved_for_crisis=200.0,  # 20% reserved for crisis
            gentle_allocation_rate=50.0  # 50Mbps increments
        )
        
        # Storage Space Pool
        pools[ResourceType.STORAGE_SPACE] = CareResourcePool(
            pool_id="storage_care_pool",
            resource_type=ResourceType.STORAGE_SPACE,
            total_capacity=1048576.0,  # 1TB in MB
            available_capacity=1048576.0,
            reserved_for_crisis=209715.2,  # 20% reserved for crisis
            gentle_allocation_rate=10240.0  # 10GB increments
        )
        
        # Crisis Responders Pool (human resources)
        pools[ResourceType.CRISIS_RESPONDERS] = CareResourcePool(
            pool_id="crisis_responder_pool",
            resource_type=ResourceType.CRISIS_RESPONDERS,
            total_capacity=10.0,  # 10 crisis responders available
            available_capacity=10.0,
            reserved_for_crisis=10.0,  # 100% reserved for crisis
            gentle_allocation_rate=1.0  # 1 responder increments
        )
        
        return pools
    
    def _initialize_care_queues(self) -> Dict[CareQueue, List[ResourceRequest]]:
        """Initialize priority queues for different care contexts"""
        return {
            CareQueue.IMMEDIATE_CRISIS: [],
            CareQueue.URGENT_SUPPORT: [],
            CareQueue.HEALING_PRIORITY: [],
            CareQueue.STABLE_SUPPORT: [],
            CareQueue.COMMUNITY_BUILDING: [],
            CareQueue.BACKGROUND_PROCESSING: []
        }
    
    async def start_allocation_processing(self):
        """Start processing resource allocation requests"""
        if self.allocator_active:
            return
        
        self.allocator_active = True
        logger.info("Started wellbeing-focused resource allocation")
        
        # Start allocation tasks
        asyncio.create_task(self._process_allocation_queues())
        asyncio.create_task(self._monitor_resource_health())
        asyncio.create_task(self._optimize_allocations())
    
    async def stop_allocation_processing(self):
        """Stop processing resource allocations"""
        self.allocator_active = False
        logger.info("Stopped wellbeing-focused resource allocation")
    
    async def request_resources(self, user_id: str, operation: str,
                              care_queue: CareQueue,
                              resource_requirements: Dict[ResourceType, float],
                              estimated_duration_ms: int,
                              user_context: Optional[Dict[str, Any]] = None,
                              healing_priority: int = 5,
                              gentle_loading_preferred: bool = False) -> str:
        """
        Request resources with care context prioritization
        
        Args:
            user_id: User requesting resources
            operation: Operation requiring resources
            care_queue: Priority queue based on care urgency
            resource_requirements: Required resources by type
            estimated_duration_ms: Estimated operation duration
            user_context: Additional user context for allocation decisions
            healing_priority: Priority based on healing impact (1-10)
            gentle_loading_preferred: Whether user prefers gentle resource allocation
            
        Returns:
            Request ID for tracking
        """
        try:
            request_id = f"req_{user_id}_{operation}_{datetime.utcnow().isoformat()}"
            
            # Create resource request
            request = ResourceRequest(
                request_id=request_id,
                user_id=user_id,
                operation=operation,
                care_queue=care_queue,
                resource_requirements=resource_requirements,
                estimated_duration_ms=estimated_duration_ms,
                user_context=user_context or {},
                healing_priority=healing_priority,
                submitted_at=datetime.utcnow(),
                gentle_loading_preferred=gentle_loading_preferred
            )
            
            # Set max wait time based on care urgency
            if care_queue == CareQueue.IMMEDIATE_CRISIS:
                request.max_wait_time_ms = 0  # No wait for crisis
            elif care_queue == CareQueue.URGENT_SUPPORT:
                request.max_wait_time_ms = 500  # Max 500ms wait
            elif care_queue == CareQueue.HEALING_PRIORITY:
                request.max_wait_time_ms = 2000  # Max 2s wait
            else:
                request.max_wait_time_ms = 10000  # Max 10s wait for others
            
            # Add to appropriate queue
            heapq.heappush(self.request_queues[care_queue], request)
            
            logger.info(f"Resource request {request_id} added to {care_queue.value} queue")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to request resources: {str(e)}")
            return ""
    
    async def _process_allocation_queues(self):
        """Process resource allocation requests by priority"""
        while self.allocator_active:
            try:
                # Process queues in priority order
                queue_order = [
                    CareQueue.IMMEDIATE_CRISIS,
                    CareQueue.URGENT_SUPPORT,
                    CareQueue.HEALING_PRIORITY,
                    CareQueue.STABLE_SUPPORT,
                    CareQueue.COMMUNITY_BUILDING,
                    CareQueue.BACKGROUND_PROCESSING
                ]
                
                for care_queue in queue_order:
                    queue = self.request_queues[care_queue]
                    
                    # Process up to 5 requests from each queue per cycle
                    processed = 0
                    while queue and processed < 5:
                        request = heapq.heappop(queue)
                        
                        # Check if request has expired
                        if self._request_expired(request):
                            logger.warning(f"Request {request.request_id} expired")
                            continue
                        
                        # Attempt allocation
                        allocation = await self._allocate_resources(request)
                        if allocation:
                            processed += 1
                            
                            # Trigger allocation callbacks
                            for callback in self.allocation_callbacks:
                                try:
                                    await callback(allocation)
                                except Exception as e:
                                    logger.error(f"Allocation callback failed: {str(e)}")
                        else:
                            # Put request back in queue if allocation failed
                            heapq.heappush(queue, request)
                            break  # Try other queues
                
                await asyncio.sleep(0.1)  # Small delay between processing cycles
                
            except Exception as e:
                logger.error(f"Allocation queue processing error: {str(e)}")
                await asyncio.sleep(1)  # Wait longer on error
    
    def _request_expired(self, request: ResourceRequest) -> bool:
        """Check if request has exceeded max wait time"""
        if not request.max_wait_time_ms:
            return False
        
        wait_time_ms = (datetime.utcnow() - request.submitted_at).total_seconds() * 1000
        return wait_time_ms > request.max_wait_time_ms
    
    async def _allocate_resources(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Allocate resources for a request based on care priorities"""
        try:
            # Check if we can satisfy the request
            if not self._can_satisfy_request(request):
                return None
            
            # Apply user preferences
            user_prefs = self.user_allocation_preferences.get(request.user_id, {})
            
            # Allocate resources
            allocated_resources = {}
            for resource_type, amount in request.resource_requirements.items():
                pool = self.resource_pools[resource_type]
                
                # Apply gentle loading if preferred
                if request.gentle_loading_preferred and resource_type != ResourceType.CRISIS_RESPONDERS:
                    amount = min(amount, pool.gentle_allocation_rate)
                
                # Crisis gets unlimited access to reserved resources
                if request.care_queue == CareQueue.IMMEDIATE_CRISIS:
                    # Crisis can use all available resources
                    available = pool.available_capacity
                else:
                    # Other users respect crisis reservation
                    available = pool.available_capacity - pool.reserved_for_crisis
                
                if available >= amount:
                    allocated_resources[resource_type] = amount
                    pool.available_capacity -= amount
                else:
                    # Can't satisfy request, rollback allocations
                    for prev_type, prev_amount in allocated_resources.items():
                        self.resource_pools[prev_type].available_capacity += prev_amount
                    return None
            
            # Create allocation record
            allocation = ResourceAllocation(
                allocation_id=f"alloc_{request.request_id}",
                request_id=request.request_id,
                allocated_resources=allocated_resources,
                allocation_started=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(milliseconds=request.estimated_duration_ms)
            )
            
            # Record allocation
            self.allocation_history.append(allocation)
            for resource_type in allocated_resources:
                self.resource_pools[resource_type].current_allocations[allocation.allocation_id] = allocation
            
            logger.info(f"Allocated resources for {request.request_id}: {allocated_resources}")
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to allocate resources: {str(e)}")
            return None
    
    def _can_satisfy_request(self, request: ResourceRequest) -> bool:
        """Check if request can be satisfied with current resources"""
        for resource_type, amount in request.resource_requirements.items():
            pool = self.resource_pools[resource_type]
            
            if request.care_queue == CareQueue.IMMEDIATE_CRISIS:
                # Crisis can use all available resources
                if pool.available_capacity < amount:
                    return False
            else:
                # Other users respect crisis reservation
                available = pool.available_capacity - pool.reserved_for_crisis
                if available < amount:
                    return False
        
        return True
    
    async def release_resources(self, allocation_id: str,
                              actual_duration_ms: Optional[int] = None,
                              user_satisfaction: Optional[float] = None,
                              wellbeing_impact: Optional[float] = None) -> bool:
        """
        Release allocated resources and record metrics
        
        Args:
            allocation_id: ID of allocation to release
            actual_duration_ms: Actual duration of resource usage
            user_satisfaction: User satisfaction score (1-10)
            wellbeing_impact: Measured wellbeing impact (1-10)
            
        Returns:
            True if resources were successfully released
        """
        try:
            # Find allocation
            allocation = None
            for pool in self.resource_pools.values():
                if allocation_id in pool.current_allocations:
                    allocation = pool.current_allocations[allocation_id]
                    break
            
            if not allocation:
                logger.warning(f"Allocation {allocation_id} not found for release")
                return False
            
            # Release resources back to pools
            for resource_type, amount in allocation.allocated_resources.items():
                pool = self.resource_pools[resource_type]
                pool.available_capacity += amount
                
                # Remove from current allocations
                if allocation_id in pool.current_allocations:
                    del pool.current_allocations[allocation_id]
            
            # Update allocation record
            allocation.actual_completion = datetime.utcnow()
            allocation.user_satisfaction = user_satisfaction
            allocation.wellbeing_impact = wellbeing_impact
            
            if actual_duration_ms:
                estimated_ms = (allocation.estimated_completion - allocation.allocation_started).total_seconds() * 1000
                allocation.allocation_efficiency = estimated_ms / actual_duration_ms
            
            logger.info(f"Released resources for allocation {allocation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to release resources: {str(e)}")
            return False
    
    async def _monitor_resource_health(self):
        """Monitor resource pool health and adjust reservations"""
        while self.allocator_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for resource_type, pool in self.resource_pools.items():
                    utilization = 1.0 - (pool.available_capacity / pool.total_capacity)
                    
                    # Adjust crisis reservation based on utilization
                    if utilization > 0.9:  # 90% utilization
                        # Increase crisis reservation during high load
                        new_reservation = min(pool.total_capacity * 0.3, pool.total_capacity)
                        pool.reserved_for_crisis = new_reservation
                        logger.info(f"Increased crisis reservation for {resource_type.value} due to high utilization")
                    elif utilization < 0.5:  # 50% utilization
                        # Reduce crisis reservation during low load
                        new_reservation = pool.total_capacity * 0.2
                        pool.reserved_for_crisis = new_reservation
                
            except Exception as e:
                logger.error(f"Resource health monitoring error: {str(e)}")
    
    async def _optimize_allocations(self):
        """Optimize resource allocations based on historical data"""
        while self.allocator_active:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Analyze allocation patterns
                recent_allocations = [
                    alloc for alloc in self.allocation_history
                    if alloc.allocation_started >= datetime.utcnow() - timedelta(hours=1)
                ]
                
                if len(recent_allocations) >= 10:
                    # Optimize gentle allocation rates based on satisfaction
                    self._optimize_gentle_allocation_rates(recent_allocations)
                    
                    # Optimize queue processing based on wait times
                    self._optimize_queue_processing(recent_allocations)
                
            except Exception as e:
                logger.error(f"Allocation optimization error: {str(e)}")
    
    def _optimize_gentle_allocation_rates(self, allocations: List[ResourceAllocation]):
        """Optimize gentle allocation rates based on user satisfaction"""
        try:
            # Group allocations by resource type
            resource_satisfaction = defaultdict(list)
            for alloc in allocations:
                if alloc.user_satisfaction is not None:
                    for resource_type in alloc.allocated_resources:
                        resource_satisfaction[resource_type].append(alloc.user_satisfaction)
            
            # Adjust gentle allocation rates
            for resource_type, satisfaction_scores in resource_satisfaction.items():
                if len(satisfaction_scores) >= 5:
                    avg_satisfaction = statistics.mean(satisfaction_scores)
                    pool = self.resource_pools[resource_type]
                    
                    if avg_satisfaction < 7.0:  # Low satisfaction
                        # Reduce gentle allocation rate for smoother experience
                        pool.gentle_allocation_rate *= 0.9
                    elif avg_satisfaction > 8.5:  # High satisfaction
                        # Can increase gentle allocation rate
                        pool.gentle_allocation_rate *= 1.1
                    
                    # Keep within reasonable bounds
                    max_rate = pool.total_capacity * 0.1
                    min_rate = pool.total_capacity * 0.01
                    pool.gentle_allocation_rate = max(min_rate, min(max_rate, pool.gentle_allocation_rate))
        
        except Exception as e:
            logger.error(f"Failed to optimize gentle allocation rates: {str(e)}")
    
    def _optimize_queue_processing(self, allocations: List[ResourceAllocation]):
        """Optimize queue processing based on allocation patterns"""
        # This would analyze wait times and processing efficiency
        # For now, just log that optimization is happening
        logger.debug("Queue processing optimization completed")
    
    def add_allocation_callback(self, callback: Callable[[ResourceAllocation], None]):
        """Add callback for resource allocations"""
        self.allocation_callbacks.append(callback)
    
    async def get_resource_utilization_report(self) -> Dict[str, Any]:
        """Get comprehensive resource utilization report"""
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'resource_pools': {},
                'queue_status': {},
                'allocation_metrics': {},
                'wellbeing_impact': {}
            }
            
            # Resource pool status
            for resource_type, pool in self.resource_pools.items():
                utilization = 1.0 - (pool.available_capacity / pool.total_capacity)
                report['resource_pools'][resource_type.value] = {
                    'total_capacity': pool.total_capacity,
                    'available_capacity': pool.available_capacity,
                    'utilization_percentage': utilization * 100,
                    'reserved_for_crisis': pool.reserved_for_crisis,
                    'current_allocations': len(pool.current_allocations),
                    'gentle_allocation_rate': pool.gentle_allocation_rate
                }
            
            # Queue status
            for care_queue, queue in self.request_queues.items():
                report['queue_status'][care_queue.value] = {
                    'pending_requests': len(queue),
                    'oldest_request_age_seconds': self._get_oldest_request_age(queue)
                }
            
            # Recent allocation metrics
            recent_allocations = [
                alloc for alloc in self.allocation_history
                if alloc.allocation_started >= datetime.utcnow() - timedelta(hours=1)
            ]
            
            if recent_allocations:
                satisfactions = [a.user_satisfaction for a in recent_allocations if a.user_satisfaction]
                wellbeing_impacts = [a.wellbeing_impact for a in recent_allocations if a.wellbeing_impact]
                
                report['allocation_metrics'] = {
                    'total_allocations_last_hour': len(recent_allocations),
                    'average_satisfaction': statistics.mean(satisfactions) if satisfactions else None,
                    'average_wellbeing_impact': statistics.mean(wellbeing_impacts) if wellbeing_impacts else None,
                    'allocation_efficiency': statistics.mean([
                        a.allocation_efficiency for a in recent_allocations 
                        if a.allocation_efficiency
                    ]) if any(a.allocation_efficiency for a in recent_allocations) else None
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate resource utilization report: {str(e)}")
            return {'error': str(e)}
    
    def _get_oldest_request_age(self, queue: List[ResourceRequest]) -> Optional[float]:
        """Get age of oldest request in queue"""
        if not queue:
            return None
        
        oldest_time = min(req.submitted_at for req in queue)
        return (datetime.utcnow() - oldest_time).total_seconds()
    
    async def update_user_allocation_preferences(self, user_id: str, 
                                               preferences: Dict[str, Any]) -> bool:
        """Update user's resource allocation preferences"""
        try:
            valid_keys = [
                'gentle_loading_preferred', 'max_resource_usage', 'priority_preference',
                'allow_background_processing', 'crisis_resource_sharing_consent',
                'community_resource_sharing', 'performance_vs_gentleness_balance'
            ]
            
            validated_prefs = {
                key: value for key, value in preferences.items()
                if key in valid_keys
            }
            
            self.user_allocation_preferences[user_id] = validated_prefs
            
            logger.info(f"Updated allocation preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user allocation preferences: {str(e)}")
            return False