from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

STRICT_UNIT_INTERVAL_EPSILON = 0.01


def strict_unit_interval(value: float) -> float:
    """Keep validator-visible numeric outputs strictly inside (0, 1).

    The margin is intentionally wide enough to stay inside the interval even
    after reward values are formatted to two decimal places in inference logs.
    """
    return max(STRICT_UNIT_INTERVAL_EPSILON, min(1.0 - STRICT_UNIT_INTERVAL_EPSILON, value))

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Lightweight fallback so the project can still be imported before
    # openenv-core is installed locally.
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        reward: float = 0.0
        done: bool = False
        metadata: Dict[str, object] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: str = ""
        step_count: int = 0


class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class SignalPhase(str, Enum):
    NS_GREEN = "ns_green"
    EW_GREEN = "ew_green"
    ALL_RED = "all_red"


class VehicleType(str, Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"


class TaskId(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TrafficCommand(str, Enum):
    SET_NS_GREEN = "set_ns_green"
    SET_EW_GREEN = "set_ew_green"
    HOLD_CURRENT_PHASE = "hold_current_phase"
    SET_ALL_RED = "set_all_red"


class TrafficControlAction(Action):
    command: TrafficCommand = Field(
        ...,
        description="Traffic signal command for the current timestep.",
    )


class VehicleSpawn(BaseModel):
    arrival_step: int = Field(..., ge=0)
    direction: Direction
    vehicle_type: VehicleType = VehicleType.NORMAL
    count: int = Field(1, ge=1, description="How many vehicles arrive together.")


class VehicleRecord(BaseModel):
    vehicle_id: str
    direction: Direction
    vehicle_type: VehicleType = VehicleType.NORMAL
    wait_time: int = Field(0, ge=0)
    arrival_step: int = Field(0, ge=0)


class TrafficMetrics(BaseModel):
    total_wait_time: float = 0.0
    average_wait_time: float = 0.0
    max_wait_time: int = 0
    total_queue_length: int = 0
    total_vehicles_passed: int = 0
    vehicles_passed_by_direction: Dict[str, int] = Field(default_factory=dict)
    emergency_vehicle_active: bool = False
    emergency_vehicle_direction: Optional[Direction] = None
    emergency_wait_time: int = 0
    total_emergency_wait_time: int = 0


class TaskScenario(BaseModel):
    task_id: TaskId
    name: str
    description: str
    horizon_steps: int = Field(..., gt=0)
    initial_phase: SignalPhase = SignalPhase.ALL_RED
    pass_capacity_per_lane: int = Field(
        1,
        ge=1,
        description="How many vehicles can pass from one permitted lane in a step.",
    )
    spawn_schedule: List[VehicleSpawn] = Field(default_factory=list)
    grader_weights: Dict[str, float] = Field(default_factory=dict)


class TrafficControlObservation(Observation):
    current_phase: SignalPhase = SignalPhase.ALL_RED
    current_timestep: int = 0
    steps_remaining: int = 0
    time_since_last_phase_change: int = 0

    queue_north: int = 0
    queue_south: int = 0
    queue_east: int = 0
    queue_west: int = 0

    avg_wait_north: float = 0.0
    avg_wait_south: float = 0.0
    avg_wait_east: float = 0.0
    avg_wait_west: float = 0.0

    emergency_present: bool = False
    emergency_direction: Optional[Direction] = None
    vehicles_passed_total: int = 0
    final_score: Optional[float] = None
    status_message: str = ""


class TrafficControlState(State):
    task_id: TaskId = TaskId.EASY
    task_name: str = ""
    done: bool = False
    current_phase: SignalPhase = SignalPhase.ALL_RED
    last_phase_change_step: int = 0
    cumulative_reward: float = 0.0
    total_vehicles_passed: int = 0
    switch_count: int = 0
    lane_queues: Dict[str, List[VehicleRecord]] = Field(default_factory=dict)
    pending_spawns: List[VehicleSpawn] = Field(default_factory=list)
    metrics: TrafficMetrics = Field(default_factory=TrafficMetrics)
    final_score: Optional[float] = None
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
