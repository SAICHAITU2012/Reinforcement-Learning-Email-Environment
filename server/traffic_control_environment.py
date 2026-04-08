from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        def __init__(self) -> None:
            pass


try:
    from ..models import (
        Direction,
        SignalPhase,
        TaskId,
        TaskScenario,
        TrafficControlAction,
        TrafficControlObservation,
        TrafficControlState,
        TrafficMetrics,
        VehicleRecord,
        VehicleSpawn,
        VehicleType,
        strict_unit_interval,
    )
    from ..task_bank import DEFAULT_TASK_ID, get_task
except ImportError:
    from models import (
        Direction,
        SignalPhase,
        TaskId,
        TaskScenario,
        TrafficControlAction,
        TrafficControlObservation,
        TrafficControlState,
        TrafficMetrics,
        VehicleRecord,
        VehicleSpawn,
        VehicleType,
        strict_unit_interval,
    )
    from task_bank import DEFAULT_TASK_ID, get_task


class TrafficControlEnvironment(Environment):
    """
    A simple deterministic 4-way traffic-control environment.

    The environment uses a simple deterministic simulator:
    - the action chooses or holds the traffic phase
    - permitted lanes allow vehicles through
    - queued vehicles accumulate waiting time
    - future arrivals spawn on fixed timesteps
    - step reward balances throughput, delay, and emergency priority
    """

    def __init__(self, default_task_id: TaskId = DEFAULT_TASK_ID):
        super().__init__()
        self.default_task_id = default_task_id
        self._task: TaskScenario = get_task(default_task_id)
        self._state = self._build_initial_state(self._task, episode_id=str(uuid4()))

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TrafficControlObservation:
        """
        Reset the environment into one of the fixed task scenarios.

        Parameters other than ``task_id`` are accepted for compatibility but
        ignored in this deterministic Phase 2 implementation.
        """
        selected_task_id = task_id or kwargs.get("scenario_id") or kwargs.get("difficulty")
        resolved_task_id = TaskId(selected_task_id) if selected_task_id else self.default_task_id

        self._task = get_task(resolved_task_id)
        self._state = self._build_initial_state(
            task=self._task,
            episode_id=episode_id or str(uuid4()),
        )
        return self._build_observation(
            reward=self._reported_reward(0.0),
            done=False,
            status_message=(
                f"Loaded task '{self._task.name}'. "
                "The intersection is ready for control."
            ),
        )

    def step(self, action: TrafficControlAction) -> TrafficControlObservation:
        if self._state.done:
            return self._build_observation(
                reward=self._reported_reward(0.0),
                done=True,
                status_message="Episode is already complete. Call reset() to start a new run.",
            )

        phase_changed = False
        invalid_action = False
        status_parts: List[str] = []
        steps_since_previous_phase_change = (
            self._state.step_count - self._state.last_phase_change_step
        )

        command = self._extract_command(action)
        if command is None or not self._is_valid_command(command):
            invalid_action = True
            status_parts.append(
                "Invalid action received. Falling back to the current phase for this step."
            )
        else:
            phase_changed, command_message = self._apply_command(command)
            status_parts.append(command_message)

        passed_vehicles = self._move_vehicles_for_current_phase()
        self._increment_wait_times()
        spawned_vehicles = self._spawn_vehicles_for_step(self._state.step_count + 1)

        self._state.step_count += 1
        self._state.metrics = self._compute_metrics(self._state.lane_queues)

        reward = self._compute_step_reward(
            passed_vehicles=passed_vehicles,
            phase_changed=phase_changed,
            invalid_action=invalid_action,
            steps_since_previous_phase_change=steps_since_previous_phase_change,
        )
        self._state.cumulative_reward += reward
        reported_reward = self._reported_reward(reward)

        if passed_vehicles:
            passed_summary = ", ".join(
                f"{key}={value}" for key, value in sorted(passed_vehicles.items()) if value > 0
            )
            status_parts.append(f"Vehicles cleared this step: {passed_summary}.")
        else:
            status_parts.append("No vehicles cleared this step.")

        if spawned_vehicles:
            spawned_summary = ", ".join(
                f"{key}={value}" for key, value in sorted(spawned_vehicles.items()) if value > 0
            )
            status_parts.append(f"New arrivals for next timestep: {spawned_summary}.")

        done = self._should_end_episode()
        self._state.done = done

        if done:
            score_breakdown = self._grade_episode()
            self._state.final_score = score_breakdown["final_score"]
            self._state.score_breakdown = score_breakdown
            status_parts.append(
                f"Episode complete. Final score: {self._state.final_score:.3f}."
            )

        return self._build_observation(
            reward=reported_reward,
            done=done,
            status_message=" ".join(status_parts),
        )

    @property
    def state(self) -> TrafficControlState:
        return self._state

    def _build_initial_state(
        self,
        task: TaskScenario,
        episode_id: str,
    ) -> TrafficControlState:
        lane_queues: Dict[str, List[VehicleRecord]] = {
            direction.value: [] for direction in Direction
        }

        pending_spawns: List[VehicleSpawn] = []
        for spawn in task.spawn_schedule:
            if spawn.arrival_step == 0:
                lane_queues[spawn.direction.value].extend(self._expand_spawn(spawn))
            else:
                pending_spawns.append(spawn.model_copy(deep=True))

        metrics = self._compute_metrics(lane_queues)
        return TrafficControlState(
            episode_id=episode_id,
            step_count=0,
            task_id=task.task_id,
            task_name=task.name,
            done=False,
            current_phase=task.initial_phase,
            last_phase_change_step=0,
            cumulative_reward=0.0,
            total_vehicles_passed=0,
            switch_count=0,
            lane_queues=lane_queues,
            pending_spawns=pending_spawns,
            metrics=metrics,
            final_score=None,
            score_breakdown={},
        )

    def _expand_spawn(self, spawn: VehicleSpawn) -> List[VehicleRecord]:
        vehicles: List[VehicleRecord] = []
        for offset in range(spawn.count):
            vehicles.append(
                VehicleRecord(
                    vehicle_id=(
                        f"{spawn.vehicle_type.value}-"
                        f"{spawn.direction.value}-"
                        f"{spawn.arrival_step}-"
                        f"{offset}"
                    ),
                    direction=spawn.direction,
                    vehicle_type=spawn.vehicle_type,
                    wait_time=0,
                    arrival_step=spawn.arrival_step,
                )
            )
        return vehicles

    def _build_observation(
        self,
        reward: float,
        done: bool,
        status_message: str,
    ) -> TrafficControlObservation:
        lane_queues = self._state.lane_queues
        wait_map = self._average_wait_by_direction(lane_queues)
        metrics = self._state.metrics

        return TrafficControlObservation(
            current_phase=self._state.current_phase,
            current_timestep=self._state.step_count,
            steps_remaining=max(self._task.horizon_steps - self._state.step_count, 0),
            time_since_last_phase_change=(
                self._state.step_count - self._state.last_phase_change_step
            ),
            queue_north=len(lane_queues[Direction.NORTH.value]),
            queue_south=len(lane_queues[Direction.SOUTH.value]),
            queue_east=len(lane_queues[Direction.EAST.value]),
            queue_west=len(lane_queues[Direction.WEST.value]),
            avg_wait_north=wait_map[Direction.NORTH.value],
            avg_wait_south=wait_map[Direction.SOUTH.value],
            avg_wait_east=wait_map[Direction.EAST.value],
            avg_wait_west=wait_map[Direction.WEST.value],
            emergency_present=metrics.emergency_vehicle_active,
            emergency_direction=metrics.emergency_vehicle_direction,
            vehicles_passed_total=self._state.total_vehicles_passed,
            final_score=self._state.final_score,
            status_message=status_message,
            reward=reward,
            done=done,
            metadata={
                "task_id": self._state.task_id.value,
                "switch_count": self._state.switch_count,
                "total_queue_length": metrics.total_queue_length,
                "average_wait_time": metrics.average_wait_time,
                "final_score": self._state.final_score,
                "score_breakdown": self._state.score_breakdown,
            },
        )

    def _average_wait_by_direction(
        self,
        lane_queues: Dict[str, List[VehicleRecord]],
    ) -> Dict[str, float]:
        averages: Dict[str, float] = {}
        for direction in Direction:
            vehicles = lane_queues[direction.value]
            if not vehicles:
                averages[direction.value] = 0.0
                continue
            averages[direction.value] = sum(v.wait_time for v in vehicles) / len(vehicles)
        return averages

    def _compute_metrics(
        self,
        lane_queues: Dict[str, List[VehicleRecord]],
    ) -> TrafficMetrics:
        all_vehicles = [
            vehicle
            for vehicles in lane_queues.values()
            for vehicle in vehicles
        ]
        total_wait_time = float(sum(vehicle.wait_time for vehicle in all_vehicles))
        total_queue_length = len(all_vehicles)
        max_wait_time = max((vehicle.wait_time for vehicle in all_vehicles), default=0)
        average_wait_time = (
            total_wait_time / total_queue_length if total_queue_length else 0.0
        )

        emergency_vehicle = next(
            (vehicle for vehicle in all_vehicles if vehicle.vehicle_type == VehicleType.EMERGENCY),
            None,
        )

        return TrafficMetrics(
            total_wait_time=total_wait_time,
            average_wait_time=average_wait_time,
            max_wait_time=max_wait_time,
            total_queue_length=total_queue_length,
            total_vehicles_passed=self._state.total_vehicles_passed if hasattr(self, "_state") else 0,
            vehicles_passed_by_direction=(
                self._state.metrics.vehicles_passed_by_direction.copy()
                if hasattr(self, "_state")
                else {direction.value: 0 for direction in Direction}
            ),
            emergency_vehicle_active=emergency_vehicle is not None,
            emergency_vehicle_direction=(
                emergency_vehicle.direction if emergency_vehicle is not None else None
            ),
            emergency_wait_time=emergency_vehicle.wait_time if emergency_vehicle else 0,
            total_emergency_wait_time=(
                self._state.metrics.total_emergency_wait_time if hasattr(self, "_state") else 0
            ),
        )

    def _extract_command(self, action: Any) -> Optional[str]:
        if isinstance(action, TrafficControlAction):
            return action.command.value
        if hasattr(action, "command"):
            raw_command = getattr(action, "command")
            if isinstance(raw_command, str):
                return raw_command
            if hasattr(raw_command, "value"):
                return raw_command.value
        return None

    def _is_valid_command(self, command: str) -> bool:
        return command in {
            "set_ns_green",
            "set_ew_green",
            "hold_current_phase",
            "set_all_red",
        }

    def _apply_command(self, command: str) -> tuple[bool, str]:
        target_phase = self._state.current_phase
        if command == "set_ns_green":
            target_phase = SignalPhase.NS_GREEN
        elif command == "set_ew_green":
            target_phase = SignalPhase.EW_GREEN
        elif command == "set_all_red":
            target_phase = SignalPhase.ALL_RED
        elif command == "hold_current_phase":
            return False, f"Held phase {self._state.current_phase.value}."
        else:
            return False, f"Unknown command '{command}'. Using current phase."

        if target_phase != self._state.current_phase:
            self._state.current_phase = target_phase
            self._state.last_phase_change_step = self._state.step_count
            self._state.switch_count += 1
            return True, f"Switched phase to {target_phase.value}."

        return False, f"Phase already {target_phase.value}; no switch needed."

    def _allowed_directions(self) -> List[Direction]:
        if self._state.current_phase == SignalPhase.NS_GREEN:
            return [Direction.NORTH, Direction.SOUTH]
        if self._state.current_phase == SignalPhase.EW_GREEN:
            return [Direction.EAST, Direction.WEST]
        return []

    def _move_vehicles_for_current_phase(self) -> Dict[str, int]:
        passed_counts = {
            "normal": 0,
            "emergency": 0,
        }
        for direction in self._allowed_directions():
            queue = self._state.lane_queues[direction.value]
            for _ in range(self._task.pass_capacity_per_lane):
                vehicle = self._pop_next_vehicle(queue)
                if vehicle is None:
                    break
                self._state.total_vehicles_passed += 1
                self._state.metrics.vehicles_passed_by_direction[direction.value] += 1
                passed_counts[vehicle.vehicle_type.value] += 1
        return passed_counts

    def _pop_next_vehicle(self, queue: List[VehicleRecord]) -> Optional[VehicleRecord]:
        if not queue:
            return None

        emergency_index = next(
            (
                index
                for index, vehicle in enumerate(queue)
                if vehicle.vehicle_type == VehicleType.EMERGENCY
            ),
            None,
        )
        if emergency_index is None:
            return queue.pop(0)
        return queue.pop(emergency_index)

    def _increment_wait_times(self) -> None:
        emergency_waiting_count = 0
        for vehicles in self._state.lane_queues.values():
            for vehicle in vehicles:
                vehicle.wait_time += 1
                if vehicle.vehicle_type == VehicleType.EMERGENCY:
                    emergency_waiting_count += 1
        self._state.metrics.total_emergency_wait_time += emergency_waiting_count

    def _spawn_vehicles_for_step(self, arrival_step: int) -> Dict[str, int]:
        spawned_counts = {
            direction.value: 0 for direction in Direction
        }
        remaining_spawns: List[VehicleSpawn] = []

        for spawn in self._state.pending_spawns:
            if spawn.arrival_step == arrival_step:
                vehicles = self._expand_spawn(spawn)
                self._state.lane_queues[spawn.direction.value].extend(vehicles)
                spawned_counts[spawn.direction.value] += len(vehicles)
            else:
                remaining_spawns.append(spawn)

        self._state.pending_spawns = remaining_spawns
        return spawned_counts

    def _compute_step_reward(
        self,
        passed_vehicles: Dict[str, int],
        phase_changed: bool,
        invalid_action: bool,
        steps_since_previous_phase_change: int,
    ) -> float:
        metrics = self._state.metrics

        throughput_reward = (
            passed_vehicles["normal"] * 1.0 + passed_vehicles["emergency"] * 3.0
        )
        queue_penalty = metrics.total_queue_length * self._queue_penalty_scale()
        wait_penalty = metrics.average_wait_time * self._wait_penalty_scale()
        emergency_penalty = self._emergency_delay_penalty(metrics.emergency_wait_time)
        switch_penalty = self._switch_penalty(phase_changed, steps_since_previous_phase_change)
        all_red_penalty = self._all_red_penalty()
        imbalance_penalty = self._imbalance_penalty()
        invalid_penalty = 2.0 if invalid_action else 0.0

        return (
            throughput_reward
            - queue_penalty
            - wait_penalty
            - emergency_penalty
            - switch_penalty
            - all_red_penalty
            - imbalance_penalty
            - invalid_penalty
        )

    def _should_end_episode(self) -> bool:
        if self._state.step_count >= self._task.horizon_steps:
            return True

        no_pending_spawns = len(self._state.pending_spawns) == 0
        all_queues_empty = all(
            len(vehicles) == 0 for vehicles in self._state.lane_queues.values()
        )
        return no_pending_spawns and all_queues_empty

    def _grade_episode(self) -> Dict[str, float]:
        metrics = self._state.metrics
        weights = self._task.grader_weights
        total_scheduled = self._total_scheduled_vehicles()
        scheduled_by_direction = self._scheduled_vehicles_by_direction()
        emergency_total = self._total_scheduled_emergency_vehicles()

        throughput_score = self._clamp(
            metrics.total_vehicles_passed / total_scheduled if total_scheduled else 1.0
        )

        acceptable_average_wait = self._acceptable_average_wait()
        average_wait_score = self._clamp(
            1.0 - (metrics.average_wait_time / acceptable_average_wait)
        )

        stability_score = self._compute_stability_score()

        emergency_handling_score = 1.0
        if emergency_total > 0:
            emergency_pass_rate = self._count_emergency_passed() / emergency_total
            emergency_wait_quality = self._clamp(
                1.0
                - (
                    metrics.total_emergency_wait_time
                    / max(self._emergency_wait_budget() * emergency_total, 1.0)
                )
            )
            emergency_clearance_bonus = 1.0 if not metrics.emergency_vehicle_active else 0.0
            emergency_handling_score = (
                emergency_pass_rate * 0.45
                + emergency_wait_quality * 0.4
                + emergency_clearance_bonus * 0.15
            )

        fairness_score = self._compute_fairness_score(scheduled_by_direction)

        component_scores = {
            "throughput": throughput_score,
            "average_wait": average_wait_score,
            "stability": stability_score,
            "emergency_handling": emergency_handling_score,
            "fairness": fairness_score,
        }

        weighted_total = 0.0
        weight_sum = 0.0
        for metric_name, weight in weights.items():
            weighted_total += component_scores.get(metric_name, 0.0) * weight
            weight_sum += weight

        final_score = self._clamp(weighted_total / weight_sum if weight_sum else 0.0)
        return {
            **{
                metric_name: self._strict_score(score)
                for metric_name, score in component_scores.items()
            },
            "final_score": self._strict_score(final_score),
        }

    def _count_emergency_passed(self) -> int:
        total_emergency = self._total_scheduled_emergency_vehicles()
        still_waiting = sum(
            1
            for vehicles in self._state.lane_queues.values()
            for vehicle in vehicles
            if vehicle.vehicle_type == VehicleType.EMERGENCY
        )
        remaining_in_spawns = sum(
            spawn.count
            for spawn in self._state.pending_spawns
            if spawn.vehicle_type == VehicleType.EMERGENCY
        )
        return max(total_emergency - still_waiting - remaining_in_spawns, 0)

    def _compute_fairness_score(self, scheduled_by_direction: Dict[str, int]) -> float:
        direction_ratios: List[float] = []
        wait_pressures: List[float] = []
        for direction, scheduled_count in scheduled_by_direction.items():
            if scheduled_count <= 0:
                continue
            passed = self._state.metrics.vehicles_passed_by_direction.get(direction, 0)
            direction_ratios.append(passed / scheduled_count)
            queued = self._state.lane_queues[direction]
            if queued:
                wait_pressures.append(
                    sum(vehicle.wait_time for vehicle in queued) / len(queued)
                )
            else:
                wait_pressures.append(0.0)

        if not direction_ratios:
            return 1.0

        service_balance = max(direction_ratios) - min(direction_ratios)
        wait_balance = max(wait_pressures) - min(wait_pressures) if wait_pressures else 0.0
        normalized_wait_gap = wait_balance / max(self._task.horizon_steps / 2.0, 1.0)

        return self._clamp(1.0 - (service_balance * 0.65) - (normalized_wait_gap * 0.35))

    def _total_scheduled_vehicles(self) -> int:
        return sum(spawn.count for spawn in self._task.spawn_schedule)

    def _total_scheduled_emergency_vehicles(self) -> int:
        return sum(
            spawn.count
            for spawn in self._task.spawn_schedule
            if spawn.vehicle_type == VehicleType.EMERGENCY
        )

    def _scheduled_vehicles_by_direction(self) -> Dict[str, int]:
        counts = {direction.value: 0 for direction in Direction}
        for spawn in self._task.spawn_schedule:
            counts[spawn.direction.value] += spawn.count
        return counts

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(value, high))

    def _strict_score(self, value: float) -> float:
        return strict_unit_interval(value)

    def _reported_reward(self, value: float) -> float:
        return strict_unit_interval(value)

    def _queue_penalty_scale(self) -> float:
        if self._task.task_id == TaskId.HARD:
            return 0.2
        if self._task.task_id == TaskId.MEDIUM:
            return 0.17
        return 0.15

    def _wait_penalty_scale(self) -> float:
        if self._task.task_id == TaskId.HARD:
            return 0.14
        if self._task.task_id == TaskId.MEDIUM:
            return 0.12
        return 0.1

    def _emergency_delay_penalty(self, emergency_wait_time: int) -> float:
        if emergency_wait_time <= 0:
            return 0.0

        base_multiplier = 0.8 if self._task.task_id != TaskId.HARD else 1.15
        escalating_penalty = emergency_wait_time * base_multiplier
        if self._task.task_id == TaskId.HARD and emergency_wait_time >= 4:
            escalating_penalty += 1.0 + ((emergency_wait_time - 3) * 0.75)
        return escalating_penalty

    def _switch_penalty(self, phase_changed: bool, steps_since_previous_phase_change: int) -> float:
        if not phase_changed:
            return 0.0

        base_penalty = 0.25 if self._task.task_id != TaskId.HARD else 0.35
        if steps_since_previous_phase_change <= 1:
            return base_penalty + 0.35
        return base_penalty

    def _all_red_penalty(self) -> float:
        if self._state.current_phase != SignalPhase.ALL_RED:
            return 0.0
        if self._state.metrics.total_queue_length == 0:
            return 0.0
        return 0.2 if self._task.task_id != TaskId.HARD else 0.35

    def _imbalance_penalty(self) -> float:
        waits = self._average_wait_by_direction(self._state.lane_queues)
        active_waits = [value for value in waits.values() if value > 0.0]
        if len(active_waits) < 2:
            return 0.0
        imbalance = max(active_waits) - min(active_waits)
        scale = 0.03 if self._task.task_id != TaskId.HARD else 0.05
        return imbalance * scale

    def _acceptable_average_wait(self) -> float:
        if self._task.task_id == TaskId.HARD:
            return max(self._task.horizon_steps / 4.5, 1.0)
        if self._task.task_id == TaskId.MEDIUM:
            return max(self._task.horizon_steps / 3.8, 1.0)
        return max(self._task.horizon_steps / 3.0, 1.0)

    def _compute_stability_score(self) -> float:
        allowed_switches = {
            TaskId.EASY: 3,
            TaskId.MEDIUM: 4,
            TaskId.HARD: 6,
        }[self._task.task_id]
        overswitch_penalty = max(self._state.switch_count - allowed_switches, 0)
        all_red_penalty = 1 if self._state.current_phase == SignalPhase.ALL_RED else 0
        return self._clamp(
            1.0
            - (overswitch_penalty / max(self._task.horizon_steps / 3.0, 1.0))
            - (all_red_penalty * 0.1)
        )

    def _emergency_wait_budget(self) -> float:
        if self._task.task_id == TaskId.HARD:
            return max(self._task.horizon_steps / 5.0, 2.0)
        if self._task.task_id == TaskId.MEDIUM:
            return max(self._task.horizon_steps / 4.0, 2.0)
        return max(self._task.horizon_steps / 3.0, 2.0)
