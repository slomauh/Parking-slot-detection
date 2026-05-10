from __future__ import annotations

from src.detection.schemas import OccupancyDecision, ParkingSlot, SlotState


class TemporalStateManager:
    def __init__(self, config: dict) -> None:
        self.occupied_confirm_frames = int(config.get("occupied_confirm_frames", 5))
        self.free_confirm_frames = int(config.get("free_confirm_frames", 10))
        self.states: dict[int, SlotState] = {}

    def update(
        self,
        frame_idx: int,
        slots: list[ParkingSlot],
        decisions: dict[int, OccupancyDecision],
    ) -> list[SlotState]:
        active_slot_ids = {slot.slot_id for slot in slots}
        for slot in slots:
            state = self.states.setdefault(slot.slot_id, SlotState(slot_id=slot.slot_id))
            state.slot = slot
            state.last_seen_frame = frame_idx
            decision = decisions.get(
                slot.slot_id,
                OccupancyDecision(slot.slot_id, "unknown", 0.0, "missing"),
            )
            state.assigned_track_id = decision.assigned_track_id
            state.confidence = decision.confidence
            state.source = decision.source

            if decision.status == "occupied":
                state.occupied_counter += 1
                state.free_counter = 0
                if state.occupied_counter >= self.occupied_confirm_frames:
                    state.status = "occupied"
                else:
                    state.status = "potentially_occupied"
            elif decision.status == "free":
                state.free_counter += 1
                state.occupied_counter = 0
                if state.free_counter >= self.free_confirm_frames:
                    state.status = "free"
                elif state.status == "unknown":
                    state.status = "unknown"
            else:
                state.occupied_counter = 0
                state.free_counter = 0
                state.status = "unknown"

        return [state for slot_id, state in self.states.items() if slot_id in active_slot_ids]
