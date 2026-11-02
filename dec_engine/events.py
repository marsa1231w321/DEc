from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Literal, TypeVar, cast


class Event:
    pass


@dataclass(frozen=True)
class SceneChanged(Event):
    operation: Literal["added", "removed", "reparented", "updated"]
    node_id: str | None
    node_kind: str
    parent_id: str | None = None
    previous_parent_id: str | None = None
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceChanged(Event):
    resource_kind: Literal["model", "texture", "audio", "asset_pack"]
    action: Literal["loaded", "assigned", "removed"]
    resource_id: str
    node_id: str | None = None


@dataclass(frozen=True)
class SelectionChanged(Event):
    selected_ids: tuple[str, ...]
    active_id: str | None
    source: Literal["object_list", "light_list", "none"]


EventT = TypeVar("EventT", bound=Event)


class Subscription(Generic[EventT]):
    def __init__(self, bus: EventBus, event_type: type[EventT], token: int):
        self._bus = bus
        self._event_type = event_type
        self._token = token
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def close(self) -> None:
        if self._active:
            self._bus._unsubscribe(self._event_type, self._token)
            self._active = False


class EventBus:
    def __init__(self):
        self._listeners: dict[type[Event], dict[int, Callable[[Event], None]]] = {}
        self._next_token = 0

    def subscribe(
        self,
        event_type: type[EventT],
        listener: Callable[[EventT], None],
    ) -> Subscription[EventT]:
        token = self._next_token
        self._next_token += 1
        listeners = self._listeners.setdefault(event_type, {})
        listeners[token] = cast(Callable[[Event], None], listener)
        return Subscription(self, event_type, token)

    def publish(self, event: Event) -> None:
        for listener in tuple(self._listeners.get(type(event), {}).values()):
            listener(event)

    def _unsubscribe(self, event_type: type[Event], token: int) -> None:
        listeners = self._listeners.get(event_type)
        if listeners is None:
            return
        listeners.pop(token, None)
        if not listeners:
            del self._listeners[event_type]

    def listener_count(self, event_type: type[Event] | None = None) -> int:
        if event_type is not None:
            return len(self._listeners.get(event_type, {}))
        return sum(len(listeners) for listeners in self._listeners.values())

    def __getstate__(self) -> dict[str, int]:
        return {"_next_token": 0}

    def __setstate__(self, state: dict[str, int]) -> None:
        self._listeners = {}
        self._next_token = state.get("_next_token", 0)
