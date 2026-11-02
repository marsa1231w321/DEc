from collections.abc import Callable

from .events import ResourceChanged, SceneChanged, Subscription
from .scene import Scene
from .validation import collect_validation_issues


class SceneEventConsumers:
    def __init__(
        self,
        scene: Scene,
        on_change: Callable[[int, tuple[str, ...]], None] | None = None,
    ) -> None:
        self.scene = scene
        self.revision = 0
        self.dirty = False
        self.validation_issues = tuple(collect_validation_issues(scene))
        self.validation_problems = tuple(
            issue.message for issue in self.validation_issues
        )
        self._on_change = on_change
        self._subscriptions: tuple[
            Subscription[SceneChanged] | Subscription[ResourceChanged], ...
        ] = (
            scene.events.subscribe(SceneChanged, self._handle_change),
            scene.events.subscribe(ResourceChanged, self._handle_change),
        )

    def _handle_change(self, _event: SceneChanged | ResourceChanged) -> None:
        self.revision += 1
        self.dirty = True
        self.validation_issues = tuple(collect_validation_issues(self.scene))
        self.validation_problems = tuple(
            issue.message for issue in self.validation_issues
        )
        if self._on_change is not None:
            self._on_change(self.revision, self.validation_problems)

    def mark_saved(self) -> None:
        self.dirty = False

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.close()
        self._subscriptions = ()
