import threading

from triton_anchor.backends import BackendPluginConflictError
from triton_anchor.backends import BackendPluginError
from triton_anchor.backends import BackendPluginLifecycleError
from triton_anchor.backends import BackendPluginSelectionError

from ..backends import activate_backend
from ..backends import DriverBase
from ..backends import get_backend
from ..backends import get_driver_backends
from ..backends import register_backend_reset_hook


def _create_driver():
    candidates = get_driver_backends()
    actives = []
    for backend in candidates:
        driver_cls = backend.driver
        try:
            is_active = getattr(driver_cls, "is_active", None)
            if not callable(is_active):
                raise TypeError("is_active is not callable")
            if is_active():
                actives.append(backend)
        except BackendPluginError:
            raise
        except Exception as exc:
            raise BackendPluginLifecycleError(
                f"Backend driver active probe failed: {exc}",
                plugin_id=backend.plugin_id,
                entry_point=backend.entry_point_name,
                field="driver_cls.is_active",
                expected="a successful boolean active probe",
                actual=f"<error: {exc}>",
                remediation=(
                    "Fix driver_cls.is_active() so runtime selection can "
                    "probe the Registry winner without side effects."
                ),
            ) from exc

    if not actives:
        raise BackendPluginSelectionError(
            "No Registry-selected backend driver is active",
            field="driver_cls.is_active",
            expected="one active backend driver",
            actual="0",
            remediation=(
                "Install or select a backend whose driver reports active, "
                "or set TRITON_ANCHOR_BACKEND explicitly."
            ),
        )
    if len(actives) > 1:
        driver_classes = [backend.driver for backend in actives]
        raise BackendPluginConflictError(
            f"{len(actives)} Registry-selected backend drivers are active",
            field="driver_cls.is_active",
            expected="one active backend driver",
            actual=", ".join(
                getattr(driver_cls, "__name__", repr(driver_cls))
                for driver_cls in driver_classes
            ),
            remediation=(
                "Use TRITON_ANCHOR_BACKEND or hardware configuration so only "
                "one selected driver reports active."
            ),
        )

    backend = actives[0]
    try:
        active_driver = backend.driver()
    except BackendPluginError:
        raise
    except Exception as exc:
        raise BackendPluginLifecycleError(
            f"Backend driver construction failed: {exc}",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field="driver_cls.__init__",
            expected="a successfully constructed driver",
            actual=f"<error: {exc}>",
            remediation=(
                "Fix the selected driver constructor and release partial "
                "runtime resources before retrying."
            ),
        ) from exc
    try:
        target = active_driver.get_current_target()
    except BackendPluginError:
        raise
    except Exception as exc:
        raise BackendPluginLifecycleError(
            f"Backend driver target lookup failed: {exc}",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field="driver_cls.get_current_target",
            expected="a readable current target",
            actual=f"<error: {exc}>",
            remediation=(
                "Fix get_current_target() so compiler and runtime can verify "
                "the same Registry record."
            ),
        ) from exc
    activate_backend(backend, target=target)
    return active_driver


class LazyProxy:

    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._obj = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._initializing = False
        self._generation = 0

    def _initialize_obj(self):
        while True:
            with self._condition:
                if self._obj is not None:
                    return self._obj
                if self._initializing:
                    self._condition.wait()
                    continue
                generation = self._generation
                self._initializing = True

            # Backend construction invokes plugin code.  Run it without the
            # proxy lock so Registry.reset() can invalidate this attempt
            # without waiting on a plugin-owned lock.
            try:
                obj = self._init_fn()
            except BaseException:
                with self._condition:
                    self._initializing = False
                    self._condition.notify_all()
                raise

            with self._condition:
                self._initializing = False
                if (
                    self._generation == generation
                    and self._obj is None
                ):
                    self._obj = obj
                    self._condition.notify_all()
                    return obj
                self._condition.notify_all()

    def reset(self):
        with self._condition:
            self._generation += 1
            self._obj = None
            self._condition.notify_all()

    def __getattr__(self, name):
        return getattr(self._initialize_obj(), name)

    def __setattr__(self, name, value):
        if name in [
            "_init_fn",
            "_obj",
            "_lock",
            "_condition",
            "_initializing",
            "_generation",
        ]:
            super().__setattr__(name, value)
        else:
            setattr(self._initialize_obj(), name, value)

    def __delattr__(self, name):
        delattr(self._initialize_obj(), name)

    def __repr__(self):
        with self._lock:
            obj = self._obj
            init_fn = self._init_fn
        if obj is None:
            return (
                f"<{self.__class__.__name__} for "
                f"{init_fn} not yet initialized>"
            )
        return repr(obj)

    def __str__(self):
        return str(self._initialize_obj())


class DriverConfig:

    def __init__(self):
        self._lock = threading.RLock()
        self._reset_epoch = 0
        self.default = LazyProxy(_create_driver)
        self.active = self.default

    def _check_reset_epoch(self, expected, backend=None):
        with self._lock:
            actual = self._reset_epoch
        if actual != expected:
            raise BackendPluginLifecycleError(
                "Explicit backend activation was invalidated by "
                "Registry reset",
                plugin_id=getattr(backend, "plugin_id", None),
                entry_point=getattr(
                    backend,
                    "entry_point_name",
                    None,
                ),
                field="registry reset",
                expected=str(expected),
                actual=str(actual),
                remediation=(
                    "Retry driver.set_active() after Registry reset "
                    "completes."
                ),
            )

    def set_active(self, driver: DriverBase):
        with self._lock:
            if driver is self.default:
                self.active = self.default
                return
            reset_epoch = self._reset_epoch

        # Plugin callbacks and Registry operations stay outside the local
        # config lock.  Registry.reset() invokes this object's reset hook, so
        # holding the lock here would permit a plugin-lock/config-lock cycle.
        try:
            get_current_target = getattr(
                driver,
                "get_current_target",
                None,
            )
            if not callable(get_current_target):
                raise TypeError("get_current_target is not callable")
            target = get_current_target()
        except BackendPluginError:
            raise
        except Exception as exc:
            raise BackendPluginLifecycleError(
                f"Explicit backend driver target lookup failed: {exc}",
                field="driver.get_current_target",
                expected="a readable current target",
                actual=f"<error: {exc}>",
                remediation=(
                    "Pass a backend driver whose get_current_target() "
                    "identifies its compiler target."
                ),
            ) from exc

        self._check_reset_epoch(reset_epoch)
        backend = get_backend(target)
        self._check_reset_epoch(reset_epoch, backend)
        driver_cls = backend.driver
        if (
            not isinstance(driver_cls, type)
            or not isinstance(driver, driver_cls)
        ):
            raise BackendPluginSelectionError(
                "Explicit active driver does not match the "
                "Registry-selected compiler/driver pair",
                plugin_id=backend.plugin_id,
                entry_point=backend.entry_point_name,
                field="compiler_cls,driver_cls",
                expected=getattr(
                    driver_cls,
                    "__name__",
                    repr(driver_cls),
                ),
                actual=type(driver).__name__,
                remediation=(
                    "Activate the driver from the backend selected for "
                    "this target, or remove the overlapping Manifest "
                    "before using a manual Legacy pair."
                ),
            )
        activate_backend(backend, target=target)
        self._check_reset_epoch(reset_epoch, backend)
        with self._lock:
            if self._reset_epoch != reset_epoch:
                actual = self._reset_epoch
            else:
                self.active = driver
                return
        raise BackendPluginLifecycleError(
            "Explicit backend activation was invalidated by Registry reset",
            plugin_id=backend.plugin_id,
            entry_point=backend.entry_point_name,
            field="registry reset",
            expected=str(reset_epoch),
            actual=str(actual),
            remediation=(
                "Retry driver.set_active() after Registry reset completes."
            ),
        )

    def reset_active(self):
        with self._lock:
            self.active = self.default

    def _reset_registry_state(self):
        with self._lock:
            self._reset_epoch += 1
            self.default.reset()
            self.active = self.default


driver = DriverConfig()
register_backend_reset_hook(driver._reset_registry_state)
