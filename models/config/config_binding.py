"""Typed read/write accessor for a key path in an in-memory config dict.

Intended for UI controls that need to read initial values from a config dict,
write back on user interaction, and optionally invoke side effects (e.g. pushing
values to a running camera or model object).  The config dict is mutated in place;
persisting to disk is the caller's responsibility.
"""


class ConfigBinding:
    """Typed read/write accessor for a key path in the in-memory config dict.

    Usage::

        binding = ConfigBinding(cfg, "cameras", "bean", "auto_exposure", type_=bool)
        val = binding.read()
        binding.write(False)
    """

    def __init__(self, cfg, *path, default=None, type_=int):
        self._cfg = cfg
        self._path = path
        self._default = default
        self._type = type_
        self._on_change = None  # callable(coerced_value) or None

    # ── public API ────────────────────────────────────────────────────

    def read(self):
        """Return the current value at *self._path*, falling back to *default*."""
        d = self._cfg
        for key in self._path[:-1]:
            if not isinstance(d, dict):
                return self._default
            d = d.get(key, {})
        return self._type(d.get(self._path[-1], self._default))

    def write(self, value):
        """Coerce *value* to the binding's type, write into cfg, fire callback."""
        d = self._cfg
        for key in self._path[:-1]:
            d = d.setdefault(key, {})
        coerced = self._type(value)
        d[self._path[-1]] = coerced
        if self._on_change is not None:
            self._on_change(coerced)

    def on_change(self, callback):
        """Register a side-effect callback invoked after each :meth:`write`.

        Returns *self* so that callers can chain ``.on_change(...)`` calls.
        """
        self._on_change = callback
        return self
