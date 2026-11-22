"""Small stand-ins for pydantic and litellm, so the orchestrator's control flow
can be tested without installing either. Only what those tests need is
implemented: required fields, defaults, nested models and field validators.
The real libraries are covered by test_with_real_libraries.py.
"""
import inspect
import sys
import types


class FakeValidationError(Exception):
    pass


class _FieldInfo:
    _UNSET = object()

    def __init__(self, default=_UNSET, default_factory=None, description=None):
        self.default = default
        self.default_factory = default_factory
        self.description = description

    @property
    def required(self):
        return self.default is self._UNSET and self.default_factory is None

    def resolve_default(self):
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is not self._UNSET:
            return self.default
        raise FakeValidationError("no default available")


def Field(default=_FieldInfo._UNSET, default_factory=None, description=None, **_kwargs):
    return _FieldInfo(default=default, default_factory=default_factory, description=description)


def field_validator(*field_names, **_kwargs):
    def decorator(fn):
        # Support being wrapped in @classmethod in real pydantic usage;
        # unwrap if so, we just need the underlying function.
        if isinstance(fn, classmethod):
            fn = fn.__func__
        fn.__fake_validator_fields__ = field_names
        return classmethod(fn)
    return decorator


class _FakeModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        annotations = {}
        for base in reversed(cls.__mro__):
            annotations.update(getattr(base, "__annotations__", {}))
        cls.__fake_annotations__ = annotations

        field_infos = {}
        validators = {}
        for base in cls.__mro__:
            for attr_name, attr_val in vars(base).items():
                if isinstance(attr_val, _FieldInfo):
                    field_infos.setdefault(attr_name, attr_val)
                if isinstance(attr_val, classmethod) and hasattr(attr_val.__func__, "__fake_validator_fields__"):
                    for fname in attr_val.__func__.__fake_validator_fields__:
                        validators.setdefault(fname, []).append(attr_val.__func__)
        cls.__fake_field_infos__ = field_infos
        cls.__fake_validators__ = validators
        return cls


class FakeBaseModel(metaclass=_FakeModelMeta):
    def __init__(self, **kwargs):
        annotations = self.__fake_annotations__
        field_infos = self.__fake_field_infos__
        validators = self.__fake_validators__

        for name, type_hint in annotations.items():
            if name in kwargs:
                value = kwargs[name]
            elif name in field_infos:
                value = field_infos[name].resolve_default()
            elif hasattr(self.__class__, name) and not isinstance(getattr(self.__class__, name), _FieldInfo):
                value = getattr(self.__class__, name)  # plain class-level default e.g. `= "1.0"`
            else:
                raise FakeValidationError(f"missing required field: {name}")

            # Nested BaseModel coercion: if the declared type is a FakeBaseModel
            # subclass and we got a dict, construct it.
            if isinstance(type_hint, type) and issubclass(type_hint, FakeBaseModel) and isinstance(value, dict):
                value = type_hint(**value)

            if name in validators:
                for validator_fn in validators[name]:
                    value = validator_fn(self.__class__, value)

            setattr(self, name, value)

    @classmethod
    def model_validate(cls, data):
        if not isinstance(data, dict):
            raise FakeValidationError(f"expected dict, got {type(data)}")
        if data.get("_invalid"):
            raise FakeValidationError("payload marked invalid for test")
        return cls(**data)

    def model_dump(self):
        return {k: (v.model_dump() if isinstance(v, FakeBaseModel) else v)
                for k, v in self.__dict__.items()}

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.__dict__}>"


def install_fake_pydantic():
    mod = types.ModuleType("pydantic")
    mod.BaseModel = FakeBaseModel
    mod.ValidationError = FakeValidationError
    mod.Field = Field
    mod.field_validator = field_validator
    sys.modules["pydantic"] = mod
