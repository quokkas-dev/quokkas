from functools import wraps


from ..transforms.operation import Operation
from copy import deepcopy as _deepcopy
from ..core.frames.functional import Functional
from ..core.pipeline.lock import Lock


def preserve(update):
    """
    Preserves the pipeline of the object when transforming it. If update, the
    function will be also added to the transformed object.

    If transformation happens inplace, func() returns None and result is set to self.
    If the function returns a dataframe, the pipeline will be assigned to it.

    :param update: if true, the executed operation will be added to pipeline via Operation class
    :return: result of the underlying function application
    """
    def _preserve(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not Lock._unlocked or not self.pipeline._enabled:
                return func(self, *args, **kwargs)
            pipeline = self.pipeline
            target = self.target
            with pipeline:
                result = func(self, *args, **kwargs)
            if result is None and update:
                op = Operation(func.__name__, *args, **kwargs)
                op.make_inplace()
                self.pipeline.add(op)
            if isinstance(result, Functional):
                if result is self:
                    result.pipeline = pipeline
                    if not func.__name__ == 'targetize':
                        result.target = target
                else:
                    result.pipeline = _deepcopy(pipeline)
                    result.target = _deepcopy(target)
                if update:
                    result.pipeline.add(Operation(func.__name__, *args, **kwargs))

            return result

        return wrapper
    return _preserve


def preserve_inplace(update):
    """
    Preserves the pipeline of the object when transforming it, only meant to be used
    with transformations that happen inplace (e.g. pop()) and return something other than None.

    :param update: if true, the executed operation will be added to the pipeline via Operation
    class
    :return: result of the function application
    """
    def _preserve(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not Lock._unlocked or not self.pipeline._enabled:
                return func(self, *args, **kwargs)
            pipeline = self.pipeline
            with pipeline:
                result = func(self, *args, **kwargs)
            if update:
                self.pipeline.add(Operation(func.__name__, *args, **kwargs))
            return result

        return wrapper
    return _preserve


def incept(func):
    """
    Adds the underlying function as inception to the pipeline.
    See core/pipeline/inception
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if Lock._unlocked and isinstance(result, Functional):
            result.pipeline.add_inception(func, *args, **kwargs)
        return result

    return wrapper


def complete(func):
    """
    Adds the underlying function as completion to the pipeline.
    See core/pipeline/completion
    """
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        if Lock._unlocked:
            df.pipeline.add_completion(getattr(df.__class__, func.__name__), *args, **kwargs)
            with df.pipeline:
                return func(df, *args, **kwargs)
        else:
            return func(df, *args, **kwargs)

    return wrapper
