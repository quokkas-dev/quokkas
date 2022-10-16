

class Lock:
    """
    The global lock for all pipelines. If the pipelines are locked
    via Lock.global_lock(), the pipelines of all dataframes will stop
    being updated. Depending on the operation, they might not be preserved.
    This increases the speed of access to the dataframe, which is
    especially critical if many indexing / slicing operations are performed.

    The user may opt to lock the dataframe with a context manager, i.e. via:
    with qk.Lock.lock():
        print('All pipelines are locked!')
    print('All pipelines are unlocked!')

    Otherwise, the lock / unlock functionality can be used directly:
    qk.Lock.global_lock()
    print('All pipelines are locked!')
    qk.Lock.global_unlock()
    print('All pipelines are unlocked!')
    """

    _unlocked = True

    # don't use lock_count itself to check for locked-ness
    # because python evaluates bools marginally faster than ints
    _lock_count = 0
    _instance = None

    @classmethod
    def global_lock(cls):
        """
        Locks all pipelines

        """
        cls._lock_count += 1
        cls._unlocked = False

    @classmethod
    def global_unlock(cls):
        """
        Unlocks all pipelines

        """
        cls._unlocked = True
        cls._lock_count -= 1

    @classmethod
    def lock(cls):
        """
        Returns an element of the Lock class. This element can be used
        in the context manager

        :return: an element of the Lock class
        """
        if cls._instance is None:
            cls._instance = Lock()
        return cls._instance

    @classmethod
    def __enter__(cls):
        """
        Locks the pipleine, increases lock count

        """
        cls._lock_count += 1
        if cls._unlocked:
            cls._unlocked = False

    @classmethod
    def __exit__(cls, exc_type, exc_value, tb):
        """
        Decreases lock count. If the lock count is at 0, unlocks
        the pipeline

        """
        cls._lock_count -= 1
        if cls._lock_count <= 0:
            cls._unlocked = True
        if exc_type is None:
            return True

