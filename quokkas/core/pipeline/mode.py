from ..frames.functional import Functional


class Mode:
    @staticmethod
    def inplace():
        Functional.global_inplace = True

    @staticmethod
    def outplace():
        Functional.global_inplace = False
