from dataclasses import dataclass, field, fields, asdict

@dataclass
class BaseConfig:
    def __init__(self):
        pass

    def get_fileds(self):
        return list(fields(self))

    def to_dict(self):
        return asdict(self)