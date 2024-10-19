from dataclasses import dataclass, fields, asdict
import polars as pl

@dataclass
class BaseSchema:
    def __init__(self):
        pass
    def get_fileds(self):
        return list(fields(self))
    def to_dict(self):
        d = asdict(self)
        return {k: v[0] for k,v in d.items()}

@dataclass
class DataSchema(BaseSchema):
    PulsesN: type = pl.Int32,
    PrimeTheta: type = pl.Float32,
    PrimePhi: type = pl.Float32,
    PrimeEn: type = pl.Float32,
    PrimeNuclN: type = pl.Int32,
    ResponseMuN: type = pl.Int32,
    BundleEnReg: type = pl.Float32,
    EventWeight: type = pl.Float32,
    PulsesChID: type = pl.List(pl.Int16),
    PulsesAmpl: type = pl.List(pl.Float32),
    PulsesTime: type = pl.List(pl.Float32),
    PulsesFlg: type = pl.List(pl.Int32),
    RespMuTheta: type = pl.List(pl.Float32),
    RespMuPhi: type = pl.List(pl.Float32),
    RespMuTrackX: type = pl.List(pl.Float32),
    RespMuTrackY: type = pl.List(pl.Float32),
    RespMuTrackZ: type = pl.List(pl.Float32),
    RespMuDelay: type = pl.List(pl.Float32),
    RespMuEn: type = pl.List(pl.Float32),
    #ev_id: type = pl.Utf8