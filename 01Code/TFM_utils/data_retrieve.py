from dataclasses import dataclass, field

@dataclass
class Data_Retrieved_per_file:
    # Data information
    map_number: int | None = None # internal mapping
    row_db: int | None = None

    # Wind information
    heights: list[float] | None = None
    wind_real: list[float] | None = None # value for each height
    wind_direction: float | None = None # Degrees
    wind_variance: list[float] | None = None # value for each height
    wind_retrieved_10m: dict[int, float] = field(default_factory=dict) # value for each level

    # Atmospheric information
    L_real: float | None = None
    L_retrieved: float | None = None
    u_star_retrieved: float | None = None
    stability: str | None = None

    # Wind velocity from profile extrapolation
    wind_new_value_10m: float | None = None


@dataclass
class Data_Retrieved:
    results: dict[str, Data_Retrieved_per_file] = field(default_factory=dict)
    
all_data = Data_Retrieved()