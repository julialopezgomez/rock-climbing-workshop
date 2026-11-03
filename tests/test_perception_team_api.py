import pytest

from rcr.perception.challenge.team_api import wall_model
from rcr.types import LIMBS


def test_wall_export_uses_mount_not_socket_and_requires_start():
    mapping = {
        "frame": "wall",
        "units": "metres",
        "holds": [
            {
                "id": "cup_c00_r02",
                "mounting_transform": {"translation_m": [0.0325, 0.04, 0.0]},
                "contact_transform": {"translation_m": [0.0325, 0.026, 0.026]},
            }
        ],
    }
    start = dict.fromkeys(LIMBS, None)
    start["left_hand"] = "cup_c00_r02"
    wall = wall_model(mapping, wall_id="observed", level=1, verified_start_contacts=start)
    assert wall.holds[0].id == "c0_r2"
    assert wall.holds[0].position == (0.0325, 0.0, 0.04)
    assert wall.start.contacts["left_hand"] == "c0_r2"
    with pytest.raises(ValueError, match="all four"):
        wall_model(mapping, wall_id="observed", level=1, verified_start_contacts={})
