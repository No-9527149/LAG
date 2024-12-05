import pymap3d
import numpy as np
from envs.JSBSim.utils.utils import LLA2NEU, NEU2LLA


def get_dis(ego_x, ego_y, ego_z, enm_x, enm_y, enm_z):
    dis = np.array([ego_x - enm_x, ego_y - enm_y, ego_z - enm_z])
    return np.linalg.norm(dis)


if __name__ == "__main__":
    ego_x, ego_y, ego_z = pymap3d.geodetic2ned(120.0, 60.0, 8000.0, 120.0, 60.0, 0.0)
    enm_x, enm_y, enm_z = pymap3d.geodetic2ned(120.0, 60.2, 8000.0, 120.0, 60.0, 0.0)
    print(get_dis(ego_x, ego_y, ego_z, enm_x, enm_y, enm_z))
