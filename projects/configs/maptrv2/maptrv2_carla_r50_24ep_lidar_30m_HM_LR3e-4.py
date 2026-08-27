"""30m HM (polyline geometry) config with lr=3e-4.

"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM.py']

optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.01)