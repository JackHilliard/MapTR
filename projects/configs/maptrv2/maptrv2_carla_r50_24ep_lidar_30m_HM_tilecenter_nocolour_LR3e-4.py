"""Colour-free variant of the 30m tile-centred HM (polyline geometry) config with lr=3e-4.

"""

_base_ = ['./maptrv2_carla_r50_24ep_lidar_30m_HM_tilecenter_nocolour.py']

optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.01)