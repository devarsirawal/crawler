from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch

USE_SPHERE=True

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Test applying forces on wheels of crawler")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'


sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# # load crawler asset
asset_root = "./crawler_description/"
asset_file = f"urdf/crawler_{'sphere' if USE_SPHERE else 'caster'}.urdf"

asset_options = gymapi.AssetOptions()
asset_options.replace_cylinder_with_capsule = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies)
body_dict = gym.get_asset_rigid_body_dict(asset)
print('rigid bodies', body_dict)
dof_dict = gym.get_asset_dof_dict(asset)
print('asset_dof', body_dict)

# indices for bodies
C_WHEEL_ID = body_dict["caster_wheel"]
LF_WHEEL_ID = body_dict["left_front_wheel"]
RF_WHEEL_ID = body_dict["right_front_wheel"]


# default pose
pose = gymapi.Transform()
pose.p.z = 0.025

# set up the env grid
num_envs = 1
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# set random seed
np.random.seed(17)

envs = []
handles = []
wheel_ids = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    ahandle = gym.create_actor(env, asset, pose, "actor", i, 1)
    handles.append(ahandle)
    
    actor_dof_dict = gym.get_actor_dof_dict(env, ahandle)
    print('actor_dof_dict', actor_dof_dict)
    lfw_id = actor_dof_dict["left_front_wheel_joint"]
    rfw_id = actor_dof_dict["right_front_wheel_joint"]
    wheel_ids.append((lfw_id, rfw_id))
    props = gym.get_actor_dof_properties(env, ahandle)
    
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["driveMode"][lfw_id] = gymapi.DOF_MODE_VEL
    props["driveMode"][rfw_id] = gymapi.DOF_MODE_VEL
    
    props["stiffness"].fill(0.0)

    props["damping"][lfw_id] = 1
    props["damping"][rfw_id] = 1

    props["stiffness"][lfw_id] = 0
    props["stiffness"][rfw_id] = 0

    gym.set_actor_dof_properties(env, ahandle, props)


gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(3, 3, 3), gymapi.Vec3(0, 0, 0))

gym.prepare_sim(sim)

rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_tensor)
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)

frame_count = 0
while not gym.query_viewer_has_closed(viewer):

    
    gym.refresh_rigid_body_state_tensor(sim)

    # set forces and force positions
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
    force_positions = rb_positions.clone()
    # set force in negative z-direction at wheel center-of-mass to simulate magnetism
    forces[:, LF_WHEEL_ID, 2] = -400
    forces[:, RF_WHEEL_ID, 2] = -400
    forces[:, C_WHEEL_ID, 2] = -100
    gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
   

    # get total number of DOFs
    num_dofs = gym.get_sim_dof_count(sim)
    # apply velocity to each wheel at 1 rad/s
    velocities = torch.zeros(num_dofs, dtype=torch.float32, device=device)

    velocities[lfw_id] = 1
    velocities[rfw_id] = 1

    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(velocities))

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)