from asyncio.log import logger
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch
from math import pi
from plotter import Plotter

# initialize gym
gym = gymapi.acquire_gym()


custom_parameters = [
        {"name": "--vertical", "action": "store_true", "default": False, "help": "Set ground plane vertical, normal to the x-axis"},
        {"name": "--use_sphere", "action": "store_true", "default": False, "help": "Use crawler model with a sphere as the caster wheel instead of swivel mechanism"},
        {"name": "--use_capsule", "action": "store_true", "default": True, "help": "Replace cylinders with capsules"},
        {"name": "--damping", "type": float, "default": 1.0e8, "help": "Set the damping of the front left/right wheels"},
        {"name": "--stiffness", "type": float, "default": 1.0e4, "help": "Set the stiffness of the front left/right wheels"},
        {"name": "--friction", "type": float, "default": 1., "help": "Set the friction of the front left/right wheels"},
        {"name": "--local", "action": "store_true", "default": False, "help": "Apply force of magnetism in local frame (otherwise env frame)"},
        {"name": "--fix_base", "action": "store_true", "default": False, "help": "Fix base link of robot"},
        {"name": "--force", "type": float, "default": 400., "help": "Apply acceleration for force of magnetism (N/m)"},
        {"name": "--velocity", "type": float, "default": 2., "help": "Apply velocity to front left/right wheels (rad/s)"},
        {"name": "--max_plot_time", "type": int, "default": 200, "help": "Iterations to plot"},
        {"name": "--use_torque", "action": "store_true", "default": False, "help": "Apply effort to DOFs (otherwise velocity)"},
        {"name": "--torque", "type": float, "default": 2., "help": "Apply torque to front left/right wheels (Nm)"}
]

# parse arguments
args = gymutil.parse_arguments(
    description="Test applying forces on wheels of crawler",
    custom_parameters=custom_parameters)

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

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# add ground plane & set default pose
plane_params = gymapi.PlaneParams()
pose = gymapi.Transform()
if args.vertical:
    plane_params.normal = gymapi.Vec3(1, 0, 0)
    pose.p.x = 0.05
    pose.r = gymapi.Quat.from_euler_zyx(0.0, -pi/2, pi)
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2, 0, 0), gymapi.Vec3(0, 0, 0))
else:
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    pose.p.z = 0.05
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(3, 3, 3), gymapi.Vec3(0, 0, 0))

gym.add_ground(sim, plane_params)
    



# # load crawler asset
asset_root = "./crawler_description/"
asset_file = f"urdf/crawler_{'sphere' if args.use_sphere else 'caster'}.urdf"


asset_options = gymapi.AssetOptions()
asset_options.replace_cylinder_with_capsule = args.use_capsule
asset_options.fix_base_link = args.fix_base 
# asset_options.slices_per_cylinder = 20
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

print("== BODY INFORMATION ==")
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

# create force sensors at each wheel
sensor_pose1 = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
sensor_pose2 = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))

sensor_lf = gym.create_asset_force_sensor(asset, LF_WHEEL_ID, sensor_pose1)
sensor_rf = gym.create_asset_force_sensor(asset, RF_WHEEL_ID, sensor_pose2)
sensor_c = gym.create_asset_force_sensor(asset, C_WHEEL_ID, sensor_pose2)


# set up the env grid (only sets up 1 environment)
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
    
    # get actor-relative dof index for front left/right wheels
    actor_dof_dict = gym.get_actor_dof_dict(env, ahandle)
    print('actor_dof_dict', actor_dof_dict)
    fl_id = actor_dof_dict["left_front_wheel_joint"]
    fr_id = actor_dof_dict["right_front_wheel_joint"]
    wheel_ids.append((fl_id, fr_id))

    props = gym.get_actor_dof_properties(env, ahandle)
    
    props["driveMode"].fill(gymapi.DOF_MODE_NONE)
    props["driveMode"][fl_id] = gymapi.DOF_MODE_EFFORT if args.use_torque else gymapi.DOF_MODE_VEL
    props["driveMode"][fr_id] = gymapi.DOF_MODE_EFFORT if args.use_torque else gymapi.DOF_MODE_VEL
    
    # set stiffness on FL and FR wheels
    # stiffness not needed for velocity drive mode
    props["stiffness"].fill(0.0)
    props["stiffness"][fl_id] = args.stiffness
    props["stiffness"][fr_id] = args.stiffness

    # set damping on FL and FR wheels for PD controller (tunable)
    props["damping"][fl_id] = args.damping
    props["damping"][fr_id] = args.damping

    props["friction"][fl_id] = args.friction
    props["friction"][fr_id] = args.friction

    gym.set_actor_dof_properties(env, ahandle, props)

    # rigid_props = gym.get_actor_rigid_shape_properties(env, ahandle)
    # rigid_props[fl_id].friction = 100
    # rigid_props[fr_id].friction = 100

    # rigid_props[fl_id].rolling_friction = 100
    # rigid_props[fr_id].rolling_friction = 100
    # gym.set_actor_rigid_shape_properties(env, ahandle, rigid_props)


    # enable force sensor for actor
    gym.enable_actor_dof_force_sensors(env, ahandle)
    

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(3, 3, 3), gymapi.Vec3(0, 0, 0))

gym.prepare_sim(sim)


rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(rb_tensor)
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)

_fsdata = gym.acquire_force_sensor_tensor(sim)
fsdata = gymtorch.wrap_tensor(_fsdata)

# acquire root state tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
# wrap it in a PyTorch Tensor and create convenient views
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]
root_velocities = root_tensor[:, 7:]

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

frame_count = 0
is_local = args.local
magnet_force = -args.force
velocity = args.velocity
torque = args.torque
force_direction = 2 # z-axis

# Init graph plotter
plot_params = {
    "Vertical": args.vertical,
    "Sphere": args.use_sphere,
    "Capsule": args.use_capsule,
    "Damping": args.damping,
    "Stiffness": args.stiffness,
    "Local": args.local,
    "Force": args.force
}
if args.use_torque:
    plot_params["Torque"] = args.torque
else:
    plot_params["Velocity"] = args.velocity

plotter = Plotter(1, plot_params)
robot_index = 0

while not gym.query_viewer_has_closed(viewer):

    
    gym.refresh_rigid_body_state_tensor(sim)
    # set forces and force positions
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
    if is_local:
        force_positions = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        force_space = gymapi.LOCAL_SPACE
    else:
        force_positions = rb_positions.clone()
        force_space = gymapi.ENV_SPACE
        if args.vertical:
            force_direction = 0 # x-axis
    # set force in negative z-direction at wheel center-of-mass to simulate magnetism
    forces[:, LF_WHEEL_ID, force_direction] = magnet_force
    forces[:, RF_WHEEL_ID, force_direction] = magnet_force
    forces[:, C_WHEEL_ID, force_direction] = magnet_force

    gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), force_space)
   
    # get total number of DOFs
    num_dofs = gym.get_sim_dof_count(sim)
    # apply velocity to each wheel (rad/s)
    actions = torch.zeros(num_dofs, dtype=torch.float32, device=device)


    actions[fl_id] = torque if args.use_torque else velocity
    actions[fr_id] = torque if args.use_torque else velocity


    # apply velocity after 50 frames, allows robot to settle from inital pose
    if args.use_torque:
        if frame_count >= 50:
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))
    else:
        if frame_count >= 50:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(actions))

    gym.refresh_force_sensor_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    
    # print(fsdata)
    # print(dof_states)
    # print(root_positions)
    # print(root_velocities)

    gym.draw_env_rigid_contacts(viewer, env, gymapi.Vec3(1, 0, 0), 1, False)

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

    # only graph after robot has spawned & settled
    if 50 < frame_count < args.max_plot_time + 50:
        logger_vars = {
                'x_pos': root_positions[robot_index, 0].item(),
                'y_pos': root_positions[robot_index, 1].item(),
                'lf_track_vel': dof_states[fl_id, 1].item(),
                'rf_track_vel': dof_states[fr_id, 1].item(),
                'lf_track_torque': -fsdata[sensor_lf, 4].item(),
                'rf_track_torque': -fsdata[sensor_rf, 4].item(),
            }
        if args.use_torque:
            logger_vars["lf_cmd_torque"] = torque
            logger_vars["rf_cmd_torque"] = torque
        else:
            logger_vars["lf_cmd_vel"] = velocity
            logger_vars["rf_cmd_vel"] = velocity
        plotter.log_states(logger_vars)
    elif frame_count == args.max_plot_time + 50:
        plotter.plot_states()

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
