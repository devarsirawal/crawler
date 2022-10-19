from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, normalize
import numpy as np
import torch
import math
from math import pi

from plotter import Plotter

# initialize gym
gym = gymapi.acquire_gym()

custom_parameters = [
        {"name": "--vertical", "action": "store_true", "default": False, "help": "Set ground plane vertical, normal to the y-axis"},
        {"name": "--use_capsule", "action": "store_true", "default": False, "help": "Replace cylinders with capsules"},
        {"name": "--damping", "type": float, "default": 200.0, "help": "Set the damping of the front left/right wheels"},
        {"name": "--stiffness", "type": float, "default": 800.0, "help": "Set the stiffness of the front left/right wheels"},
        {"name": "--friction", "type": float, "default": 0.01, "help": "Set the friction of the front left/right wheels"},
        {"name": "--local", "action": "store_true", "default": False, "help": "Apply force of magnetism in local frame (otherwise env frame)"},
        {"name": "--fix_base", "action": "store_true", "default": False, "help": "Fix base link of robot"},
        {"name": "--rand_orient", "action": "store_true", "default": False, "help": "Start with a random orientation on the crawler"},
        {"name": "--magnet", "type": float, "default": 200., "help": "Apply acceleration for force of magnetism (N/m)"},
        {"name": "--tether", "type": float, "default": 5., "help": "Apply acceleration for force of tether (N/m)"},
        {"name": "--velocity", "type": float, "default": 0.2, "help": "Apply velocity to front left/right wheels (rad/s)"},
        {"name": "--max_plot_time", "type": int, "default": 300, "help": "Iterations to plot"},
]
# parse arguments
args = gymutil.parse_arguments(
    description="Keyboard control of crawler", custom_parameters=custom_parameters)

vertical = args.vertical 
# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
# sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
sim_params.dt = 1/20.
sim_params.substeps = 30 
sim_params.physx.solver_type = 1 # TGS Solver on GPU
sim_params.physx.num_threads = 4
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 2
sim_params.physx.contact_offset = 0.0025
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.0
sim_params.physx.max_depenetration_velocity = 10.0
sim_params.physx.default_buffer_size_multiplier = 5.0
sim_params.physx.max_gpu_contact_pairs = 1048576
sim_params.physx.num_subscenes = 4
sim_params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'
print(device)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# set random seed
np.random.seed(17)
# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# add ground plane
plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.normal = gymapi.Vec3(0,1.0,0) if vertical else gymapi.Vec3(0,0,1)
gym.add_ground(sim, plane_params)

# load crawler asset
asset_root = "../../IsaacGymEnvs/assets/urdf/crawler/"
asset_file = "crawler_caster.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = args.fix_base 
asset_options.disable_gravity = False
asset_options.armature = 0.01
asset_options.replace_cylinder_with_capsule = args.use_capsule 
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
print("== BODY INFORMATION ==")
num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies)
body_dict = gym.get_asset_rigid_body_dict(asset)
print('rigid bodies', body_dict)
dof_dict = gym.get_asset_dof_dict(asset)
print('asset_dof', body_dict)
dof_names = gym.get_asset_dof_names(asset)
print('dof_names', dof_names)
num_dof = len(dof_names)

# subscribe to input events. This allows input to be used to interact
# with the simulation
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "up")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "down")

# Force Sensor stuff
sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
sensor_pose_lw = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
sensor_pose_rw = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
sensor_pose_cw = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))

sensor_props = gymapi.ForceSensorProperties()
sensor_props.use_world_frame = True

sensor_idx_lw = gym.create_asset_force_sensor(asset, 4, sensor_pose, sensor_props)
sensor_idx_rw = gym.create_asset_force_sensor(asset, 5, sensor_pose, sensor_props)
sensor_idx_cw = gym.create_asset_force_sensor(asset, 3, sensor_pose, sensor_props)


num_envs = 1
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, num_per_row)
pose = gymapi.Transform()
if vertical:
    pose.p.y = 0.05 
else: 
    pose.p.z = 0.05
heading = np.random.rand() * 2 * pi if args.rand_orient else -pi/2 
pose.r = gymapi.Quat.from_euler_zyx(-pi/2, heading, 0) if vertical else gymapi.Quat.from_euler_zyx(0, 0, heading)
crawler = gym.create_actor(env, asset, pose, 'actor', 0, 1)

tether_base = torch.tensor([0, 0., 0], dtype=torch.float, requires_grad=False, device=device).repeat(num_envs)
marker_options = gymapi.AssetOptions()
marker_options.fix_base_link = True
# marker_asset = gym.create_sphere(sim, 0.05, marker_options)
# marker_handle = gym.create_actor(env, marker_asset, gymapi.Transform(), "marker", 0, 1)

# indices for bodies
cw_body = body_dict["caster_wheel"]
l_body = body_dict["left_front_wheel"]
r_body = body_dict["right_front_wheel"]

actor_dof_dict = gym.get_actor_dof_dict(env, crawler)
wheel_dof = [actor_dof_dict["left_front_wheel_joint"], actor_dof_dict["right_front_wheel_joint"]]
cwj_dof = actor_dof_dict["caster_wheel_joint"]
cwb_dof = actor_dof_dict["caster_wheel_base_joint"]
props = gym.get_actor_dof_properties(env, crawler)
props["driveMode"].fill(gymapi.DOF_MODE_NONE)
props["driveMode"][wheel_dof] = gymapi.DOF_MODE_VEL
# props["driveMode"][cwb_dof] = gymapi.DOF_MODE_POS
props["stiffness"].fill(0.0) 
props["stiffness"][wheel_dof] = args.stiffness 
props["stiffness"][cwb_dof] = 0.0 
props["damping"].fill(0.0)
props["damping"][wheel_dof] = args.damping
props["friction"][wheel_dof] = args.friction
props["armature"].fill(0.001)
props["effort"].fill(1000.0)
# cwb_dof_handle = gym.find_actor_dof_handle(env, crawler, 'caster_wheel_base_joint')
# gym.set_dof_target_position(env, cwb_dof_handle, pi/2)
# gym.set_actor_dof_properties(env, crawler, props)
# props["stiffness"][cwb_dof] = 0 
gym.set_actor_dof_properties(env, crawler, props)
gym.enable_actor_dof_force_sensors(env, crawler)

# print(gym.get_env_rigid_body_states(env, gymapi.STATE_POS))
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(3, 3, 3), gymapi.Vec3(0, 0, 0))

gym.prepare_sim(sim)

# acquire root state tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
# wrap it in a PyTorch Tensor and create convenient views
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]
root_velocities = root_tensor[:, 7:]

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

_rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_tensor)
print(rb_states)
rb_positions = rb_states[:, 0:3].view(num_envs, num_bodies, 3)
rb_orients = rb_states[:, 3:7].view(num_envs, num_bodies, 4)

# _dof_forces = gym.acquire_dof_force_tensor(sim)
# dof_forces = gymtorch.wrap_tensor(_dof_forces)
# 
_fsdata = gym.acquire_force_sensor_tensor(sim)
fsdata = gymtorch.wrap_tensor(_fsdata)

num_dofs = gym.get_sim_dof_count(sim)
init_dof_pos = dof_states[:, :].clone() 
init_dof_pos[cwb_dof, 0] = np.random.rand() * 2 * pi
# init_dof_pos = dof_states.clone() 
# init_dof_pos[cwb_dof, 0] = pi/2
print(dof_states.shape)
print(dof_states)
print(init_dof_pos)

result = gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(init_dof_pos))
print(result)
# gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(init_dof_pos))
frame_count = 0

turn = 0
forward = 0 
speed = 5 
plotter = Plotter(1, {})
robot_index = 0
is_local = args.local 
magnetism = -args.magnet
tether = args.tether
force_dir = 1 if (vertical and not is_local) else 2
while not gym.query_viewer_has_closed(viewer):
    # set forces and force positions
    
    # print("LW Pos: ", dof_states[wheel_dof[0], 0])
    # print("RW Pos: ", dof_states[wheel_dof[1], 0])

    # print("Orients Shape: ",rb_orients.shape)
    # print("Forces Shape: ", forces.shape)
    # forces[:, 3, :] = quat_rotate(rb_orients[:, 3, :], forces[:, 3, :])
    # forces[:, 4, :] = quat_rotate(rb_orients[:, 4, :], forces[:, 4, :])
    # forces[:, 5, :] = quat_rotate(rb_orients[:, 5, :], forces[:, 5, :])
    # final_orients = rb_orients[:, 0, :].repeat(1, num_bodies, 1)
    # print("Rot Forces Shape: ", forces.shape)
    # print("Final Orients Shape: ", final_orients.shape)
    # forces = quat_rotate_inverse(final_orients, forces)
    # forces[:, 3, :] = quat_rotate_inverse(final_orients[:, 3, :], forces[:, 3, :])
    # forces[:, 4, :] = quat_rotate_inverse(final_orients[:, 4, :], forces[:, 4, :])
    # forces[:, 5, :] = quat_rotate_inverse(final_orients[:, 5, :], forces[:, 5, :])
    # print(forces)
    # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
    # print("Pos: ", rb_positions)
    # print("Orient: ", rb_orients)
    # print("lw_pos: ", rb_positions[0,4] - rb_positions[0,0])
    # force_positions = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float, requires_grad=False)
    # force_positions[:, 2] = torch.Tensor([-0.13, 0.0, 0.0])
    # force_positions[:, 4] = torch.Tensor([0.0, 0.0975, -0.0425])
    # force_positions[:, 5] = torch.Tensor([0.0, -0.0975, -0.0425])
    # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.LOCAL_SPACE)
    # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
    # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(rb_positions.clone()), gymapi.ENV_SPACE)

    # get total number of DOFs
    num_dofs = gym.get_sim_dof_count(sim)

    if is_local:
        magnet_forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float, requires_grad=False)
        # theta_lw = dof_states[wheel_dof[0], 0].item()
        # forces[:, 4, 0] = -np.sin(theta_lw) * magnetism
        # forces[:, 4, 2] = np.cos(theta_lw) * magnetism
        # theta_rw = dof_states[wheel_dof[1], 0].item()
        # forces[:, 5, 0] = -np.sin(theta_rw) * magnetism
        # forces[:, 5, 2] = np.cos(theta_rw) * magnetism
        # theta_cw = dof_states[cwj_dof, 0].item()
        # forces[:, 3, 0] = -np.sin(theta_cw) * magnetism
        # forces[:, 3, 2] = np.cos(theta_cw) * magnetism
        # # print(forces)
        # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)

        theta_lw = dof_states[wheel_dof[0]::num_dof, 0]
        magnet_forces[:, 4, 0] = -torch.sin(theta_lw) * magnetism
        magnet_forces[:, 4, 2] = torch.cos(theta_lw) * magnetism
        theta_rw = dof_states[wheel_dof[1]::num_dof, 0]
        magnet_forces[:, 5, 0] = -torch.sin(theta_rw) * magnetism
        magnet_forces[:, 5, 2] = torch.cos(theta_rw) * magnetism
        theta_cw = dof_states[cwj_dof::num_dof, 0]
        magnet_forces[:, 3, 0] = -torch.sin(theta_cw) * magnetism
        magnet_forces[:, 3, 2] = torch.cos(theta_cw) * magnetism
        # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(magnet_forces), None, gymapi.LOCAL_SPACE)
    else:
        magnet_forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float, requires_grad=False)
        magnet_forces[:, 0, 2] = magnetism 
        # forces[:, 2, force_dir] = -magnetism * 0.1
        # gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(rb_positions.clone()), gymapi.ENV_SPACE)
        # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
    # print(dof_forces)
    # print(fsdata)


    _tether_forces = tether * normalize(tether_base - rb_positions)
    # print("Global Tether Forces: \n", _tether_forces)

    
    tether_forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float, requires_grad=False)
    tether_forces[:, 1, 2] = _tether_forces[:, 1, 2] 
    # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
    final_orients = rb_orients[:, 1, :].repeat(1, num_bodies, 1)
    # print("Base Orient: \n", final_orients)
    tether_forces[:, 1, :] = quat_rotate_inverse(final_orients[:, 1, :], _tether_forces[:, 1, :])
    # print("Local Tether Forces: \n", tether_forces)
    forces = magnet_forces + tether_forces
    # print("Total forces: \n", forces)
    gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
    
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "left":
            turn = 0.5 if evt.value > 0 else 0
        if evt.action == "right":
            turn = -0.5 if evt.value > 0 else 0
        if evt.action == "up":
            forward = 1 if evt.value > 0 else 0
        if evt.action == "down":
            forward = -1 if evt.value > 0 else 0
    # apply velocity to each wheel (rad/s)
    actions = torch.zeros(num_dofs, dtype=torch.float32, device=device)

    lw_vel = (forward-turn) * speed 
    rw_vel = (forward+turn) * speed

    actions[wheel_dof[0]] = lw_vel 
    actions[wheel_dof[1]] = rw_vel 
    # result = gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(init_dof_pos))
    # print(result)

    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(actions))
    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_force_sensor_tensor(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1

    # only graph after robot has spawned & settled
    # if 50 < frame_count < args.max_plot_time + 50:
    #     logger_vars = {
    #             'x_pos': root_positions[robot_index, 0].item(),
    #             'y_pos': root_positions[robot_index, 1].item(),
    #             'lf_track_vel': dof_states[wheel_dof[0], 1].item(),
    #             'rf_track_vel': dof_states[wheel_dof[1], 1].item(),
    #         }
    #     logger_vars["lf_cmd_vel"] = lw_vel 
    #     logger_vars["rf_cmd_vel"] = rw_vel 
    #     plotter.log_states(logger_vars)
    # elif frame_count == args.max_plot_time + 50:
    #     plotter.plot_states()

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)



