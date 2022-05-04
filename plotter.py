# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process

# To plot
# - (x,y) position
# - velocity of each wheel (expected vs. actual)
# - 

class Plotter:
    def __init__(self, dt, params):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.params = params
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 2
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        fig.suptitle(f"{self.params}", fontsize=14)
        # plot joint targets and measured positions
        a = axs[0,0]
        if log["x_pos"] and log["y_pos"]: a.plot(log["x_pos"], log["y_pos"], label='path')
        a.set(xlabel='x', ylabel='y', title=f'Crawler Path')
        a.legend()
        # plot left front wheel velocity
        a = axs[0,1]
        if log["lf_track_vel"]: a.plot(time, log["lf_track_vel"], label='measured')
        if log["lf_cmd_vel"]: a.plot(time, log["lf_cmd_vel"], label='target')
        a.set(xlabel='time [frames]', ylabel=' Angular Velocity [rad/s]', title='Left Front Wheel')
        a.set_ylim([0, 2])
        a.legend()
        # plot right front wheel velocity
        a = axs[0,2]
        if log["rf_track_vel"]: a.plot(time, log["rf_track_vel"], label='measured')
        if log["rf_cmd_vel"]: a.plot(time, log["rf_cmd_vel"], label='target')
        a.set(xlabel='time [frames]', ylabel=' Angular Velocity [rad/s]', title='Right Front Wheel')
        a.set_ylim([0, 2])
        a.legend()
        # plot left front wheel torque
        a = axs[1,1]
        if log["lf_track_torque"]: a.plot(time, log["lf_track_torque"], label='measured')
        if log["lf_cmd_torque"]: a.plot(time, log["lf_cmd_torque"], label='target')
        a.set(xlabel='time [frames]', ylabel=' Torque [Nm]', title='Left Front Wheel')
        a.legend()
        # plot right front wheel torque
        a = axs[1,2]
        if log["rf_track_torque"]: a.plot(time, log["rf_track_torque"], label='measured')
        if log["rf_cmd_torque"]: a.plot(time, log["rf_cmd_torque"], label='target')
        a.set(xlabel='time [frames]', ylabel=' Torque [Nm]', title='Right Front Wheel')
        a.legend()
        plt.show()
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
    