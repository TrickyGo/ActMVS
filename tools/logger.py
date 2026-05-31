import os
import pickle

class TextColors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    RESET = "\033[0m" 


class Logger:
    def __init__(self, save_dir, cfg):
        self.save_dir = save_dir
        self.budget = cfg["experiment"]["budget"]
        self.record_interval = cfg["experiment"]["record_interval"]
        self.record_time = 1

        self.time_dict = {"mapping": 0, "planning": 0, "flight": 0}
        self.accum_path_length = 0
        self.camera_params_list = []

    @property
    def is_alive(self):
        return self.t_mission < self.budget
    

    def save_dataframe(self, dataframe):
        extrinsic = dataframe["extrinsic"].cpu().view(-1).numpy().tolist()
        intrinsic = dataframe["intrinsic"].cpu().view(-1).numpy().tolist()
        camera_params = extrinsic + intrinsic
        self.camera_params_list.append(camera_params)


    def save_map(self, gaussian_map, map_index):
        map_path = os.path.join(self.save_dir, "map")
        os.makedirs(map_path, exist_ok=True)

        print(
            f"\n {TextColors.YELLOW}----------save map after {self.t_mission} seconds----------{TextColors.RESET}"
        )
        gaussian_map.save(map_path, index=map_index)

        gaussian_map.save_ply(map_path, index=map_index)

        camera_pose_file = os.path.join(map_path, f"cameras_{map_index}.pkl")
        with open(camera_pose_file, "wb") as pickle_file:
            pickle.dump(self.camera_params_list, pickle_file)

        record_file = f"{map_path}/record_info.txt"
        mode = "a" if os.path.exists(record_file) else "w"
        record_data = [
            map_index,
            self.t_mission,
            self.accum_path_length,
        ]
        with open(record_file, mode) as f:
            f.write(" ".join(map(str, record_data)) + "\n")



 
    def update_time(self, item, time_consumption):
        self.time_dict[item] += time_consumption
        print(f"\n {item} time (step): {time_consumption:.2f}")

    def log(self):
        mission_time = self.t_mission
        mapping_percent = self.t_mapping / mission_time
        planning_percent = self.t_planning / mission_time
        flight_percent = self.t_flight / mission_time
        print(f"\n {TextColors.GREEN}-----Log Mission Info:{TextColors.RESET}")
        print(
            f"\n total mission time: {mission_time:.2f},\
                mapping: {mapping_percent *100:.2f}%,\
                planning: {planning_percent*100:.2f}%,\
                flight: {flight_percent*100:.2f}%"
        )

    @property
    def require_record(self):
        if self.t_mission > self.record_time:
            self.record_time += self.record_interval
            return True
        else:
            return False

    @property
    def t_mapping(self):
        return self.time_dict["mapping"]

    @property
    def t_planning(self):
        return self.time_dict["planning"]

    @property
    def t_flight(self):
        return self.time_dict["flight"]

    @property
    def t_mission(self):
        return self.t_mapping + self.t_planning + self.t_flight


