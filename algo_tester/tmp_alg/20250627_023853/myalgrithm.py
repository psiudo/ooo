# ───────────────────── 0. 공통 임포트 & 출력 함수 ─────────────────────
import json, collections, heapq, random, time, copy, os, sys
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import numpy as np

# 로컬 테스트를 위한 util.py와 jsbeautifier 임포트
try:
    import util
    import jsbeautifier
except ImportError:
    # 대회 서버 등 특정 환경에서는 없을 수 있으므로 pass 처리
    pass

# ---------------------------------------------------------------------
# 1. 유틸리티 함수
# ---------------------------------------------------------------------
def get_shortest_path(graph, start, end):
    if start == end: return [start]
    q = collections.deque([(start, [start])])
    visited = {start}
    while q:
        curr, path = q.popleft()
        if curr == end: return path
        for nbr in graph.get(curr, []):
            if nbr not in visited:
                visited.add(nbr); q.append((nbr, path + [nbr]))
    return None

# ---------------------------------------------------------------------
# 2. V2.6 스마트 환경 클래스 (학습 코드와 동일)
# ---------------------------------------------------------------------
class ShipEnv:
    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.num_nodes, self.edges, self.num_ports, self.fixed_cost = \
            problem_data['N'], problem_data['E'], problem_data['P'], problem_data['F']
        self.graph = collections.defaultdict(list)
        for u, v in self.edges: self.graph[u].append(v); self.graph[v].append(u)
        self.shortest_paths = {i: {j: get_shortest_path(self.graph, i, j) for j in range(self.num_nodes)} for i in range(self.num_nodes)}
        self.cars = []
        car_id_counter = 0
        for i, ((o, d), q) in enumerate(problem_data['K']):
            for _ in range(q):
                self.cars.append({'id': car_id_counter, 'demand_id': i, 'origin': o, 'dest': d}); car_id_counter += 1
        self.total_cars = len(self.cars)
        self.reset()

    def reset(self):
        self.current_port, self.node_status, self.car_locations = 0, [-1] * self.num_nodes, [-1] * self.total_cars
        self.cars_on_board, self.newly_generated_routes = set(), []
        return self._get_state()

    def _get_state(self):
        node_state = np.zeros(self.num_nodes)
        for i, car_id in enumerate(self.node_status):
            if car_id != -1: node_state[i] = self.cars[car_id]['dest'] - self.current_port + 1
        port_state = np.zeros(self.num_ports)
        if self.current_port < self.num_ports: port_state[self.current_port] = 1
        waiting_cars = np.zeros(self.num_ports)
        for car in self.cars:
            if car['id'] not in self.cars_on_board and car['origin'] == self.current_port:
                waiting_cars[car['dest']] += 1
        cars_on_board_info = np.zeros((self.total_cars, 2))
        for car_id in self.cars_on_board:
            cars_on_board_info[car_id, 0] = self.cars[car_id]['dest'] - self.current_port
            path_to_gate = self.shortest_paths.get(self.car_locations[car_id], {}).get(0)
            if path_to_gate:
                cars_on_board_info[car_id, 1] = sum(1 for node in path_to_gate[1:] if self.node_status[node] != -1)
        return np.concatenate([
            node_state, port_state, waiting_cars, cars_on_board_info.flatten()
        ]).astype(np.float32)

    def _calculate_path_cost(self, path):
        if not path or len(path) <= 1: return 0
        return self.fixed_cost + (len(path) - 1)

    def get_legal_actions(self):
        legal_actions = []
        unload_cars = {c['id'] for c in self.cars if c['id'] in self.cars_on_board and c['dest'] == self.current_port}
        for car_id in unload_cars:
            legal_actions.append(('UNLOAD', car_id))
        load_cars = {c['id'] for c in self.cars if c['id'] not in self.cars_on_board and c['origin'] == self.current_port}
        if any(s == -1 for s in self.node_status[1:]):
            for car_id in load_cars:
                legal_actions.append(('LOAD', car_id))
        legal_actions.append(('WAIT', -1))
        return legal_actions

    def _find_best_spot(self, car_id_to_load):
        cars_to_leave_later = {c['id'] for c in self.cars if c['id'] in self.cars_on_board and c['dest'] > self.cars[car_id_to_load]['dest']}
        best_spot, max_depth, min_blocks = -1, -1, float('inf')
        empty_nodes = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]
        if not empty_nodes: return -1 # No empty spots
        for spot in empty_nodes:
            path_to_spot = self.shortest_paths[0][spot]
            blocks = sum(1 for car_id in cars_to_leave_later if self.car_locations[car_id] in path_to_spot)
            depth = len(path_to_spot)
            if blocks < min_blocks:
                min_blocks, max_depth, best_spot = blocks, depth, spot
            elif blocks == min_blocks and depth > max_depth:
                max_depth, best_spot = depth, spot
        return best_spot

    def step(self, action):
        a_type, car_id = action
        self.newly_generated_routes = []
        total_cost, reward = 1.0, 0.0
        
        # 실제 경로를 생성하여 반환하도록 수정
        if a_type == 'LOAD':
            target_node = self._find_best_spot(car_id)
            if target_node == -1: return self._get_state(), -100, True
            path = self.shortest_paths[0][target_node]
            self.node_status[target_node], self.car_locations[car_id] = car_id, target_node
            self.cars_on_board.add(car_id)
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": path})
            total_cost += self._calculate_path_cost(path)
            reward += 1
        elif a_type == 'UNLOAD':
            if car_id not in self.cars_on_board: return self._get_state(), -1000, True
            start_node = self.car_locations[car_id]
            path_to_gate = self.shortest_paths[start_node][0]
            blockers = [self.node_status[n] for n in path_to_gate[1:] if self.node_status[n] != -1]
            
            # 재배치 시뮬레이션 및 경로 생성
            temp_unloaded = {}
            for blk_id in blockers:
                blk_node = self.car_locations[blk_id]
                path_out = self.shortest_paths[blk_node][0]
                self.newly_generated_routes.append({"demand_id": self.cars[blk_id]["demand_id"], "route": list(reversed(path_out))})
                total_cost += self._calculate_path_cost(path_out)
                self.node_status[blk_node] = -1
                temp_unloaded[blk_id] = blk_node
                
            # 원래 차 하역
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": list(reversed(path_to_gate))})
            self.node_status[start_node] = -1
            self.cars_on_board.remove(car_id)
            self.car_locations[car_id] = -1
            total_cost += self._calculate_path_cost(path_to_gate)
            reward += 5
            
            # 임시 하역 차량 재적재
            for blk_id, old_node in temp_unloaded.items():
                new_spot = self._find_best_spot(blk_id)
                if new_spot == -1: return self._get_state(), -100, True # Should not happen
                path_in = self.shortest_paths[0][new_spot]
                self.newly_generated_routes.append({"demand_id": self.cars[blk_id]["demand_id"], "route": path_in})
                self.node_status[new_spot] = blk_id
                self.car_locations[blk_id] = new_spot
                total_cost += self._calculate_path_cost(path_in)
        elif a_type == 'WAIT':
            self.current_port += 1
            reward -= 0.5
            
        final_reward = reward - total_cost
        done = self.current_port >= self.num_ports
        if done:
            if not self.cars_on_board and all(loc == -1 for loc in self.car_locations):
                final_reward += 1000
            else:
                final_reward -= 1000
        return self._get_state(), final_reward, done

# ---------------------------------------------------------------------
# 3. 에이전트, 신경망, 액션 매퍼 클래스 (학습 코드와 동일)
# ---------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, action_size)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)

class ActionMapper:
    def __init__(self, problem):
        self.action_to_idx, self.idx_to_action = {}, []
        self._add(('WAIT', -1)); num_cars = sum(q for _, q in problem['K'])
        for car_id in range(num_cars):
            self._add(('UNLOAD', car_id)); self._add(('LOAD', car_id))
    def _add(self, action):
        if action not in self.action_to_idx:
            self.idx_to_action.append(action); self.action_to_idx[action] = len(self.idx_to_action) - 1
    def get(self, action): return self.action_to_idx.get(action)
    def __len__(self): return len(self.idx_to_action)

class DQNAgent:
    def __init__(self, state_size, action_mapper, model_path=None):
        self.GAMMA, self.LR = 0.99, 5e-4
        self.action_mapper = action_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        self.target_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        if model_path: self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.env = None

    def select_action(self, state, use_exploration=False): # 추론 시에는 exploration 사용 안 함
        legal = self.env.get_legal_actions()
        if not legal: return ('WAIT', -1)
        with torch.no_grad():
            qs = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))[0]
            legal_q = {a: qs[self.action_mapper.get(a)] for a in legal if self.action_mapper.get(a) is not None}
            return max(legal_q, key=legal_q.get) if legal_q else ('WAIT', -1)

# ---------------------------------------------------------------------
# 4. ✨ 최종 제출용 알고리즘 함수 ✨
# ---------------------------------------------------------------------
def algorithm(prob_info, timelimit=60):
    start_time = time.time()
    
    # [수정 1] 모델 경로 설정 (V2.6 모델 사용)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    MODEL_PATH = os.path.join(script_dir, "best_model_v2.6.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return {p: [] for p in range(prob_info['P'])}

    # [수정 2] 모델 크기 고정 (state_dict 오류 방지)
    # V2.6 학습 시 사용된 고정된 모델 크기 (학습 로그에서 확인된 값)
    MODEL_STATE_SIZE = 2017
    MODEL_ACTION_SIZE = 1625
    MASTER_MAPPER_NUM_CARS = 812 # 1625 = 1 + 812 * 2
    
    master_problem = {'K': [[(0, 1), MASTER_MAPPER_NUM_CARS]]}
    master_action_mapper = ActionMapper(master_problem)
    
    # 에이전트는 '고정된 최대 크기'로 생성
    agent = DQNAgent(MODEL_STATE_SIZE, master_action_mapper, model_path=MODEL_PATH)

    # 실제 환경과 현재 문제에 맞는 액션 매퍼 설정
    env = ShipEnv(prob_info)
    agent.env = env
    agent.action_mapper = ActionMapper(prob_info)
    
    # --- 솔루션 생성 루프 ---
    solution_routes = {p: [] for p in range(prob_info['P'])}
    state = env.reset()
    done = False
    
    while not done:
        if time.time() - start_time > timelimit - 2:
            print("Timeout reached, terminating.")
            break

        # [수정 3] 데이터 타입(dtype) 통일
        padded_state = np.zeros(MODEL_STATE_SIZE, dtype=np.float32)
        padded_state[:len(state)] = state
        
        action = agent.select_action(padded_state, use_exploration=False)
        
        if action is None: action = ('WAIT', -1)
        
        current_port = env.current_port
        next_state, reward, done = env.step(action)
        
        if env.newly_generated_routes and current_port < prob_info['P']:
            solution_routes[current_port].extend(env.newly_generated_routes)
        
        state = next_state

    # [수정 4] 최종 반환 형식 변환
    final_solution = {p: [] for p in range(prob_info['P'])}
    for port, routes in solution_routes.items():
        for route_info in routes:
            final_solution[port].append([route_info['route'], route_info['demand_id']])
            
    return final_solution

# ---------------------------------------------------------------------
# 5. ⚙️ 로컬 테스트용 실행 블록 ⚙️
# ---------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r') as f:
            prob_info = json.load(f)
            
        # numpy_to_python 헬퍼 함수 정의
        def numpy_to_python(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")

        try:
            alg_start_time = time.time()
            solution = algorithm(prob_info, timelimit)
            alg_end_time = time.time()

            checked_solution = util.check_feasibility(prob_info, solution)
            checked_solution['time'] = alg_end_time - alg_start_time
            
            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
            print(f'Results are saved as file results.json')
            sys.exit(0)
        except Exception as e:
            print(f"An exception occurred: {repr(e)}")
            sys.exit(1)
    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)