# ───────────────────── 0. 공통 임포트 & 출력 함수 ─────────────────────
import json, collections, heapq, random, time, copy, os, sys
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import numpy as np

try:
    import util
    import jsbeautifier
except ImportError:
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
# 2. V2.6 '간소화된' 환경 클래스 (✨✨ 학습 시 사용된 버전과 100% 동일 ✨✨)
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
        self.cars_on_board = set()
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
            path_to_gate = self.shortest_paths.get(self.car_locations[car_id], {}).get(0, [])
            if path_to_gate:
                cars_on_board_info[car_id, 1] = sum(1 for node in path_to_gate[1:] if self.node_status[node] != -1)
        return np.concatenate([node_state, port_state, waiting_cars, cars_on_board_info.flatten()]).astype(np.float32)

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
        if not empty_nodes: return -1
        for spot in empty_nodes:
            path_to_spot = self.shortest_paths[0].get(spot, [])
            blocks = sum(1 for car_id in cars_to_leave_later if self.car_locations.get(car_id) in path_to_spot)
            depth = len(path_to_spot)
            if blocks < min_blocks or (blocks == min_blocks and depth > max_depth):
                min_blocks, max_depth, best_spot = blocks, depth, spot
        return best_spot

    def step(self, action):
        a_type, car_id = action
        total_cost, reward = 1.0, 0.0
        if a_type == 'LOAD':
            target_node = self._find_best_spot(car_id)
            if target_node == -1: return self._get_state(), -100, True
            path = self.shortest_paths[0][target_node]
            self.node_status[target_node], self.car_locations[car_id] = car_id, target_node
            self.cars_on_board.add(car_id)
            total_cost += self._calculate_path_cost(path)
            reward += 1
        elif a_type == 'UNLOAD':
            if car_id not in self.cars_on_board: return self._get_state(), -1000, True
            start_node = self.car_locations[car_id]
            path_to_gate = self.shortest_paths[start_node][0]
            rehandling_cost = sum(100 for n in path_to_gate[1:] if self.node_status[n] != -1)
            total_cost += rehandling_cost
            reward -= rehandling_cost / 10
            self.node_status[start_node] = -1
            self.cars_on_board.remove(car_id)
            self.car_locations[car_id] = -1
            total_cost += self._calculate_path_cost(path_to_gate)
            reward += 5
        elif a_type == 'WAIT':
            self.current_port += 1
            reward -= 0.5
        final_reward = reward - total_cost
        done = self.current_port >= self.num_ports
        if done:
            if not self.cars_on_board and all(loc == -1 for loc in self.car_locations if loc is not None):
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
        self.action_mapper = action_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        if model_path: self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        self.env = None

    def select_action(self, state):
        legal = self.env.get_legal_actions()
        if not legal: return ('WAIT', -1)
        with torch.no_grad():
            qs = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))[0]
            legal_q = {a: qs[self.action_mapper.get(a)] for a in legal if self.action_mapper.get(a) is not None}
            return max(legal_q, key=legal_q.get) if legal_q else ('WAIT', -1)

# ---------------------------------------------------------------------
# 4. ✨ 최종 제출용 알고리즘 함수 (전략 결정 + 전략 실행) ✨
# ---------------------------------------------------------------------
def algorithm(prob_info, timelimit=60):
    start_time = time.time()
    
    # [1단계: 전략 결정] =================================================
    # 학습된 에이전트를 '간소화된 환경'에서 실행하여 최적의 행동 순서를 결정합니다.
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    MODEL_PATH = os.path.join(script_dir, "best_model_v2.6.pth")
    if not os.path.exists(MODEL_PATH): return {p: [] for p in range(prob_info['P'])}

    MODEL_STATE_SIZE, MODEL_ACTION_SIZE, MASTER_MAPPER_NUM_CARS = 2017, 1625, 812
    master_problem = {'K': [[(0, 1), MASTER_MAPPER_NUM_CARS]]}
    master_action_mapper = ActionMapper(master_problem)
    agent = DQNAgent(MODEL_STATE_SIZE, master_action_mapper, model_path=MODEL_PATH)
    
    env = ShipEnv(prob_info)
    agent.env = env
    agent.action_mapper = ActionMapper(prob_info)
    
    state = env.reset()
    done = False
    action_history = []
    
    while not done:
        if time.time() - start_time > timelimit - 2: break
        padded_state = np.zeros(MODEL_STATE_SIZE, dtype=np.float32)
        padded_state[:len(state)] = state
        action = agent.select_action(padded_state)
        action_history.append((env.current_port, action))
        state, _, done = env.step(action)

    # [2단계: 전략 실행] =================================================
    # 결정된 행동 순서를 바탕으로, '실제 규칙'에 따라 상세 경로를 생성합니다.
    
    solution = {p: [] for p in range(prob_info['P'])}
    node_allocations = np.ones(prob_info['N'], dtype=int) * -1
    car_locations = {} # car_id -> node
    cars = ShipEnv(prob_info).cars # 차량 정보 가져오기
    shortest_paths = {i: {j: get_shortest_path(env.graph, i, j) for j in range(env.num_nodes)} for i in range(env.num_nodes)}

    for port, (a_type, car_id) in action_history:
        if a_type == 'LOAD':
            # find_best_spot과 유사한 로직으로 최적 위치 탐색 (실제 규칙 적용)
            cars_on_board_ids = set(car_locations.keys())
            cars_to_leave_later = {c_id for c_id in cars_on_board_ids if cars[c_id]['dest'] > cars[car_id]['dest']}
            best_spot, max_depth, min_blocks = -1, -1, float('inf')
            empty_nodes = [i for i, s in enumerate(node_allocations) if s == -1 and i != 0]
            if not empty_nodes: continue

            for spot in empty_nodes:
                path_to_spot = shortest_paths[0].get(spot, [])
                blocks = sum(1 for c_id in cars_to_leave_later if car_locations.get(c_id) in path_to_spot)
                depth = len(path_to_spot)
                if blocks < min_blocks or (blocks == min_blocks and depth > max_depth):
                    min_blocks, max_depth, best_spot = blocks, depth, spot
            
            if best_spot != -1:
                path = shortest_paths[0][best_spot]
                solution[port].append([path, cars[car_id]['demand_id']])
                node_allocations[best_spot] = car_id
                car_locations[car_id] = best_spot
        
        elif a_type == 'UNLOAD':
            if car_id not in car_locations: continue
            start_node = car_locations[car_id]
            path_to_gate = shortest_paths[start_node][0]
            blockers = [node_allocations[n] for n in path_to_gate[1:] if node_allocations[n] != -1]

            temp_unloaded = {}
            for blk_id in blockers:
                blk_node = car_locations[blk_id]
                path_out = list(reversed(shortest_paths[blk_node][0]))
                solution[port].append([path_out, cars[blk_id]['demand_id']])
                node_allocations[blk_node] = -1
                temp_unloaded[blk_id] = blk_node

            main_unload_path = list(reversed(shortest_paths[start_node][0]))
            solution[port].append([main_unload_path, cars[car_id]['demand_id']])
            node_allocations[start_node] = -1
            del car_locations[car_id]

            for blk_id, old_node in temp_unloaded.items():
                cars_on_board_ids = set(car_locations.keys())
                cars_to_leave_later = {c_id for c_id in cars_on_board_ids if cars[c_id]['dest'] > cars[blk_id]['dest']}
                best_spot, max_depth, min_blocks = -1, -1, float('inf')
                empty_nodes = [i for i, s in enumerate(node_allocations) if s == -1 and i != 0]
                if not empty_nodes: continue
                for spot in empty_nodes:
                    path_to_spot = shortest_paths[0].get(spot, [])
                    blocks = sum(1 for c_id in cars_to_leave_later if car_locations.get(c_id) in path_to_spot)
                    depth = len(path_to_spot)
                    if blocks < min_blocks or (blocks == min_blocks and depth > max_depth):
                        min_blocks, max_depth, best_spot = blocks, depth, spot
                if best_spot != -1:
                    path_in = shortest_paths[0][best_spot]
                    solution[port].append([path_in, cars[blk_id]['demand_id']])
                    node_allocations[best_spot] = blk_id
                    car_locations[blk_id] = best_spot
                    
    return solution

# ---------------------------------------------------------------------
# 5. ⚙️ 로컬 테스트용 실행 블록 ⚙️
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # ... (이전과 동일한 로컬 테스트 코드) ...
    if len(sys.argv) == 4:
        prob_name, prob_file, timelimit = sys.argv[1], sys.argv[2], int(sys.argv[3])
        with open(prob_file, 'r') as f: prob_info = json.load(f)
        def numpy_to_python(obj):
            if isinstance(obj, (np.int64, np.int32)): return int(obj)
            if isinstance(obj, (np.float64, np.float32)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")
        try:
            start_t = time.time()
            solution = algorithm(prob_info, timelimit)
            end_t = time.time()
            checked_solution = util.check_feasibility(prob_info, solution)
            checked_solution['time'] = end_t - start_t
            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options(); opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
            print(f'Results are saved as file results.json')
            sys.exit(0)
        except Exception as e:
            print(f"An exception occurred: {repr(e)}"); sys.exit(1)
    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>"); sys.exit(2)