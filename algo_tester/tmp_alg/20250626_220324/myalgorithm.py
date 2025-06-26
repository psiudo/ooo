import os
import json
import collections
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util
# =====================================================================
# 1. 헬퍼 함수 및 환경/에이전트 클래스 (사용자 코드 기반)
# =====================================================================

def get_shortest_path(graph, start, end):
    """주어진 그래프에서 두 노드 간의 최단 경로를 찾습니다."""
    if start == end:
        return [start]
    q = collections.deque([(start, [start])])
    visited = {start}
    while q:
        curr, path = q.popleft()
        if curr == end:
            return path
        for nbr in graph.get(curr, []):
            if nbr not in visited:
                visited.add(nbr)
                q.append((nbr, path + [nbr]))
    return None

class ShipEnv:
    """강화학습을 위한 선박 환경 시뮬레이터 클래스"""
    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.num_nodes, self.edges, self.num_ports, self.fixed_cost = \
            problem_data['N'], problem_data['E'], problem_data['P'], problem_data['F']
        self.graph = collections.defaultdict(list)
        for u, v in self.edges:
            self.graph[u].append(v)
            self.graph[v].append(u)
        self.shortest_paths = {i: {j: get_shortest_path(self.graph, i, j) for j in range(self.num_nodes)} for i in range(self.num_nodes)}
        self.cars = []
        car_id_counter = 0
        for i, ((o, d), q) in enumerate(problem_data['K']):
            for _ in range(q):
                self.cars.append({'id': car_id_counter, 'demand_id': i, 'origin': o, 'dest': d})
                car_id_counter += 1
        self.total_cars = len(self.cars)
        self.reset()

    def _scale_reward(self, raw):
        return np.sign(raw) * np.sqrt(abs(raw))

    def reset(self):
        self.current_port, self.node_status, self.car_locations = 0, [-1] * self.num_nodes, [-1] * self.total_cars
        self.cars_on_board, self.newly_generated_routes = set(), []
        return self._get_state()
        
    def _get_state(self):
        node_state = np.zeros(self.num_nodes)
        port_state = np.zeros(self.num_ports)
        waiting_state = np.zeros(self.num_ports)
        urgency_state = np.zeros(self.total_cars)
        blocker_state = np.zeros(self.total_cars)
        for i, car_id in enumerate(self.node_status):
            if car_id != -1:
                node_state[i] = self.cars[car_id]['dest'] + 1
        if self.current_port < self.num_ports:
            port_state[self.current_port] = 1
        for car in self.cars:
            if car['id'] not in self.cars_on_board and car['origin'] == self.current_port:
                waiting_state[car['dest']] += 1
        for car_id in self.cars_on_board:
            urgency_state[car_id] = self.cars[car_id]['dest'] - self.current_port
            if self.car_locations[car_id] != -1:
                path = self.shortest_paths.get(self.car_locations[car_id], {}).get(0)
                if path:
                    blocker_state[car_id] = sum(1 for node in path[1:] if self.node_status[node] != -1)
        
        # 상태 벡터들을 하나의 벡터로 결합
        # 주의: 학습 시 사용된 max_state_size에 맞춰 패딩이 필요할 수 있으나, 
        # 에이전트가 처리하므로 여기서는 생성만 담당합니다.
        state_parts = [node_state, port_state, waiting_state, urgency_state, blocker_state]
        return np.concatenate(state_parts).astype(np.float32)


    def _calculate_path_cost(self, path):
        if not path or len(path) <= 1:
            return 0
        return self.fixed_cost + (len(path) - 1)
        
    def get_legal_actions(self):
        legal_actions = []
        for car_id in self.cars_on_board:
            if self.cars[car_id]['dest'] == self.current_port:
                legal_actions.append(('UNLOAD', car_id, -1))
        cars_to_load = [c['id'] for c in self.cars if c['origin'] == self.current_port and c['id'] not in self.cars_on_board]
        if any(self.node_status[i] == -1 for i in range(1, self.num_nodes)):
            for car_id in cars_to_load:
                legal_actions.append(('LOAD', car_id, -1))
        legal_actions.append(('WAIT', -1, -1))
        return legal_actions

    def step(self, action):
        self.newly_generated_routes = []
        a_type, car_id, _ = action
        total_cost = 1.0
        if a_type == 'LOAD':
            empties = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]
            if not empties:
                return self._get_state(), -100.0, False
            tgt = max(empties, key=lambda n: len(self.shortest_paths[0][n]))
            path = self.shortest_paths[0][tgt]
            self.node_status[tgt], self.car_locations[car_id] = car_id, tgt
            self.cars_on_board.add(car_id)
            total_cost += self._calculate_path_cost(path)
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": path})
        elif a_type == 'UNLOAD':
            if car_id not in self.cars_on_board:
                return self._get_state(), -1000.0, False
            start = self.car_locations[car_id]
            gate_p = self.shortest_paths[start][0]
            re_cnt = 0
            while re_cnt < self.total_cars:
                blockers = [(self.node_status[n], n) for n in gate_p[1:] if self.node_status[n] != -1]
                if not blockers:
                    break
                blk_id, blk_node = blockers[0]
                m_type, paths = self._find_best_relocation(blk_id, gate_p)
                if m_type == 'RELOCATE':
                    self.node_status[blk_node], self.node_status[paths[-1]] = -1, blk_id
                    self.car_locations[blk_id] = paths[-1]
                    total_cost += self._calculate_path_cost(paths) + 50.0
                    self.newly_generated_routes.append({"demand_id": self.cars[blk_id]["demand_id"], "route": paths})
                else:
                    p_out, _ = paths
                    total_cost += self._calculate_path_cost(p_out) * 2 + 100.0
                    self.node_status[blk_node] = -1
                    empties = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]
                    if not empties:
                        return self._get_state(), -1000.0, True
                    new_node = max(empties, key=lambda n: len(self.shortest_paths[0][n]))
                    p_new = self.shortest_paths[0][new_node]
                    self.node_status[new_node], self.car_locations[blk_id] = blk_id, new_node
                    self.newly_generated_routes.extend([
                        {"demand_id": self.cars[blk_id]["demand_id"], "route": p_out},
                        {"demand_id": self.cars[blk_id]["demand_id"], "route": p_new}
                    ])
                re_cnt += 1
            self.node_status[start], self.car_locations[car_id] = -1, -1
            self.cars_on_board.remove(car_id)
            total_cost += self._calculate_path_cost(gate_p)
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": gate_p})
        elif a_type == 'WAIT':
            self.current_port += 1
        
        raw_r = -total_cost
        done = self.current_port >= self.num_ports
        if done:
            ok = len(self.cars_on_board) == 0 and all(loc == -1 for loc in self.car_locations)
            raw_r += 1000.0 if ok else -500.0
            
        return self._get_state(), float(self._scale_reward(raw_r)), done

    def _find_best_relocation(self, blocker_id, path_to_clear):
        start_node = self.car_locations[blocker_id]
        best_cost, best_path = float('inf'), None
        for empty in [i for i, s in enumerate(self.node_status) if s == -1 and i != 0 and i not in path_to_clear]:
            path = self.shortest_paths.get(start_node, {}).get(empty)
            if path:
                cost = self._calculate_path_cost(path)
                if cost < best_cost:
                    best_cost, best_path = cost, path
        
        p_out = self.shortest_paths.get(start_node, {}).get(0)
        
        if p_out and best_path and best_cost <= self._calculate_path_cost(p_out) * 2:
            return 'RELOCATE', best_path
        
        return 'TEMP_UNLOAD', [p_out, list(reversed(p_out))] if p_out else ('RELOCATE', best_path)

class PolicyNetwork(nn.Module):
    """DQN을 위한 신경망 모델"""
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayBuffer:
    """리플레이 버퍼 (추론 시에는 사용되지 않음)"""
    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(args)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class ActionMapper:
    """행동과 인덱스를 매핑하는 클래스"""
    def __init__(self, problem):
        self.action_to_idx, self.idx_to_action = {}, []
        self._add(('WAIT', -1, -1))
        num_cars = sum(q for _, q in problem['K'])
        for car_id in range(num_cars):
            self._add(('UNLOAD', car_id, -1))
            self._add(('LOAD', car_id, -1))
            
    def _add(self, action):
        if action not in self.action_to_idx:
            self.idx_to_action.append(action)
            self.action_to_idx[action] = len(self.idx_to_action) - 1
            
    def get(self, action):
        return self.action_to_idx.get(action)
    
    def __len__(self):
        return len(self.idx_to_action)

class DQNAgent:
    """DQN 에이전트 클래스"""
    def __init__(self, state_size, action_mapper, model_path=None):
        self.BATCH_SIZE, self.GAMMA, self.EPS_DECAY, self.TAU, self.LR = 64, 0.99, 20000, 1e-3, 1e-4
        self.EPS_START, self.EPS_END, self.CLIP_NORM = 0.9, 0.02, 1.0
        self.action_mapper = action_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        self.target_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        
        if model_path:
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory, self.steps_done, self.env = ReplayBuffer(50000), 0, None

    def select_action(self, state, use_exploration=True):
        legal = self.env.get_legal_actions()
        if not legal:
            return None
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        
        if use_exploration and random.random() < eps:
            return random.choice(legal)
            
        with torch.no_grad():
            qs = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))[0]
            legal_q = {a: qs[self.action_mapper.get(a)] for a in legal if self.action_mapper.get(a) is not None and self.action_mapper.get(a) < len(qs)}
            if not legal_q:
                return random.choice(legal) if legal else None
            return max(legal_q, key=legal_q.get)

    # optimize_model과 update_target_net은 추론 시에 호출되지 않으므로 그대로 둡니다.
    def optimize_model(self):
        pass
    def update_target_net(self):
        pass


# =====================================================================
# 2. 최종 제출용 알고리즘 함수
# =====================================================================
def algorithm(prob_info, timelimit=60):
    """
    대회 제출용 메인 알고리즘 함수입니다.
    DQNAgent를 사용하여 문제를 해결하고, 채점 형식에 맞는 솔루션을 반환합니다.
    """
    start_time = time.time()

    # --- 모델 파일 경로 설정 ---
    try:
        # __file__은 현재 실행 중인 스크립트의 경로를 나타냅니다.
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Jupyter Notebook과 같은 환경에서는 __file__이 정의되지 않으므로 현재 작업 디렉토리를 사용합니다.
        script_dir = os.getcwd()

    MODEL_PATH = os.path.join(script_dir, "best_model_v2.pth")
    
    # --- 모델 파일 존재 여부 확인 ---
    if not os.path.exists(MODEL_PATH):
        # 모델 파일이 없으면 비어있는 솔루션을 반환합니다.
        print(f"Error: Model file not found at {MODEL_PATH}")
        return {p: [] for p in range(prob_info['P'])}

    # --- 환경 및 에이전트 초기화 ---
    # prob_info가 사용자의 코드에서 'problem' 딕셔너리에 해당합니다.
    env = ShipEnv(prob_info)

    # 상태 및 액션 공간 크기 계산
    # 여러 문제에 대응하기 위해 최대 크기를 가정하여 에이전트를 생성합니다.
    # 이는 학습 코드의 방식과 일치시킵니다.
    # 최대 노드, 포트, 차량 수를 기반으로 state_size를 넉넉하게 잡습니다.
    # 실제로는 학습 시 사용한 max_state_size를 알아야 하지만, 여기서는 일반화합니다.
    max_nodes = 100 # 예시 최대값
    max_ports = 20  # 예시 최대값
    max_cars_total = 100 # 예시 최대값
    # state: node_state, port_state, waiting_state, urgency_state, blocker_state
    max_state_size = max_nodes + max_ports + max_ports + max_cars_total + max_cars_total
    
    # 마스터 액션 매퍼 생성
    # 가장 큰 액션 공간을 가질 수 있는 문제 기준으로 매퍼 생성
    max_cars_for_mapper = sum(q for _, q in prob_info['K'])
    master_problem_for_mapper = {'K': [[(0, 1), max_cars_for_mapper]]}
    action_mapper = ActionMapper(master_problem_for_mapper)
    
    agent = DQNAgent(max_state_size, action_mapper, model_path=MODEL_PATH)
    agent.env = env
    # 현재 문제에 맞는 액션 매퍼로 교체
    agent.action_mapper = ActionMapper(prob_info)
    
    # --- 솔루션 생성 루프 ---
    solution_routes_from_env = {p: [] for p in range(prob_info['P'])}
    state = env.reset()
    done = False
    
    while not done:
        # 타임리밋 확인
        if time.time() - start_time > timelimit - 2: # 2초의 여유시간
            break

        # 상태 벡터를 신경망 입력 크기에 맞게 패딩
        padded_state = np.zeros(max_state_size, dtype=np.float32)
        padded_state[:len(state)] = state
        
        action = agent.select_action(padded_state, use_exploration=False)
        if action is None:
            action = ('WAIT', -1, -1) # 수행할 액션이 없으면 WAIT
        
        next_state, reward, done = env.step(action)
        
        # 현재 포트는 액션이 수행된 시점의 포트
        port_for_route = env.current_port - 1 if action[0] == 'WAIT' else env.current_port
        
        if env.newly_generated_routes and port_for_route >= 0:
            solution_routes_from_env[port_for_route].extend(env.newly_generated_routes)
        
        state = next_state
        
    # --- 출력 형식 변환 (중요) ---
    # env에서 나온 결과 `{"demand_id": ..., "route": ...}`를
    # 채점기가 요구하는 `[route, demand_id]` 형식으로 변환합니다.
    final_solution = {p: [] for p in range(prob_info['P'])}
    for port, routes in solution_routes_from_env.items():
        for route_info in routes:
            demand_id = route_info['demand_id']
            route_path = route_info['route']
            # 거꾸로 된 경로가 필요하다면 여기서 `list(reversed(route_path))` 사용
            # 현재 로직에서는 env에서 생성된 경로를 그대로 사용
            final_solution[port].append([route_path, demand_id])
            
    return final_solution

# =====================================================================
# 3. 로컬 테스트를 위한 실행 블록 (원본 코드에서 복원)
# =====================================================================
if __name__ == "__main__":
    # 이 블록은 'python myalgorithm.py ...' 와 같이 직접 실행될 때만 작동합니다.
    # 대회 서버에서는 import해서 사용하므로 이 부분은 실행되지 않습니다.

    import sys
    import jsbeautifier # 이 라이브러리가 필요합니다.

    def numpy_to_python(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)  
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        raise TypeError(f"Type {type(obj)} not serializable")
    
    
    # Arguments list should be problem_name, problem_file, timelimit (in seconds)
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r') as f:
            prob_info = json.load(f)

        exception = None
        solution = None

        try:
            alg_start_time = time.time()

            # Run algorithm!
            solution = algorithm(prob_info, timelimit)

            alg_end_time = time.time()

            # util.py의 check_feasibility 함수를 사용하여 결과 검증
            checked_solution = util.check_feasibility(prob_info, solution)

            checked_solution['time'] = alg_end_time - alg_start_time
            checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 2 # allowing additional 2 second!
            checked_solution['exception'] = exception
            checked_solution['prob_name'] = prob_name
            checked_solution['prob_file'] = prob_file


            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
                print(f'Results are saved as file results.json')
                
            sys.exit(0)

        except Exception as e:
            print(f"Exception: {repr(e)}")
            sys.exit(1)

    else:
        print("Usage: python myalgorithm.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)