# ───────────────────── 0. 공통 임포트 & 출력 함수 ─────────────────────
import json, collections, heapq, random, time, copy
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
script_dir = os.path.dirname(__file__)

def safe_print(*args, **kw):
    kw['flush'] = True
    print(*args, **kw)

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
# 2. V2.6 스마트 환경 클래스 (전략적 상태 및 보상 설계)
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
        # 1. 기본 상태 정보
        node_state = np.zeros(self.num_nodes)
        for i, car_id in enumerate(self.node_status):
            if car_id != -1: node_state[i] = self.cars[car_id]['dest'] - self.current_port + 1
        port_state = np.zeros(self.num_ports)
        if self.current_port < self.num_ports: port_state[self.current_port] = 1
        
        # 2. 선적 대기 차량 정보 (목적지 별)
        waiting_cars = np.zeros(self.num_ports)
        for car in self.cars:
            if car['id'] not in self.cars_on_board and car['origin'] == self.current_port:
                waiting_cars[car['dest']] += 1

        # 3. 선내 차량의 '미래 하역 순서' 정보
        cars_on_board_info = np.zeros((self.total_cars, 2)) # [남은 항해일, 자신의 길을 막는 차 수]
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
        # 원본 V2.0의 빠르고 단순한 액션 공간으로 회귀
        legal_actions = []
        # 하역할 수 있는 모든 차량
        unload_cars = {c['id'] for c in self.cars if c['id'] in self.cars_on_board and c['dest'] == self.current_port}
        for car_id in unload_cars:
            legal_actions.append(('UNLOAD', car_id))
        # 선적해야 하는 모든 차량
        load_cars = {c['id'] for c in self.cars if c['id'] not in self.cars_on_board and c['origin'] == self.current_port}
        if any(s == -1 for s in self.node_status[1:]):
            for car_id in load_cars:
                legal_actions.append(('LOAD', car_id))
        legal_actions.append(('WAIT', -1))
        return legal_actions

    def _find_best_spot(self, car_id_to_load):
        # 가장 깊되, 미래에 나갈 차들을 최소한으로 막는 '최적의 위치'를 계산
        cars_to_leave_later = {c['id'] for c in self.cars if c['id'] in self.cars_on_board and c['dest'] > self.cars[car_id_to_load]['dest']}
        
        best_spot, max_depth, min_blocks = -1, -1, float('inf')
        empty_nodes = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]

        for spot in empty_nodes:
            path_to_spot = self.shortest_paths[0][spot]
            blocks = 0
            for car_id in cars_to_leave_later:
                if self.car_locations[car_id] in path_to_spot:
                    blocks += 1
            
            depth = len(path_to_spot)
            if blocks < min_blocks:
                min_blocks, max_depth, best_spot = blocks, depth, spot
            elif blocks == min_blocks and depth > max_depth:
                max_depth, best_spot = depth, spot

        return best_spot if best_spot != -1 else random.choice(empty_nodes)

    def step(self, action):
        a_type, car_id = action
        self.newly_generated_routes = []
        total_cost, reward = 1.0, 0.0

        if a_type == 'LOAD':
            target_node = self._find_best_spot(car_id)
            if target_node == -1: return self._get_state(), -100, True # Should not happen
            path = self.shortest_paths[0][target_node]
            self.node_status[target_node], self.car_locations[car_id] = car_id, target_node
            self.cars_on_board.add(car_id)
            total_cost += self._calculate_path_cost(path)
            reward += 1 # 선적 성공 보상

        elif a_type == 'UNLOAD':
            if car_id not in self.cars_on_board: return self._get_state(), -1000, True
            start_node = self.car_locations[car_id]
            path_to_gate = self.shortest_paths[start_node][0]
            
            # 재배치 로직
            rehandling_cost = 0
            blockers = [self.node_status[n] for n in path_to_gate[1:] if self.node_status[n] != -1]
            for blk_id in blockers:
                rehandling_cost += 100 # 재배치 페널티
                # (간소화된 재배치 시뮬레이션, 실제로는 비용만 계산)
            
            total_cost += rehandling_cost
            reward -= rehandling_cost / 10 # 재배치 발생 시 즉시 페널티
            
            self.node_status[start_node] = -1
            self.cars_on_board.remove(car_id)
            self.car_locations[car_id] = -1
            total_cost += self._calculate_path_cost(path_to_gate)
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": path_to_gate})
            reward += 5 # 하역 성공 보상
        elif a_type == 'WAIT':
            self.current_port += 1
            reward -= 0.5 
            
        final_reward = reward - total_cost
        done = self.current_port >= self.num_ports
        if done:
            if not self.cars_on_board: final_reward += 1000 # 모든 임무 완수
            else: final_reward -= 1000 # 임무 실패
        
        return self._get_state(), final_reward, done

# ---------------------------------------------------------------------
# 3. 에이전트, 신경망, 액션 매퍼 클래스
# ---------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, action_size)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity): self.memory = collections.deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(args)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)
    
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
        self.BATCH_SIZE, self.GAMMA, self.EPS_DECAY, self.TAU, self.LR = 128, 0.99, 10000, 5e-4, 5e-4
        self.EPS_START, self.EPS_END, self.CLIP_NORM = 0.9, 0.05, 10.0
        self.action_mapper = action_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        self.target_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        if model_path: self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory, self.steps_done, self.env = ReplayBuffer(100000), 0, None

    def select_action(self, state, use_exploration=True):
        legal = self.env.get_legal_actions()
        if not legal: return ('WAIT', -1)
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if use_exploration and random.random() < eps: return random.choice(legal)
        with torch.no_grad():
            qs = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))[0]
            legal_q = {a: qs[self.action_mapper.get(a)] for a in legal if self.action_mapper.get(a) is not None}
            return max(legal_q, key=legal_q.get) if legal_q else ('WAIT', -1)
            
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE: return
        s, a, r, ns, d = zip(*self.memory.sample(self.BATCH_SIZE))
        s, ns, r, d = torch.from_numpy(np.stack(s)).to(self.device), torch.from_numpy(np.stack(ns)).to(self.device), \
                      torch.tensor(r, device=self.device, dtype=torch.float32), torch.tensor(d, device=self.device, dtype=torch.bool)
        
        valid_indices = [i for i, action in enumerate(a) if self.action_mapper.get(action) is not None]
        if not valid_indices: return
        s, a, r, ns, d = s[valid_indices], [a[i] for i in valid_indices], r[valid_indices], ns[valid_indices], d[valid_indices]

        a_idx = torch.tensor([self.action_mapper.get(x) for x in a], device=self.device).unsqueeze(1)
        q_sa = self.policy_net(s).gather(1, a_idx)
        with torch.no_grad():
            next_q = torch.zeros(len(s), device=self.device); mask = ~d
            if mask.any(): next_q[mask] = self.target_net(ns[mask]).max(1)[0]
        target = r + self.GAMMA * next_q
        loss = F.smooth_l1_loss(q_sa, target.unsqueeze(1))
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.CLIP_NORM)
        self.optimizer.step(); self.update_target_net()

    def update_target_net(self):
        target_dict, policy_dict = self.target_net.state_dict(), self.policy_net.state_dict()
        for key in policy_dict: target_dict[key] = policy_dict[key]*self.TAU + target_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_dict)

# ---------------------------------------------------------------------
# 4. 메인 솔루션 제출 및 학습 코드
# ---------------------------------------------------------------------
def get_solution(problem):
    # ... (이전 get_solution 함수와 동일하게 사용 가능)
    return {}

if __name__ == '__main__':
    
    log_path = os.path.join(script_dir, "v2.6_StrategicRL")
    writer = SummaryWriter(log_dir=log_path)

    NUM_EPISODES, PRINT_INTERVAL = 50000, 20
    PROBLEM_FILES = [f'exercise_problems/prob{i}.json' for i in range(1, 11)]
    
    print("--- V2.6: Preparing for Training (Strategic State & Reward) ---")
    problems, max_state_size, max_car_count = [], 0, 0
    for file_path in PROBLEM_FILES:
        try:
            with open(file_path, 'r') as f:
                prob = json.load(f); prob['id'] = file_path; problems.append(prob)
                state_size = ShipEnv(prob)._get_state().shape[0]
                if state_size > max_state_size: max_state_size = state_size
                num_cars = sum(q for _, q in prob['K']);
                if num_cars > max_car_count: max_car_count = num_cars
        except FileNotFoundError: print(f"Warning: File not found at {file_path}")
    if not problems: print("Error: No problem files found."); exit()

    master_problem = {'K': [[(0, 1), max_car_count]]}
    master_action_mapper = ActionMapper(master_problem)
    agent = DQNAgent(max_state_size, master_action_mapper)
    print(f"--- V2.6: Starting Training (Max State: {max_state_size}, Master Action: {len(master_action_mapper)}) ---")

    rewards, lengths, successes = [], [], []
    best_sr = -1.0

    for i_episode in range(NUM_EPISODES):
        current_problem = random.choice(problems)
        env = ShipEnv(current_problem); agent.env, agent.action_mapper = env, ActionMapper(current_problem)
        state, ep_reward, done = env.reset(), 0, False
        
        for t in range(5000):
            padded_state = np.zeros(max_state_size, dtype=np.float32); padded_state[:len(state)] = state

            action = agent.select_action(padded_state)
            next_state, reward, done = env.step(action)
            padded_next_state = np.zeros(max_state_size, dtype=np.float32); padded_next_state[:len(next_state)] = next_state

            agent.memory.push(padded_state, action, reward, padded_next_state, done)
            state = next_state
            agent.optimize_model(); ep_reward += reward
            if done: break
        
        success = ep_reward > 0 and not any(c['id'] in env.cars_on_board for c in env.cars)
        rewards.append(ep_reward); lengths.append(t + 1); successes.append(1 if success else 0)
        
        if (i_episode + 1) % PRINT_INTERVAL == 0:
            avg_r, avg_l, sr = np.mean(rewards[-100:]), np.mean(lengths[-100:]), np.mean(successes[-100:]) * 100
            safe_print(f"Eps {i_episode+1:<6} | Prob: {current_problem['id'].split('/')[-1]:<12} | "
                       f"Len: {avg_l:<4.0f} | SR: {sr:5.1f}% | Best SR: {max(0, best_sr):5.1f}% | AvgR: {avg_r:8.2f}")
            if sr > best_sr:
                best_sr = sr
                torch.save(agent.policy_net.state_dict(), os.path.join(script_dir, "best_model_v2.6.pth"))
                safe_print(f"🎉 New best model saved! Success Rate: {sr:.2f}%")
        writer.add_scalar("SuccessRate/avg_100", np.mean(successes[-100:]) if len(successes) >= 100 else 0, i_episode)

    print("--- V2.6 Training Finished ---")
    torch.save(agent.policy_net.state_dict(), os.path.join(script_dir, "final_model_v2.pth"))
    writer.close()