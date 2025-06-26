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
# 2. V2.5 스마트 환경 클래스 (원본 V2.0 기반 + 버그 수정)
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
                self.cars.append({'id': car_id_counter, 'demand_id': i, 'origin': o, 'dest': d})
                car_id_counter += 1
        self.total_cars = len(self.cars)
        self.reset()

    def _scale_reward(self, raw): return np.sign(raw) * np.sqrt(abs(raw))

    def reset(self):
        self.current_port, self.node_status, self.car_locations = 0, [-1] * self.num_nodes, [-1] * self.total_cars
        self.cars_on_board, self.newly_generated_routes = set(), []
        return self._get_state()
        
    def _get_state(self):
        node_state = np.zeros(self.num_nodes); port_state = np.zeros(self.num_ports)
        waiting_state = np.zeros(self.num_ports); urgency_state = np.zeros(self.total_cars)
        blocker_state = np.zeros(self.total_cars)
        for i, car_id in enumerate(self.node_status):
            if car_id != -1: node_state[i] = self.cars[car_id]['dest'] + 1
        if self.current_port < self.num_ports: port_state[self.current_port] = 1
        for car in self.cars:
            if car['id'] not in self.cars_on_board and car['origin'] == self.current_port:
                waiting_state[car['dest']] += 1
        for car_id in self.cars_on_board:
            urgency_state[car_id] = self.cars[car_id]['dest'] - self.current_port
            if self.car_locations[car_id] != -1:
                path = self.shortest_paths.get(self.car_locations[car_id], {}).get(0)
                if path: blocker_state[car_id] = sum(1 for node in path[1:] if self.node_status[node] != -1)
        return np.concatenate([node_state, port_state, waiting_state, urgency_state, blocker_state]).astype(np.float32)

    def _calculate_path_cost(self, path):
        if not path or len(path) <= 1: return 0
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
            raw_r += 5.0 # LOAD 성공 시 작은 보상
            empties = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]
            if not empties: return self._get_state(), -100.0, False
            tgt = max(empties, key=lambda n: len(self.shortest_paths[0][n]))
            path = self.shortest_paths[0][tgt]
            self.node_status[tgt], self.car_locations[car_id] = car_id, tgt
            self.cars_on_board.add(car_id); total_cost += self._calculate_path_cost(path)
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": path})

        elif a_type == 'UNLOAD':
            raw_r += 5.0 # UNLOAD 성공 시 작은 보상
            if car_id not in self.cars_on_board: return self._get_state(), -1000.0, False
            start = self.car_locations[car_id]; gate_p = self.shortest_paths[start][0]
            re_cnt = 0
            while re_cnt < self.total_cars:
                blockers = [(self.node_status[n], n) for n in gate_p[1:] if self.node_status[n] != -1]
                if not blockers: break
                blk_id, blk_node = blockers[0]
                m_type, paths = self._find_best_relocation(blk_id, gate_p)
                if m_type == 'RELOCATE':
                    self.node_status[blk_node], self.node_status[paths[-1]] = -1, blk_id
                    self.car_locations[blk_id] = paths[-1]; total_cost += self._calculate_path_cost(paths) + 50.0
                    self.newly_generated_routes.append({"demand_id": self.cars[blk_id]["demand_id"], "route": paths})
                else:
                    p_out, _ = paths; total_cost += self._calculate_path_cost(p_out) * 2 + 100.0
                    self.node_status[blk_node] = -1
                    empties = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]
                    if not empties: return self._get_state(), -1000.0, True
                    new_node = max(empties, key=lambda n: len(self.shortest_paths[0][n]))
                    p_new = self.shortest_paths[0][new_node]
                    self.node_status[new_node], self.car_locations[blk_id] = blk_id, new_node
                    self.newly_generated_routes.extend([{"demand_id": self.cars[blk_id]["demand_id"], "route": p_out},
                                                        {"demand_id": self.cars[blk_id]["demand_id"], "route": p_new}])
                re_cnt += 1
            self.node_status[start], self.car_locations[car_id] = -1, -1
            self.cars_on_board.remove(car_id); total_cost += self._calculate_path_cost(gate_p)
            self.newly_generated_routes.append({"demand_id": self.cars[car_id]["demand_id"], "route": gate_p})
        elif a_type == 'WAIT':
            cars_to_load = [c['id'] for c in self.cars if c['origin'] == self.current_port and c['id'] not in self.cars_on_board]
            if len(cars_to_load) > 0:
                raw_r -= 10.0 # 해야 할 일이 있는데 기다리면 페널티
        raw_r = -total_cost
        done = self.current_port >= self.num_ports

        if done:
            cars_left_on_board = len(self.cars_on_board)
            cars_not_delivered = sum(1 for loc in self.car_locations if loc != -1 and loc not in self.cars_on_board) # 혹시 모를 예외 처리
            total_failed_cars = cars_left_on_board + cars_not_delivered

            if total_failed_cars == 0:
                raw_r += 1000.0 # 완벽한 성공
            else:
                # 실패한 차 한대당 -1000점씩 큰 페널티를 부과
                raw_r -= float(total_failed_cars * 1000)

        return self._get_state(), float(self._scale_reward(raw_r)), done

    def _find_best_relocation(self, blocker_id, path_to_clear):
        start_node = self.car_locations[blocker_id]
        best_cost, best_path = float('inf'), None
        for empty in [i for i, s in enumerate(self.node_status) if s == -1 and i != 0 and i not in path_to_clear]:
            path = self.shortest_paths.get(start_node, {}).get(empty)
            if path and self._calculate_path_cost(path) < best_cost:
                best_cost, best_path = self._calculate_path_cost(path), path
        p_out = self.shortest_paths.get(start_node, {}).get(0)
        if p_out and best_path and best_cost <= self._calculate_path_cost(p_out) * 2: return 'RELOCATE', best_path
        return 'TEMP_UNLOAD', [p_out, list(reversed(p_out))] if p_out else ('RELOCATE', best_path)

# ---------------------------------------------------------------------
# 3. 에이전트, 신경망, 액션 매퍼 클래스
# ---------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 256); self.layer2 = nn.Linear(256, 256); self.layer3 = nn.Linear(256, action_size)
    def forward(self, x):
        x = F.relu(self.layer1(x)); x = F.relu(self.layer2(x)); return self.layer3(x)

class ReplayBuffer:
    def __init__(self, capacity): self.memory = collections.deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(args)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)
    
class ActionMapper:
    def __init__(self, problem):
        self.action_to_idx, self.idx_to_action = {}, []
        self._add(('WAIT', -1, -1))
        num_cars = sum(q for _, q in problem['K'])
        for car_id in range(num_cars):
            self._add(('UNLOAD', car_id, -1)); self._add(('LOAD', car_id, -1))
    def _add(self, action):
        if action not in self.action_to_idx:
            self.idx_to_action.append(action); self.action_to_idx[action] = len(self.idx_to_action) - 1
    def get(self, action): return self.action_to_idx.get(action)
    def __len__(self): return len(self.idx_to_action)

class DQNAgent:
    def __init__(self, state_size, action_mapper, model_path=None):
        self.BATCH_SIZE, self.GAMMA, self.EPS_DECAY, self.TAU, self.LR = 64, 0.99, 20000, 1e-3, 1e-4
        self.EPS_START, self.EPS_END, self.CLIP_NORM = 0.9, 0.02, 1.0
        self.action_mapper = action_mapper
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        self.target_net = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        if model_path: self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory, self.steps_done, self.env = ReplayBuffer(50000), 0, None

    def select_action(self, state, use_exploration=True):
        legal = self.env.get_legal_actions()
        if not legal: return None
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if use_exploration and random.random() < eps: return random.choice(legal)
        with torch.no_grad():
            qs = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))[0]
            legal_q = {a: qs[self.action_mapper.get(a)] for a in legal if self.action_mapper.get(a) is not None and self.action_mapper.get(a) < len(qs)}
            return max(legal_q, key=legal_q.get) if legal_q else (random.choice(legal) if legal else None)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE: return
        transitions = self.memory.sample(self.BATCH_SIZE)
        s, a, r, ns, d = zip(*transitions)
        
        valid_indices = [i for i, action in enumerate(a) if self.action_mapper.get(action) is not None]
        if len(valid_indices) < 1: return
        
        s = torch.from_numpy(np.stack([s[i] for i in valid_indices])).to(self.device)
        a = [a[i] for i in valid_indices]
        r = torch.tensor([r[i] for i in valid_indices], device=self.device, dtype=torch.float32)
        ns = torch.from_numpy(np.stack([ns[i] for i in valid_indices])).to(self.device)
        d = torch.tensor([d[i] for i in valid_indices], device=self.device, dtype=torch.bool)

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
# 4. 메인 솔루션 제출 함수 (✨ 이름 변경 및 경로 수정)
# ---------------------------------------------------------------------
def algorithm(problem):
    start_time = time.time()
    script_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(script_dir, "best_model_v2.pth")
    
    if not os.path.exists(MODEL_PATH):
        # 모델 파일이 없을 경우에 대한 예외 처리 (빈 결과 반환)
        return {p: [] for p in range(problem['P'])}

    env = ShipEnv(problem)
    state_size = env._get_state().shape[0]
    action_mapper = ActionMapper(problem) 
    agent = DQNAgent(state_size, action_mapper, model_path=MODEL_PATH)
    agent.env = env
    
    solution_routes = {p: [] for p in range(problem['P'])}
    state = env.reset()
    done = False
    
    while not done:
        if time.time() - start_time > 580: break
        action = agent.select_action(state, use_exploration=False)
        if action is None: action = ('WAIT', -1, -1)
        
        next_state, reward, done = env.step(action)
        
        port_for_route = env.current_port -1 if action[0] == 'WAIT' else env.current_port
        
        if env.newly_generated_routes and port_for_route >= 0:
            solution_routes[port_for_route].extend(env.newly_generated_routes)
        
        state = next_state
        
    return solution_routes

# ---------------------------------------------------------------------
# 5. 모델 학습을 위한 실행 코드
# ---------------------------------------------------------------------
if __name__ == '__main__':
    log_path = os.path.join(script_dir, "v2.5_StableOriginal")
    writer = SummaryWriter(log_dir=log_path)
    NUM_EPISODES, PRINT_INTERVAL = 30000, 20
    PROBLEM_FILES = [f'exercise_problems/prob{i}.json' for i in range(1, 11)]
    
    print("--- V2.5: Preparing for Training (Stable Original Strategy) ---")
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
    print(f"--- V2.5: Starting Training ---")
    print(f"Max State: {max_state_size}, Master Action Size: {len(master_action_mapper)}")

    rewards, lengths, successes = [], [], []
    best_sr = -1.0

    for i_episode in range(NUM_EPISODES):
        current_problem = random.choice(problems)
        env = ShipEnv(current_problem); agent.env = env
        agent.action_mapper = ActionMapper(current_problem)
        state, ep_reward = env.reset(), 0
        
        for t in range(5000):
            padded_state = np.zeros(max_state_size, dtype=np.float32)
            padded_state[:len(state)] = state
            action = agent.select_action(padded_state, use_exploration=True)
            if action is None: action = ('WAIT', -1, -1)
            
            next_state, reward, done = env.step(action)
            
            padded_next_state = np.zeros(max_state_size, dtype=np.float32)
            padded_next_state[:len(next_state)] = next_state
            
            agent.memory.push(padded_state, action, reward, padded_next_state, done)
            state = next_state
            agent.optimize_model()
            ep_reward += reward
            if done: break
        
        rewards.append(ep_reward); lengths.append(t + 1); successes.append(1 if reward > 0 else 0)
        
        if (i_episode + 1) % PRINT_INTERVAL == 0:
            safe_print(f"Eps {i_episode+1}/{NUM_EPISODES} | Prob: {current_problem['id'].split('/')[-1]:<15} | Len: {t+1:<4} | Res: {'S' if reward > 0 else 'F'}", end='')
            if len(rewards) >= 100:
                avg_r, sr = np.mean(rewards[-100:]), np.mean(successes[-100:]) * 100.0
                safe_print(f" | AvgR: {avg_r:<8.2f} | SR: {sr:5.1f}% | Best: {max(0, best_sr):.1f}%")
                if sr > best_sr:
                    best_sr = sr
                    torch.save(agent.policy_net.state_dict(), os.path.join(script_dir, "best_model_v2.pth"))

                    safe_print(f"🎉 New best model saved! Success Rate: {sr:.2f}%")
            else:
                print()

    print("--- V2.5 Training Finished ---")
    torch.save(agent.policy_net.state_dict(), os.path.join(script_dir, "final_model_v2.pth"))

    if best_sr > -1.0:
        print(f"Best ever model was saved to best_model_v2.pth with a success rate of {best_sr:.2f}%")
    writer.close()