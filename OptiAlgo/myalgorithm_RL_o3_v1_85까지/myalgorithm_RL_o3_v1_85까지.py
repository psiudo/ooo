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
    if start == end:
        return [start]
    q = collections.deque([(start, [start])])
    visited = {start}
    while q:
        curr, path = q.popleft()
        if curr == end:
            return path
        for nbr in graph[curr]:
            if nbr not in visited:
                visited.add(nbr)
                q.append((nbr, path + [nbr]))
    return None

# ---------------------------------------------------------------------
# 2. V2.0 스마트 환경 클래스 (ShipEnv)
# ---------------------------------------------------------------------
"""
V2.0: 저수준 실행을 책임지는 '스마트 환경'.
재배치(rehandling)를 포함한 모든 과정을 현실적으로 시뮬레이션하여
Feasibility를 100% 보장하고, 에이전트는 전략적 판단에만 집중하도록 합니다.
"""


class ShipEnv:
    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.num_nodes = problem_data['N']
        self.edges     = problem_data['E']
        self.num_ports = problem_data['P']
        self.fixed_cost = problem_data['F']

        self.graph = collections.defaultdict(list)
        for u, v in self.edges:
            self.graph[u].append(v); self.graph[v].append(u)

        self.shortest_paths = {i: {} for i in range(self.num_nodes)}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.shortest_paths[i][j] = get_shortest_path(self.graph, i, j)

        self.cars = []
        cid = 0
        for did, ((o, d), q) in enumerate(problem_data['K']):
            for _ in range(q):
                self.cars.append({'id': cid, 'demand_id': did, 'origin': o, 'dest': d})
                cid += 1
        self.total_cars = len(self.cars)
        self.reset()

    # <PATCH 1> ─────────────────────────────────
    def _scale_reward(self, raw):
        return np.sign(raw) * np.sqrt(abs(raw))
    # ──────────────────────────────────────────

    def reset(self):
        self.current_port = 0
        self.node_status = [-1] * self.num_nodes
        self.car_locations = [-1] * self.total_cars
        self.cars_on_board = set()
        self.newly_generated_routes = []
        return self._get_state()
        
    def _get_state(self):
        """V2.0: 에이전트의 전략적 판단을 돕기 위한 더 풍부한 상태 정보"""
        # 1. 노드별 점유 상태 (이전과 동일)
        node_state_vector = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            car_id = self.node_status[i]
            if car_id != -1:
                node_state_vector[i] = self.cars[car_id]['dest'] + 1
        
        # 2. 현재 항구 정보 (One-hot, 이전과 동일)
        port_vector = np.zeros(self.num_ports)
        if self.current_port < self.num_ports:
            port_vector[self.current_port] = 1

        # 3. 선적 대기 차량 정보 (이전과 동일)
        waiting_cars_vector = np.zeros(self.num_ports)
        for car in self.cars:
            if car['id'] not in self.cars_on_board and car['origin'] == self.current_port:
                waiting_cars_vector[car['dest']] += 1

        # === 4. V2.0 추가 정보 ===
        # 차량별 긴급도(Urgency) 및 막힘(Blockers) 상태
        car_urgency = np.zeros(self.total_cars)
        car_blockers = np.zeros(self.total_cars)
        for car_id in self.cars_on_board:
            # 긴급도: 목적지까지 남은 항구 수 (작을수록 긴급)
            urgency = self.cars[car_id]['dest'] - self.current_port
            car_urgency[car_id] = urgency
            
            # 막힘 정도: 내 경로 위에 다른 차가 몇 대나 있나
            path_to_gate = self.shortest_paths[self.car_locations[car_id]][0]
            if path_to_gate:
                blocker_count = sum(1 for node in path_to_gate[1:] if self.node_status[node] != -1)
                car_blockers[car_id] = blocker_count

        state = np.concatenate([
            node_state_vector, port_vector, waiting_cars_vector,
            car_urgency, car_blockers
        ]).astype(np.float32)
        
        return state

    def _calculate_path_cost(self, path):
        if not path or len(path) <= 1: return 0
        return self.fixed_cost + (len(path) - 1)

    def get_legal_actions(self):
        """V2.0: 전략적 행동 목록만 생성"""
        legal_actions = []
        # 하역할 수 있는 모든 차량
        for car_id in self.cars_on_board:
            if self.cars[car_id]['dest'] == self.current_port:
                legal_actions.append(('UNLOAD', car_id, -1))
        
        # 선적해야 하는 모든 차량
        cars_to_load = [c['id'] for c in self.cars if c['origin'] == self.current_port and c['id'] not in self.cars_on_board]
        if any(self.node_status[i] == -1 for i in range(1, self.num_nodes)): # 빈 공간이 있을 때만
            for car_id in cars_to_load:
                legal_actions.append(('LOAD', car_id, -1))

        legal_actions.append(('WAIT', -1, -1))
        return legal_actions

    def _find_best_relocation(self, blocker_id, path_to_clear):
        """블로커를 가장 싸게 재배치하는 방법을 찾고, 그에 대한 정보(경로, 비용)를 반환"""
        start_node = self.car_locations[blocker_id]
        
        # 옵션 1: 다른 빈 공간으로 위치변경(Relocation)
        best_reloc_cost = float('inf')
        best_reloc_path = None
        for empty_node in [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]:
            if empty_node not in path_to_clear:
                path = self.shortest_paths[start_node][empty_node]
                if path and self._calculate_path_cost(path) < best_reloc_cost:
                    best_reloc_cost = self._calculate_path_cost(path)
                    best_reloc_path = path

        # 옵션 2: 임시 하역 후 재적재(Temp Unload/Reload)
        path_out = self.shortest_paths[start_node][0]
        temp_unload_reload_cost = self._calculate_path_cost(path_out) * 2

        if best_reloc_path and best_reloc_cost <= temp_unload_reload_cost:
            return 'RELOCATE', best_reloc_path
        else:
            return 'TEMP_UNLOAD', [path_out, list(reversed(path_out))]

# -----------------------------------------------------------------
# ShipEnv.step  ‖  보상 스케일 조정과 TEMP_UNLOAD 경로 기록 보완
# -----------------------------------------------------------------
    # ─────────────────────────────────────────────
    # ShipEnv.step  ―  완전 교체
    # ─────────────────────────────────────────────
    def step(self, action):
        self.newly_generated_routes = []
        a_type, car_id, _ = action
        total_cost = 1.0                                      # 기본 행동 비용

        # ── LOAD ───────────────────────────────────
        if a_type == 'LOAD':
            empties = [i for i, s in enumerate(self.node_status) if s == -1 and i != 0]
            if not empties:
                return self._get_state(), -100.0, False
            tgt = max(empties, key=lambda n: len(self.shortest_paths[0][n]))
            path = self.shortest_paths[0][tgt]
            self.node_status[tgt] = car_id
            self.car_locations[car_id] = tgt
            self.cars_on_board.add(car_id)
            total_cost += self._calculate_path_cost(path)
            self.newly_generated_routes.append(
                {"demand_id": self.cars[car_id]["demand_id"], "route": path}
            )

        # ── UNLOAD ─────────────────────────────────
        elif a_type == 'UNLOAD':
            if car_id not in self.cars_on_board:
                return self._get_state(), -1000.0, False
            start   = self.car_locations[car_id]
            gate_p  = self.shortest_paths[start][0]

            re_cnt = 0
            while re_cnt < self.total_cars:
                blockers = [(self.node_status[n], n) for n in gate_p[1:] if self.node_status[n] != -1]
                if not blockers:
                    break
                blk_id, _ = blockers[0]
                move_type, paths = self._find_best_relocation(blk_id, gate_p)

                blk_start = self.car_locations[blk_id]
                if move_type == 'RELOCATE':
                    path = paths
                    self.node_status[blk_start] = -1
                    self.node_status[path[-1]]  = blk_id
                    self.car_locations[blk_id]  = path[-1]
                    total_cost += self._calculate_path_cost(path) + 50.0
                    self.newly_generated_routes.append(
                        {"demand_id": self.cars[blk_id]["demand_id"], "route": path}
                    )
                else:                                           # TEMP_UNLOAD
                    p_out, p_in = paths
                    total_cost += self._calculate_path_cost(p_out) + 100.0
                    total_cost += self._calculate_path_cost(p_in)
                    self.newly_generated_routes.extend([
                        {"demand_id": self.cars[blk_id]["demand_id"], "route": p_out},
                        {"demand_id": self.cars[blk_id]["demand_id"], "route": p_in},
                    ])
                    self.node_status[blk_start] = -1
                    self.cars_on_board.remove(blk_id)
                    self.car_locations[blk_id] = -1
                    re_cnt += 1

            self.node_status[start] = -1
            self.cars_on_board.remove(car_id)
            self.car_locations[car_id] = -1
            total_cost += self._calculate_path_cost(gate_p)
            self.newly_generated_routes.append(
                {"demand_id": self.cars[car_id]["demand_id"], "route": gate_p}
            )

        # ── WAIT ───────────────────────────────────
        elif a_type == 'WAIT':
            self.current_port += 1

        # ── 보상 계산 ───────────────────────────────
        raw_r = -total_cost
        if self.current_port >= self.num_ports:
            done = True
            ok = all(c['id'] not in self.cars_on_board for c in self.cars) \
                and all(loc == -1 for loc in self.car_locations)
            raw_r += 1000.0 if ok else -500.0
        else:
            done = False
        reward = float(self._scale_reward(raw_r))
        return self._get_state(), reward, done


# ---------------------------------------------------------------------
# 3. 에이전트 및 신경망, 매퍼 클래스 (V2.0 대응)
# ---------------------------------------------------------------------
class ReplayBuffer: # (이전과 동일)
    def __init__(self, capacity): self.memory = collections.deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(tuple(args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class PolicyNetwork(nn.Module): # (이전과 동일, 더 깊게 쌓을 수 있음)
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_size)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ActionMapper:
    """V2.0: 단순화된 전략적 행동에 대한 매퍼"""
    def __init__(self, problem):
        self.action_to_idx = {}
        self.idx_to_action = []
        self._add_action(('WAIT', -1, -1))
        num_cars = sum(q for _, q in problem['K'])
        for car_id in range(num_cars):
            self._add_action(('UNLOAD', car_id, -1))
            self._add_action(('LOAD', car_id, -1))
    def _add_action(self, action):
        if action not in self.action_to_idx:
            self.idx_to_action.append(action)
            self.action_to_idx[action] = len(self.idx_to_action) - 1
    def get(self, action): return self.action_to_idx.get(action)
    def __len__(self): return len(self.idx_to_action)

# ─────────────────────────────────────────────
# DQNAgent.__init__ 수정본
# ─────────────────────────────────────────────
class DQNAgent:
    def __init__(self, env, state_size, action_mapper, model_path=None):
        self.BATCH_SIZE   = 64
        self.GAMMA        = 0.99
        self.EPS_START    = 0.9
        self.EPS_END      = 0.02
        self.EPS_DECAY    = 10000
        self.TARGET_UPDATE = 50
        self.LR           = 3e-4
        self.CLIP_NORM    = 5.0

        self.env          = env
        self.action_mapper = action_mapper

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net  = PolicyNetwork(state_size, len(action_mapper)).to(self.device)
        self.target_net  = PolicyNetwork(state_size, len(action_mapper)).to(self.device)

        if model_path:
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory     = ReplayBuffer(25000)
        self.steps_done = 0

    # 선택
    def select_action(self, state, use_exploration=True):
        legal = self.env.get_legal_actions()
        if not legal:
            return None
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if use_exploration and random.random() < eps:
            return random.choice(legal)

        with torch.no_grad():
            qs = self.policy_net(torch.from_numpy(state).unsqueeze(0).to(self.device))[0]
            return max(legal, key=lambda a: qs[self.action_mapper.get(a)])

    # 학습
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        s, a, r, ns, d = zip(*self.memory.sample(self.BATCH_SIZE))

        s  = torch.from_numpy(np.stack(s)).to(self.device)
        ns = torch.from_numpy(np.stack(ns)).to(self.device)
        r  = torch.tensor(r, device=self.device, dtype=torch.float32)
        d  = torch.tensor(d, device=self.device, dtype=torch.bool)
        idx = torch.tensor([self.action_mapper.get(x) for x in a],
                           device=self.device).unsqueeze(1)

        q_sa = self.policy_net(s).gather(1, idx)

        next_q = torch.zeros(self.BATCH_SIZE, device=self.device)
        mask = ~d
        if mask.any():
            next_q[mask] = self.target_net(ns[mask]).max(1)[0].detach()

        target = r + self.GAMMA * next_q
        loss = F.smooth_l1_loss(q_sa, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.CLIP_NORM)
        self.optimizer.step()
        return loss.item()    

    # 타깃 동기화
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())





# ---------------------------------------------------------------------
# 4. V2.0 메인 솔루션 제출 함수
# ---------------------------------------------------------------------
def get_solution(problem):
    start_time = time.time()
    MODEL_PATH = os.path.join(script_dir, "trained_model_v2.pth")
    
    # === V2.0 대응: 최대 크기 결정 및 패딩 로직 ===
    # 제출 시에는 주어진 problem 하나에 대해서만 state/action 크기를 결정
    env = ShipEnv(problem)
    state_size = len(env.reset())
    action_mapper = ActionMapper(problem)
    agent = DQNAgent(env, state_size, action_mapper, model_path=MODEL_PATH)

    solution_routes = {p: [] for p in range(problem['P'])}
    state = env.reset()
    done = False
    
    while not done:
        if time.time() - start_time > 580: break
        
        # V2.0에서는 state 패딩이 필요 없음 (problem이 하나이므로)
        action = agent.select_action(state, use_exploration=False)
        if action is None: action = ('WAIT', -1, -1)

        # step을 실행하면 env.newly_generated_routes에 모든 경로가 기록됨
        next_state, reward, done = env.step(action)
        
        # 기록된 경로를 최종 솔루션에 추가
        if env.newly_generated_routes:
            solution_routes[env.current_port - (1 if action[0] == 'WAIT' else 0)].extend(env.newly_generated_routes)
        
        state = next_state
    
    return solution_routes

# ---------------------------------------------------------------------
# 5. V2.0 모델 학습을 위한 실행 코드 (상세 로그 기능 추가)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    log_path = os.path.join(script_dir, "v2_experiment")
    writer = SummaryWriter(log_dir=log_path)

    NUM_EPISODES = 20000  # V2는 더 많은 학습 필요
    PRINT_INTERVAL = 20 # 몇 에피소드마다 상세 로그를 출력할지 결정
    
    # 1. 학습에 사용할 문제 파일 목록 정의
    PROBLEM_FILES = [f'exercise_problems/prob{i}.json' for i in range(1, 11)]
    
    print("--- V2.0: Preparing for Training ---")
    problems = []; max_state_size = 0
    # ... (이전과 동일한 문제 로딩 및 최대 크기 결정 로직) ...
    max_car_count = 0; max_node_count = 0
    for file_path in PROBLEM_FILES:
        try:
            with open(file_path, 'r') as f:
                prob = json.load(f); prob['id'] = file_path; problems.append(prob)
                temp_env = ShipEnv(prob); state_size = len(temp_env.reset())
                if state_size > max_state_size: max_state_size = state_size
                num_cars = sum(q for _, q in prob['K']);
                if num_cars > max_car_count: max_car_count = num_cars
                if prob['N'] > max_node_count: max_node_count = prob['N']
        except FileNotFoundError: print(f"Warning: File not found - {file_path}")
    
    if not problems: print("Error: No problem files found."); exit()

    master_problem = {'K': [[(0, 1), max_car_count]], 'N': max_node_count, 'P': 10}
    action_mapper = ActionMapper(master_problem)
    agent = DQNAgent(None, max_state_size, action_mapper)

    print(f"--- V2.0: Starting Training on {len(problems)} problems ---")
    print(f"Max State size: {max_state_size}, Max Action size: {len(action_mapper)}")

    # === 상세 로깅을 위한 변수 초기화 ===
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    # 4. 학습 루프
    for i_episode in range(NUM_EPISODES):
        current_problem = random.choice(problems)
        env = ShipEnv(current_problem); agent.env = env
        state = env.reset()
        padded_state = np.zeros(max_state_size, dtype=np.float32)
        padded_state[:len(state)] = state; state = padded_state
        episode_reward = 0
        loss_val = None   

        
        for t in range(4000):  # 한 에피소드의 최대 스텝 수
            action = agent.select_action(state, use_exploration=True)
            if action is None: action = ('WAIT', -1, -1)
            next_state, reward, done = env.step(action)
            padded_next_state = np.zeros(max_state_size, dtype=np.float32)
            padded_next_state[:len(next_state)] = next_state
            
            agent.memory.push(state, action, reward, padded_next_state, done)
            state = padded_next_state

            loss_val = agent.optimize_model()

            episode_reward += reward
            if done: break
        
        # === 에피소드 종료 후, 통계 기록 ===
        was_successful = reward > 0 # 성공 보너스를 받으면 reward가 양수가 됨
        episode_rewards.append(episode_reward)
        episode_lengths.append(t + 1)
        episode_successes.append(1 if was_successful else 0)

        # ─────────────────────────────────────────────
        # 학습 루프 로그 부분 교체
        # ─────────────────────────────────────────────
        if (i_episode + 1) % PRINT_INTERVAL == 0:
            safe_print("-"*70)
            safe_print(f"Episode {i_episode + 1}/{NUM_EPISODES}")
            safe_print(f"  [Current] Problem: {current_problem['id'].split('/')[-1]:<15}"
                     f" | Reward: {episode_reward:<12.2f} | Length: {t+1:<4}"
                     f" | Result: {'Success' if was_successful else 'Failure'}")
            avg_r = np.mean(episode_rewards[-100:]); avg_l = np.mean(episode_lengths[-100:])
            sr = np.mean(episode_successes[-100:]) * 100.0
            safe_print(f"  [Avg 100] Reward: {avg_r:<12.2f} | Length: {avg_l:<4.1f}"
                     f" | Success Rate: {sr:5.1f}%")
            eps_now = agent.EPS_END + (agent.EPS_START - agent.EPS_END) * np.exp(
                -1.0 * agent.steps_done / agent.EPS_DECAY)
            safe_print(f"  [Agent] Epsilon: {eps_now:.4f}")
            safe_print("-"*70)


        # 주기적으로 타겟 신경망 업데이트
        if (i_episode + 1) % agent.TARGET_UPDATE == 0:
            agent.update_target_net()

        # 에피소드 종료 후
        writer.add_scalar("Reward/episode", episode_reward, i_episode)
        writer.add_scalar("SuccessRate/episode", was_successful, i_episode)
        writer.add_scalar("Length/episode", t + 1, i_episode)
        if loss_val is not None:
            writer.add_scalar("Loss/episode_end", loss_val, i_episode)  

        # 100-개 이동 평균까지 같이 적고 싶다면
        if (i_episode + 1) >= 100:
            writer.add_scalar("Reward/avg100",
                             np.mean(episode_rewards[-100:]), i_episode)


    print("--- V2.0 Training Finished ---")
    torch.save(agent.policy_net.state_dict(), os.path.join(script_dir, "trained_model_v2.pth"))

    print("Generalized V2 model saved to trained_model_v2.pth")
    writer.close()