import ipywidgets as widgets
from IPython.display import display
import zipfile
import io
import os
from datetime import datetime
import subprocess

import matplotlib.pyplot as plt
import ipywidgets as widgets
import time
import threading
from IPython.display import display
from IPython.display import clear_output
import json
import networkx as nx
from pathlib import Path
import numpy as np
from matplotlib import colors
import glob
import shutil



def algorithm_executor():

    # myalgorithm.py 파일과 algorithm 함수 존재 여부 확인 함수
    def check_algorithm_function(directory):
        check = False
        check_results = []

        
        alg_file_check = {'item': 'myalgorithm.py 파일 확인'}
        check_results.append(alg_file_check)

        target_file = os.path.join(BASE_TEMP_ALG_DIR, directory, 'myalgorithm.py')
        # myalgorithm.py 파일이 존재여부 확인....
        if not os.path.isfile(target_file):
            alg_file_check['result'] = False
            alg_file_check['desc'] = 'myalgorithm.py 파일이 압축파일의 root 폴더에 존재하지 않음'
            
            return check, check_results

        alg_file_check['result'] = True
        # myalgorithm.py is OK

        cwd = os.getcwd()

        alg_func_check = {'item': 'algorithm() 함수 확인'}
        check_results.append(alg_func_check)

        try:
            # Change directiory to the 
            os.chdir(os.path.join(BASE_TEMP_ALG_DIR, directory))

            # algorithm() 함수 확인....

            result = subprocess.run(['python', '-c', 'from myalgorithm import algorithm'], capture_output=True, text=True)


            if result.returncode == 0:
                alg_func_check['result'] = True
                check = True
            else:
                alg_func_check['result'] = False
                alg_func_check['desc'] = f'{result.stderr}'



        except Exception as e:
            alg_func_check['result'] = False
            alg_func_check['desc'] = repr(e) #'mmyalgorithm 모듈에서 algorithm 함수가 존재하지 않거나 import할 수 없음'
            # print("myalgorithm 모듈에서 algorithm 함수가 존재하지 않거나 import할 수 없음 [ERROR]")
            

        os.chdir(cwd)

        return check, check_results
    

    def check_probfile(prob_file):
        try:
            prob_path = os.path.join(BASE_TEMP_PROB_DIR, prob_file)
            with open(prob_path, 'r') as f:
                prob_info = json.load(f)

            # Check parameters in the prob_info dictionary
            N = prob_info['N']
            E = prob_info['E']
            K = prob_info['K']
            P = prob_info['P']
            F = prob_info['F']
            LB = prob_info['LB']            

            return True, [{'item': '문제 파일 확인', 'result': True}]
        except Exception as e:
            return False, [{'item': '문제 파일 확인', 'result': False, 'desc': repr(e)}]


    # Corrected function to handle zip upload
    def handle_zip_upload(change):
        with tester_output:
            # result_output.clear_output()
            if zip_uploader.value:


                # 현재 시각 기준으로 하위 폴더 생성 (예: 20240429_153005)
                now = datetime.now()
                timestamp = now.strftime('%Y%m%d_%H%M%S')
                target_dir = os.path.join(BASE_TEMP_ALG_DIR, timestamp)
                os.makedirs(target_dir, exist_ok=True)

                fileinfos = zip_uploader.value

                zip_uploader.value = []

                for fileinfo in fileinfos:
                    print(f"알고리즘 압축파일: {fileinfo['name']}")

                    # zip 압축 해제
                    with zipfile.ZipFile(io.BytesIO(fileinfo['content'])) as zip_ref:
                        zip_ref.extractall(target_dir)
                        print(f"{fileinfo['name']} 압축 해제 → {target_dir}... [OK]")
                        print("   포함된 파일 목록:")
                        for name in zip_ref.namelist():
                            print(f" - {name}")


                    # util.py copy
                    current_file_path = os.path.abspath(__file__)
                    current_dir = os.path.dirname(current_file_path)
                    util_path = os.path.join(current_dir, 'util.py')
                    try:
                        shutil.copy(util_path, target_dir)
                        print(f"'util.py'가 {target_dir}로 복사되었습니다.")
                    except FileNotFoundError:
                        print("util.py 파일을 찾을 수 없습니다. 경로를 확인하세요.")
                    except Exception as e:
                        print(f"복사 중 오류 발생: {e}")


                    refresh_alg_selector()

                

    def run_algorithm(_):
        current_alg = alg_selector.value


        new_output = widgets.Output(
            layout=widgets.Layout(
                height='300px',
                border='1px solid gray',
                overflow='auto'
            )
        )

        new_alg_output_box = widgets.VBox([
            widgets.Label(value=f"알고리즘: {current_alg} | 문제 파일: {prob_selector.value} | Timeout: {timeout_input.value}초"),
            new_output
        ])

        result_output_tab.children = list(result_output_tab.children) + [new_alg_output_box]
        result_output_tab.selected_index = len(result_output_tab.children) - 1
        result_output_tab.set_title(len(result_output_tab.children)-1, f'Run #{len(result_output_tab.children)-1}')

        result_output = new_output
        with result_output as out:
            # result_output.clear_output()
            
            alg_folder = alg_selector.value
            prob_file = prob_selector.value

            if not alg_folder:
                print("알고리즘을 먼저 선택하세요 [ERROR]")
                return
            
            if not prob_file:
                print("문제 파일을 먼저 선택하세요 [ERROR]")
                return

            
            if not check_algorithm_function(alg_folder)[0]:
                print(f'폴더 {alg_folder}에 저장된 알고리즘에 오류가 있습니다. [ERROR]')
                return

            if not check_probfile(prob_file):
                print(f'문제 파일 {prob_file}에 오류가 있습니다. [ERROR]')
                return


            try:


                alg_folder = os.path.join(BASE_TEMP_ALG_DIR, alg_folder)
                prob_path = os.path.join(BASE_TEMP_PROB_DIR, prob_file)
                prob_abspath = os.path.abspath(prob_path)

                print(f"myalgorithm.py 실행 중... 알고리즘: {alg_folder}, 문제 파일: {prob_path}, Timeout: {timeout_input.value}초)", flush=True)

                
                cmd = ['python', 'myalgorithm.py', 'test_prob', prob_abspath, str(timeout_input.value)]

                # Use Popen for real-time output capture
                process = subprocess.Popen(
                    cmd,
                    cwd=alg_folder,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,  # Line buffered
                )
                
                # Set a timeout handler
                def kill_on_timeout():
                    time.sleep(timeout_input.value)
                    if process.poll() is None:  # If the process is still running
                        process.kill()
                        # with result_output:
                        result_output.append_stderr("알고리즘 실행이 타임아웃되었습니다 [ERROR]\n")
                
                # Start timeout thread
                timeout_thread = threading.Thread(target=kill_on_timeout)
                timeout_thread.daemon = True
                timeout_thread.start()
                
                # Read and display output in real-time
                stdout_thread = threading.Thread(target=lambda: _read_output_stream(process.stdout, result_output))
                stderr_thread = threading.Thread(target=lambda: _read_output_stream(process.stderr, result_output, is_error=True))
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for process to complete
                return_code = process.wait()
                stdout_thread.join()
                stderr_thread.join()
                
                if return_code == 0:
                    result_filename = Path(alg_folder, 'results.json')
                    
                    if os.path.exists(result_filename):
                        with open(result_filename, 'r') as f:
                            result = json.load(f)
                            if result['feasible']:
                                print(f'Elaspsed time is {result["time"]} sec')
                                print(f'The solution is feasible!!! Obj. value is {result["obj"]}')
                                print('The solution will be visualized below:')

                                new_alg_output_box.children = list(new_alg_output_box.children) + [solution_visualizer(prob_abspath, result_filename)]
                                # visualizer_box.children = [solution_visualizer(prob_abspath, result_filename)]
                            else:
                                print(f'The solution is infeasible!!!')
                                # visualizer_box.children = []
                    else:
                        print("결과 파일(results.json)을 찾을 수 없습니다 [ERROR]")
                else:
                    print(f"알고리즘이 비정상 종료되었습니다. 반환 코드: {return_code} [ERROR]")
                    # visualizer_box.children = []

            except Exception as e:
                print(f"알고리즘 실행 중 오류 발생: {e} [ERROR]")
                # visualizer_box.children = []

    def _read_output_stream(stream, output_widget, is_error=False):
        """Helper function to read from a stream and update the output widget"""
        current_alg = alg_selector.value
        for line in iter(stream.readline, ''):
            if line:
                if is_error:
                    output_widget.append_stderr(f"{line.rstrip()}\n")
                else:
                    output_widget.append_stdout(f'{line.rstrip()}\n')
                # output_widget.layout.scroll_y = '100%'
        stream.close()

    def handle_problem_file_upload(change):
        with tester_output:
            if problem_file_uploader.value:
                
                target_dir = BASE_TEMP_PROB_DIR

                # 업로드된 문제 파일 저장
                fileinfo = next(iter(problem_file_uploader.value))
                problem_file_uploader.value = []
                problem_file_path = os.path.join(target_dir, fileinfo['name'])
                with open(problem_file_path, 'wb') as f:
                    f.write(fileinfo['content'])
                print(f"문제 파일 저장 완료: {problem_file_path}")


                refresh_prob_selector(fileinfo['name'])


                
    # 임시 알고리즘 디렉토리 설정
    BASE_TEMP_ALG_DIR = './tmp_alg'
    os.makedirs(BASE_TEMP_ALG_DIR, exist_ok=True)

    # 임시 문제 디렉토리 설정
    BASE_TEMP_PROB_DIR = './tmp_prob'
    os.makedirs(BASE_TEMP_PROB_DIR, exist_ok=True)

    # zip 파일 업로드 위젯 생성
    zip_uploader = widgets.FileUpload(
        accept='.zip',
        multiple=False,
        description='알고리즘 압축파일 선택',
        layout=widgets.Layout(width='200px')

    )

    # 알고리즘 및 문제 파일 확인 결과 출력 위젯
    alg_check_box = widgets.HBox([])
    prob_check_box = widgets.HBox([])


    # 알고리즘 선택
    alg_selector = widgets.Dropdown(
        options=[],
        layout=widgets.Layout(width='200px')
    )

    # 알고리즘 선택 위젯 생성
    def refresh_alg_selector():
        alg_dirs = os.listdir(BASE_TEMP_ALG_DIR)

        if len(alg_dirs) > 0:
            sorted_alg_dir = sorted(alg_dirs, reverse=True)
                                    
            alg_selector.options = sorted_alg_dir
            alg_selector.value = sorted_alg_dir[0]
            
    # 알고리즘 선택 위젯의 이벤트 핸들러
    def handle_alg_selector(change):
        if change['type'] == 'change' and change['name'] == 'value':
            alg_folder = change['new']

            check, check_results = check_algorithm_function(alg_folder)

            alg_check_box.children = [
                widgets.Label(value=
                             " | ".join([f"{'✅' if result['result'] else '❌'} {result['item']}"
                             for result in check_results])
                             )
            ]

            if check:
                run_algorithm_button.disabled = False
                problem_file_uploader.disabled = False
            else:
                run_algorithm_button.disabled = True
                problem_file_uploader.disabled = True

                with tester_output:
                    print(f"알고리즘 {alg_folder}에 오류가 있습니다.")
                    for result in check_results:
                        if not result['result']:
                            print(f"{result['item']} 실패: {result.get('desc', '')}")
                    
            

    # 문제 파일 선택 위젯 생성
    prob_selector = widgets.Dropdown(
        options=[],
        layout=widgets.Layout(width='200px')
    )

    # 문제 파일 선택 위젯의 이벤트 핸들러
    def refresh_prob_selector(default=None):
        prob_files = glob.glob('*.json', root_dir=BASE_TEMP_PROB_DIR)

        if len(prob_files) > 0:

            prob_selector.options = prob_files
            if default is None:
                prob_selector.value = prob_files[0]
            else:
                prob_selector.value = default

    # 문제 파일 선택 위젯의 이벤트 핸들러
    def handle_prob_selector(change):
        if change['type'] == 'change' and change['name'] == 'value':
            prob_file = change['new']

            check, check_results = check_probfile(prob_file)
            prob_check_box.children = [
                widgets.Label(value="|".join(
                    [f"{'✅' if result['result'] else '❌'} {result['item']}" 
                    for result in check_results])
                )

            ]

            if check:
                run_algorithm_button.disabled = False
                problem_file_uploader.disabled = False
            else:
                run_algorithm_button.disabled = True
                problem_file_uploader.disabled = True
                with tester_output:
                    print(f"문제 {prob_file}에 오류가 있습니다.")
                    for result in check_results:
                        if not result['result']:
                            print(f"{result['item']} 실패: {result.get('desc', '')}")
                    

    # 알고리즘 압축파일 업로드 위젯의 이벤트 핸들러 등록
    zip_uploader.observe(handle_zip_upload, names='value')



    # 알고리즘 timeout 입력 위젯
    timeout_input = widgets.IntText(
        value=10,
        description='Timeout (초):',
        layout=widgets.Layout(width='130px')
    )



    # 문제 파일 선택 위젯 생성
    problem_file_uploader = widgets.FileUpload(
        accept='.json',
        multiple=False,
        description='문제 파일 선택',
        layout=widgets.Layout(width='200px')
    )



    # 알고리즘 실행 버튼 생성
    run_algorithm_button = widgets.Button(
        description='선택된 알고리즘으로 선택한 문제 풀기!',
        layout=widgets.Layout(width='300px'),
        disabled = True
    )
    run_algorithm_button.on_click(run_algorithm)

    tester_output = widgets.Output(
        layout=widgets.Layout(
            height='300px',
            border='1px solid gray',
            overflow='auto'
        )
    )

    result_output_tab = widgets.Tab([tester_output])
    result_output_tab.set_title(0, 'Tester Output')

    
    problem_file_uploader.observe(handle_problem_file_upload, names='value')

    alg_selector.observe(handle_alg_selector)
    refresh_alg_selector()

    prob_selector.observe(handle_prob_selector)
    refresh_prob_selector()


    alg_zip_selectors_box = widgets.HBox([widgets.Label(value="1. 저장된 알고리즘을 선택하세요 👉 "), alg_selector, widgets.Label(value="또는 새로운 알고리즘 압축파일을 선택하세요 👉 "), zip_uploader])
    probfile_selectors_box = widgets.HBox([widgets.Label(value="2. 문제 파일을 선택하세요 👉 "), prob_selector, widgets.Label(value="또는 새로운 문제 파일을 선택하세요 👉 "),problem_file_uploader, timeout_input])
    # param_box = widgets.HBox([widgets.Label(value="Timeout 값 지정 👉 "), timeout_input])
    run_alg_box = widgets.HBox([widgets.Label(value="3. 알고리즘 실행! 👉 "), run_algorithm_button])
    visualizer_box = widgets.HBox([])


    alg_test_box = widgets.VBox([
        alg_zip_selectors_box,
        probfile_selectors_box,
        widgets.HBox([widgets.Label(value=""), alg_check_box, widgets.Label(value=" | "), prob_check_box]),
        run_alg_box,
        widgets.Label(value="알고리즘 output 👇"),
        result_output_tab,  # 탭 위젯 사용
        visualizer_box
    ])

    plt.close('all')

    return alg_test_box




def solution_visualizer(prob_filename, result_filename):

    with open(prob_filename, 'r') as f:
        prob_info = json.load(f)


    N = prob_info['N']
    E = prob_info['E']
    E = set([(u,v) for (u,v) in E])
    K = prob_info['K']
    P = prob_info['P']
    F = prob_info['F']

    grid_graph = prob_info['grid_graph']



    with open(result_filename, 'r') as f:
        results = json.load(f)

    solution = {
        int(p): routes
        for p, routes in results['solution'].items()
    }

    node_allocations = np.ones(N, dtype=int) * -1

    alloc_status = {}

    for p in range(P):
        alloc_status_list = []
        for r,k in solution[p]:
            if r[0] != 0:
                node_allocations[r[0]] = -1
            if r[-1] != 0:
                node_allocations[r[-1]] = k

            alloc_status_list.append(node_allocations.copy())

        alloc_status[p] = alloc_status_list


    G = nx.Graph()


    for n, attr in grid_graph['nodes']:
        G.add_node(tuple(n), **attr)

    for u, v, attr in grid_graph['edges']:
        G.add_edge(tuple(u), tuple(v), **attr)

    node_id_dict = {
        G.nodes[n]['id']: n for n in G
    }



    ramp_nodes = [n for (n,attr) in G.nodes.items() if attr['type'] == 'ramp']
    hold_nodes = [n for (n,attr) in G.nodes.items() if attr['type'] == 'hold']
    gate_nodes = [n for (n,attr) in G.nodes.items() if attr['type'] == 'gate']


    with plt.ioff():
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title('')


    fig.canvas.draw_idle()
    fig.tight_layout()

    node_size = 100

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx_nodes(G, ax=ax, nodelist=ramp_nodes, pos=pos, node_shape='s', node_color='none', node_size=node_size)
    nx.draw_networkx_nodes(G, ax=ax, nodelist=hold_nodes, pos=pos, node_shape='o', node_color='none', node_size=node_size)
    nx.draw_networkx_nodes(G, ax=ax, nodelist=gate_nodes, pos=pos, node_shape='*', node_color='orange', node_size=node_size*3)


    ramp_edges = [(u, v) for (u, v, attr) in G.edges(data=True) if 'ramp' in attr]
    common_edges = [(u, v) for (u, v, attr) in G.edges(data=True) if 'ramp' not in attr]

    nx.draw_networkx_edges(G, ax=ax, edgelist=common_edges, pos=pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_edges(G, ax=ax, edgelist=ramp_edges, pos=pos, alpha=0.5, width=2, edge_color='gray', connectionstyle='arc3,rad=0.3', arrows=True, min_source_margin=0, min_target_margin=0, node_size=node_size)

    bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.5)
    nx.draw_networkx_labels(G, ax=ax, pos=pos, labels=nx.get_node_attributes(G, 'id'), font_size=6, font_color='white', bbox=bbox)




    route_patches = []


    def update_graph(p, r_idx):

        r, k = solution[p][r_idx]

        for patch in route_patches:
            patch.remove()
        route_patches.clear()

        cmap = plt.cm.Reds

        demand_nodes = [node_id_dict[idx] for idx,k in enumerate(alloc_status[p][r_idx]) if k >= 0]
        demand_node_colors = [K[k][0][1] for idx,k in enumerate(alloc_status[p][r_idx]) if k >= 0]
        
        demand_node_patch = nx.draw_networkx_nodes(G, ax=ax, nodelist=demand_nodes, pos=pos, node_shape='s', node_color=demand_node_colors, cmap=cmap, vmin=0, vmax=P-1, node_size=node_size*1.5)

        route_patches.append(demand_node_patch)

        route_ramp_edges = [(node_id_dict[i], node_id_dict[j]) for (i,j) in zip(r[:-1], r[1:]) if 'ramp' in G[node_id_dict[i]][node_id_dict[j]]]
        route_common_edges = [(node_id_dict[i], node_id_dict[j]) for (i,j) in zip(r[:-1], r[1:]) if 'ramp' not in G[node_id_dict[i]][node_id_dict[j]]]

        route_common_edges_patch = nx.draw_networkx_edges(G, ax=ax, edgelist=route_common_edges, pos=pos, alpha=1, width=2, edge_color='green', arrowstyle='-|>', arrows=True)
        route_ramp_edges_patch = nx.draw_networkx_edges(G, ax=ax, edgelist=route_ramp_edges, pos=pos, alpha=1, width=2, edge_color='green', arrowstyle='-|>', connectionstyle='arc3,rad=0.3', arrows=True, min_source_margin=2, min_target_margin=2, node_size=node_size)

        route_patches.extend(route_common_edges_patch)
        route_patches.extend(route_ramp_edges_patch)


        if r[-1] == 0:
            demand_gate = nx.draw_networkx_nodes(G, ax=ax, nodelist=[node_id_dict[r[0]]], pos=pos, node_shape='D', node_color=[K[k][0][1]], edgecolors='yellow', cmap=cmap, vmin=0, vmax=P, node_size=node_size*3)
            route_patches.append(demand_gate)

        if r[0] == 0:
            demand_loading = nx.draw_networkx_nodes(G, ax=ax, nodelist=[node_id_dict[r[-1]]], pos=pos, node_shape='D', node_color=[K[k][0][1]], edgecolors='yellow', cmap=cmap, vmin=0, vmax=P, node_size=node_size*3)
            route_patches.append(demand_loading)

        fig.canvas.draw_idle()

        update_buttons()





    def update_buttons():
        current_route_idx = route_selector.options.index(route_selector.value)
        prev_route_button.disabled = current_route_idx == 0
        next_route_button.disabled = current_route_idx == len(route_selector.options) - 1

        current_port_idx = port_selector.value
        prev_port_button.disabled = current_port_idx == 0
        next_port_button.disabled = current_port_idx == P - 1


    def on_port_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            port = change['new']
            route_selector.options = list(range(len(solution[port])))
            route_selector.value = route_selector.options[0]
            update_graph(port, 0)

    def on_route_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            port = port_selector.value
            route_idx = route_selector.options.index(change['new'])
            update_graph(port, route_idx)

    def on_prev_port_clicked(b):
        idx = port_selector.value
        if idx > 0:
            port_selector.value = idx - 1

    def on_next_port_clicked(b):
        idx = port_selector.value
        if idx < P - 1:
            port_selector.value = idx + 1

    def on_prev_route_clicked(b):
        idx = route_selector.value
        if idx > 0:
            route_selector.value = idx - 1

    def on_next_route_clicked(b):
        idx = route_selector.value
        if idx < len(route_selector.options) - 1:
            route_selector.value = idx + 1


    def auto_advance_routes():
        global auto_running
        while auto_running:
            idx = route_selector.options.index(route_selector.value)
            if idx < len(route_selector.options) - 1:
                route_selector.value = route_selector.options[idx + 1]
            else:
                auto_running = False
                auto_toggle_button.description = '자동 경로 전환'
                break
            time.sleep(1)

    def toggle_auto_route(b):
        global auto_running, auto_thread
        auto_running = not auto_running
        auto_toggle_button.description = '정지' if auto_running else '자동 경로 전환'
        if auto_running:
            auto_thread = threading.Thread(target=auto_advance_routes)
            auto_thread.start()


    def on_hover(event):
        if not event.inaxes:
            return
                
        threshold = 0.5
        x, y = event.xdata, event.ydata
        
        near_nodes = [(n,xy) for n, xy in pos.items() if abs(x-xy[0]) < threshold and abs(y-xy[1]) < threshold]

        if len(near_nodes) > 0:
            n = min([(((x-xy[0])**2 + ((y-xy[1])**2))**0.5,n) for n,xy in near_nodes])[1]
            p = port_selector.value
            r_idx = route_selector.value
            i = int(G.nodes[n]["id"])
            k = alloc_status[p][r_idx][i]
            if k == -1:
                status = 'empty'
            else:
                status = f'occupied by demand k={k}, (o={K[k][0][0]}, d={K[k][0][1]})'
            msg = f'Node: {i} | Status: {status}'

            fig.canvas.toolbar.set_message(msg)

            # fig.canvas.draw_idle()


    global auto_running
    auto_running = False

    cid = fig.canvas.mpl_connect("motion_notify_event", on_hover)


    common_style = {'description_width': '50px'}

    port_selector = widgets.Dropdown(options=list(range(P)), value=0, description='항구:', style=common_style,layout=widgets.Layout(width='150px'))
    route_selector = widgets.Dropdown(options=list(range(len(solution[0]))), value=0, description='경로:', style=common_style, layout=widgets.Layout(width='150px'))

    prev_port_button = widgets.Button(description='이전 항구')
    next_port_button = widgets.Button(description='다음 항구')
    prev_route_button = widgets.Button(description='이전 경로')
    next_route_button = widgets.Button(description='다음 경로')

    auto_toggle_button = widgets.Button(description='자동 경로 전환')


    # auto_route_button = widgets.Button(description='자동 경로 전환')
    # stop_auto_button = widgets.Button(description='정지')

    port_box = widgets.HBox([prev_port_button, port_selector, next_port_button], layout=widgets.Layout(gap='20px'))
    route_box = widgets.HBox([prev_route_button, route_selector, next_route_button, auto_toggle_button], layout=widgets.Layout(gap='10px'))
    graph_box = widgets.VBox([port_box, route_box, fig.canvas])


    # 이벤트 연결
    port_selector.observe(on_port_change)
    route_selector.observe(on_route_change)
    prev_port_button.on_click(on_prev_port_clicked)
    next_port_button.on_click(on_next_port_clicked)
    prev_route_button.on_click(on_prev_route_clicked)
    next_route_button.on_click(on_next_route_clicked)
    auto_toggle_button.on_click(toggle_auto_route)

    # auto_route_button.on_click(on_auto_route_clicked)
    # stop_auto_button.on_click(on_stop_auto_clicked)

    update_graph(0, 0)
    
    return graph_box
