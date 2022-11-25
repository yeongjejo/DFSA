import random
import numpy as np


def simulation(node_num):
    slot_num = 10 # 초기 슬롯수 설정
    priors = [1, 1] # alpha, beta
    threshold = 0.5 # 임계값 설정

    throughputs = []

    slot_success_num = [[0 for _ in range(slot_num)] for i in range(node_num)]    # 슬롯 마다 전송 성공 횟수
    slot_collision_num = [[0 for _ in range(slot_num)] for i in range(node_num)]  # 슬롯 마다 전송 실패 횟수

    # episode 시작
    for epi in range(100000):
        slot_of_frame = [0 for _ in range(slot_num)] # episode frame 초기화

        actions = []  # 노드(에이전트)들의 action값을 저장할 배열
        select_sampling = []  # episode sampling값을 저장할 배열

        th = 0 # throughput 계산용

        for node in range(node_num):
            samplings = []
            for slot in range(slot_num):
                samplings.append(random.betavariate(priors[0] + slot_success_num[node][slot], priors[1] + slot_collision_num[node][slot]))

            sampling = np.argmax(samplings)
            slot_of_frame[sampling] += 1  # 패킷 전송
            actions.append(sampling)  # action 저장
            select_sampling.append(np.max(samplings))  # sampling값 저장

        # reward 설정
        for node, action in enumerate(actions):
            if slot_of_frame[action] == 1:  # 패킷 전송 성공시
                slot_success_num[node][action] += 1
                th += 1
            else:  # 패킷 전송 실패시
                slot_collision_num[node][action] += 1

        throughputs.append(th / slot_num)  # throughput 계산

        # frame size(슬롯수) 설정
        if np.min(select_sampling) < threshold: # 임계값 보다 낮으면 슬롯 추가
            slot_num += 1
            for i in range(node_num):
                slot_success_num[i].append(0)
                slot_collision_num[i].append(0)
        if np.min(slot_of_frame) == 0: # 빈슬롯 발생시 슬롯 감소
            slot_num -= 1
            for i in range(node_num):
                slot_success_num[i].pop()
                slot_collision_num[i].pop()

    return np.mean(throughputs[-100:])


if __name__ == '__main__':
    node_num = [i+1 for i in range(100)] # 패킷을 전송할 노드수
    throughputs = []

    for node in node_num:
        throughput = simulation(node)
        throughputs.append(throughput)

    print(throughputs)