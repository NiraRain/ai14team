import random
from itertools import combinations, product
from shapely.geometry import LineString, Point, Polygon
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import pandas as pd
import time


class MACHINE():
    """
        [ MACHINE ]
        MinMax Algorithm을 통해 수를 선택하는 객체.
        - 모든 Machine Turn마다 변수들이 업데이트 됨

        ** To Do **
        MinMax Algorithm을 이용하여 최적의 수를 찾는 알고리즘 생성
           - class 내에 함수를 추가할 수 있음
           - 최종 결과는 find_best_selection을 통해 Line 형태로 도출
               * Line: [(x1, y1), (x2, y2)] -> MACHINE class에서는 x값이 작은 점이 항상 왼쪽에 위치할 필요는 없음 (System이 organize 함)
    """
    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]
        self.depth_n = 0
        self.minmax_count = 0
        self.play_time = 0
 
    def find_best_selection(self):
        available_moves = self.get_available_moves(self.drawn_lines)
        self.play_time = 0
        if(len(available_moves)) < 9:
            self.depth_n = 6
        elif(len(available_moves)) < 13:
            self.depth_n = 5
        elif(len(available_moves)) < 20:
            self.depth_n = 4
        elif(len(available_moves)) < 28:
            self.depth_n = 3
        else:
            self.depth_n = 2

        #start = time.time()
        tmp_score = self.score.copy()

        unconnected_points = [p for p in self.whole_points if p not in set(sum(self.drawn_lines, []))]
        
        if len(unconnected_points) >= 2:
            possible_lines = [[p1, p2] for p1 in unconnected_points for p2 in unconnected_points if p1 != p2]
            condition_list1 = [False, True, True, True]
            valid_lines = [line for line in possible_lines if self.check_availability(line, condition_list1)]

            if valid_lines:
                return random.choice(valid_lines) 
            else:
                _, best_line = self.minmax(self.drawn_lines[:], can_move=available_moves, depth=self.depth_n, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, tmpscore=tmp_score)
                #end = time.time()
                #print(f"{end - start:.5f} sec")
                self.minmax_count += 1
                #print(f"play_time = {self.play_time}, available = {len(available_moves)}")
                #print(self.minmax_count)
                return best_line
        else:
            _, best_line = self.minmax(self.drawn_lines[:], can_move=available_moves, depth=self.depth_n, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, tmpscore=tmp_score)
            end = time.time()
            #print(f"{end - start:.5f} sec")
            self.minmax_count += 1
            #print(f"play_time = {self.play_time}, available = {len(available_moves)}")
            #print(self.minmax_count)
            return best_line
    
    def check_availability(self, line, conditionlist):
        line_string = LineString(line)

        condition1 = True
        if conditionlist[0]:
            condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        condition2 = True
        if conditionlist[1]:
            for point in self.whole_points:
                if point==line[0] or point==line[1]:
                    continue
                else:
                    if bool(line_string.intersection(Point(point))):
                        condition2 = False

        condition3 = True
        if conditionlist[2]:
            for l in self.drawn_lines:
                if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                    continue
                elif bool(line_string.intersection(LineString(l))):
                    condition3 = False

        condition4 = True
        if conditionlist[3]:
            condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    

    # 삼각형 구성 확인 함수
    def forms_triangle(self, line1, line2, line3):
        points = set(line1 + line2 + line3)
        return len(points) == 3

    def is_valid_triangle(self, line1, line2, line3):
        triangle = Polygon([line1[0], line1[1], line2[0], line2[1], line3[0], line3[1]])
        for point in self.whole_points:
            if point not in line1 and point not in line2 and point not in line3:
                if triangle.contains(Point(point)):
                    return False
        return True
        
    # 삼각형에 따른 tmp_score 배분 함수
    def check_triangle_score(self, new_drawn_lines, tmp_score, maximizing_player):
        current_score = tmp_score.copy()
        if new_drawn_lines:
            new_line = new_drawn_lines[-1]
            point1, point2 = new_line

            connected_lines_point1 = [line for line in new_drawn_lines if point1 in line]
            connected_lines_point2 = [line for line in new_drawn_lines if point2 in line]

            for line1 in connected_lines_point1:
                for line2 in connected_lines_point2:
                    if line1 != line2 and line1 != new_line and line2 != new_line:
                        if self.forms_triangle(new_line, line1, line2) and self.is_valid_triangle(new_line, line1, line2):
                            if maximizing_player:
                                current_score[1] += 1
                            else:
                                current_score[0] += 1
                            #print(maximizing_player)
                            #print(current_score)
                            return current_score
        #print(current_score)
        return current_score
    
    # 마지막 점수차 휴리스틱 값 설정
    def calculate_heuristic_move(self, tmp_score):
        user_score, machine_score = tmp_score
        score_difference = machine_score - user_score
        total_score = score_difference
        return total_score
    
    # 그릴 수 있는 선 찾는 함수
    def get_available_moves(self, drawn_lines):
        available_moves = []
        for point1, point2 in combinations(self.whole_points, 2):
            move = [point1, point2]
            conditionlist = [False, True, True, False]
            if move not in drawn_lines and self.check_availability(move, conditionlist):
                available_moves.append(move)
        return available_moves
    
    # 진행된 상황에 따른 available_moves 업데이트 함수
    def remove_available_moves(self, available_moves):
        new_available_moves = []
        for move in available_moves:
            conditionlist = [False, False, True, False]
            if self.check_availability(move, conditionlist):
                new_available_moves.append(move)
        return new_available_moves
    
    # minmax 함수
    def minmax(self, drawn_lines, can_move, depth, alpha, beta, maximizing_player, tmpscore):
        #print(drawn_lines)
        self.play_time += 1
        if depth == 0 or not can_move:
            score = self.calculate_heuristic_move(tmpscore)
            return score, None

        if maximizing_player:
            max_eval = float('-inf')
            best_line = None

            for move in can_move:
                original_state = drawn_lines.copy()

                drawn_lines.append(move)
                new_tmp_score = self.check_triangle_score(drawn_lines, tmpscore, maximizing_player)
                new_available_moves = self.remove_available_moves(can_move)

                eval, _ = self.minmax(drawn_lines, new_available_moves, depth - 1, alpha, beta, False, new_tmp_score)
                #print(f" move = {move} score = {new_tmp_score}, {eval}, drawn_lines = {drawn_lines}")
                total_eval = eval
                if total_eval > max_eval:
                    max_eval = total_eval
                    best_line = move

                drawn_lines = original_state

                alpha = max(alpha, total_eval)
                if beta <= alpha:
                    break

            return max_eval, best_line

        else:
            min_eval = float('inf')
            best_line = None

            for move in can_move:
                original_state = drawn_lines.copy()

                drawn_lines.append(move)
                new_tmp_score = self.check_triangle_score(drawn_lines, tmpscore, maximizing_player)
                new_available_moves = self.remove_available_moves(can_move)

                eval, _ = self.minmax(drawn_lines, new_available_moves, depth - 1, alpha, beta, True, new_tmp_score)
                total_eval = eval
                #print(maximizing_player, eval)
                if total_eval < min_eval:
                    min_eval = total_eval
                    best_line = move

                drawn_lines = original_state

                beta = min(beta, total_eval)
                if beta <= alpha:
                    break

            return min_eval, best_line