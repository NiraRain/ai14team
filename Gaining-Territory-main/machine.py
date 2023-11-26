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

    def calculate_triangle_score(self, move):
        triangle_score = 0
        # 모든 선과 비교하여 삼각형 형성 가능 여부 확인
        for line in self.drawn_lines:
            for other_line in self.drawn_lines:
                if line != other_line and move != line and move != other_line:
                    if self.forms_triangle(move, line, other_line):
                        if self.is_valid_triangle(move, line, other_line):
                            triangle_score += 1
        return triangle_score

    def forms_triangle(self, line1, line2, line3):
        # 세 선이 삼각형을 형성하는지 확인
        points = set(line1 + line2 + line3)
        return len(points) == 3

    def is_valid_triangle(self, line1, line2, line3):
        # 삼각형 내부에 다른 점이 없는지 확인
        triangle = Polygon([line1[0], line1[1], line2[0], line2[1], line3[0], line3[1]])
        for point in self.whole_points:
            if point not in line1 and point not in line2 and point not in line3:
                if triangle.contains(Point(point)):
                    return False
        return True
    
    def calculate_linearity_score(self, move):
        # 선이 기존의 선과 같은 일직선상에 있으면 1점 부여
        for line in self.drawn_lines:
            if self.is_colinear(move, line):
                return 1 
        return 0 

    def calculate_heuristic_move(self, tmp_score):
        user_score, machine_score = tmp_score
        score_difference = machine_score - user_score
        #total_score = triangle_score + score_difference
        total_score = score_difference
        return total_score
    
    def parallel_minmax(self, drawn_lines, depth, alpha, beta, maximizing_player, tmpscore, move):
        _, best_move_score = self.minmax(drawn_lines + [move], depth - 1, alpha, beta, not maximizing_player, tmpscore)
        return move, best_move_score
    
    def find_best_selection(self):
        start = time.time()
        tmp_score = self.score.copy()

        if len(self.whole_points) == 15:
            self.depth_n = 3
        else:
            self.depth_n = 2

        unconnected_points = [p for p in self.whole_points if p not in set(sum(self.drawn_lines, []))]

        possible_lines = [[p1, p2] for p1 in unconnected_points for p2 in unconnected_points if p1 != p2]
        valid_lines = [line for line in possible_lines if self.check_availability(line)]

        if valid_lines:
            for line1 in self.drawn_lines:
                for line2 in self.drawn_lines:
                    if line1 != line2:
                        common_point = set(line1).intersection(set(line2))
                        if common_point:
                            common_point = common_point.pop()
                            for point1 in line1:
                                if point1 != common_point:
                                    for point2 in line2:
                                        if point2 != common_point:
                                            new_line = [point1, point2]
                                            if self.check_availability(new_line):
                                                return new_line
            return random.choice(valid_lines) 
        else:
            available_moves = self.get_available_moves(self.drawn_lines)
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.parallel_minmax, self.drawn_lines[:], self.depth_n, float('-inf'), float('inf'), True, tmp_score, move) for move in available_moves]

                results = [future.result() for future in as_completed(futures)]

            best_move = max(results, key=lambda x: x[1])[0]
            end = time.time()
            print(f"{end - start:.5f} sec")
            return best_move
 
    ####
    # find_best_selection 점2개 이상 남기 전에, 삼각형을 만들 수 있는 상황이 있다면 
    # 그 예외상황에서 삼각형을 우선으로 처리하도록 예외처리 
    """def find_best_selection(self):
        start = time.time()
        tmp_score = self.score.copy()

        unconnected_points = [p for p in self.whole_points if p not in set(sum(self.drawn_lines, []))]

        if len(unconnected_points) >= 2:
            possible_lines = [[p1, p2] for p1 in unconnected_points for p2 in unconnected_points if p1 != p2]
            valid_lines = [line for line in possible_lines if self.check_availability(line)]
            
            for line1 in self.drawn_lines:
                for line2 in self.drawn_lines:
                    if line1 != line2:
                        common_point = set(line1).intersection(set(line2))
                        if common_point:
                            common_point = common_point.pop()
                            for point1 in line1:
                                if point1 != common_point:
                                    for point2 in line2:
                                        if point2 != common_point:
                                            new_line = [point1, point2]
                                            if self.check_availability(new_line):
                                                return new_line

            if valid_lines:
                return random.choice(valid_lines) 
            else:
                _, best_line = self.minmax(self.drawn_lines[:], depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, tmpscore=tmp_score)
                end = time.time()
                print(f"{end - start:.5f} sec")
                return best_line
        else:
            _, best_line = self.minmax(self.drawn_lines[:], depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, tmpscore=tmp_score)
            end = time.time()
            print(f"{end - start:.5f} sec")
            return best_line"""
    
    # 말단노드까지 score 점수 갱신과 말단노드 도착 후 점수 초기화 작업?

    def minmax(self, drawn_lines, depth, alpha, beta, maximizing_player, tmpscore):
        print(drawn_lines)
        if depth == 0 or not self.get_available_moves(drawn_lines):
            score = self.calculate_heuristic_move(tmpscore)
            #user_score, machine_score = self.check_triangle_score(drawn_lines, tmpscore, maximizing_player)
            #score = machine_score - user_score
            print(f"depth 0 score = {score}")
            return score, None

        if maximizing_player:
            max_eval = float('-inf')
            best_line = None

            for move in self.get_available_moves(drawn_lines):
                original_state = drawn_lines.copy()

                drawn_lines.append(move)
                new_tmp_score = self.check_triangle_score(drawn_lines, tmpscore, maximizing_player)

                eval, _ = self.minmax(drawn_lines, depth - 1, alpha, beta, False, new_tmp_score)
                print(f" move = {move} score = {new_tmp_score}, {eval}, drawn_lines = {drawn_lines}")
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

            for move in self.get_available_moves(drawn_lines):
                original_state = drawn_lines.copy()

                drawn_lines.append(move)
                new_tmp_score = self.check_triangle_score(drawn_lines, tmpscore, maximizing_player)

                eval, _ = self.minmax(drawn_lines, depth - 1, alpha, beta, True, new_tmp_score)
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

    def get_available_moves(self, drawn_lines):
        available_moves = []
        for point1, point2 in combinations(self.whole_points, 2):
            move = [point1, point2]
            if move not in drawn_lines and self.check_availability(move):
                available_moves.append(move)
        return available_moves
    
    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point==line[0] or point==line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    
        
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
                            print(maximizing_player)
                            print(current_score)
                            return current_score
        print(current_score)
        return current_score