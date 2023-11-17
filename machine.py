import random
from itertools import combinations
from shapely.geometry import LineString, Point, Polygon
import math

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
        self.heuristic_scores = {}

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
    
    def calculate_proximity_score(self, move):
        # 중앙에 가까울수록 높은 점수 부여
        mid_point = (self.board_size // 2, self.board_size // 2)
        distance = math.sqrt((move[0][0] - mid_point[0])**2 + (move[0][1] - mid_point[1])**2) + \
                   math.sqrt((move[1][0] - mid_point[0])**2 + (move[1][1] - mid_point[1])**2)
        proximity_score = (self.board_size - distance) / self.board_size
        return proximity_score
    
    def calculate_linearity_score(self, move):
        # 선이 기존의 선과 같은 일직선상에 있으면 1점 부여
        for line in self.drawn_lines:
            if self.is_colinear(move, line):
                return 1 
        return 0 

    def check_line_intersection(self, move):
        move_line = LineString(move)
        for line in self.drawn_lines:
            if LineString(line).intersects(move_line):
                return True
        return False

    def calculate_all_heuristics(self):
        all_points = [(x, y) for x in range(self.board_size) for y in range(self.board_size)]
        all_possible_moves = [list(move) for move in combinations(all_points, 2) if move not in self.drawn_lines]

        for move in all_possible_moves:
            if not self.check_line_intersection(move):
                triangle_score = self.calculate_triangle_score(move) * 1.5
                proximity_score = self.calculate_proximity_score(move) / 7
                linearity_score = self.calculate_linearity_score(move)
                total_score = triangle_score + proximity_score + linearity_score
                self.heuristic_scores[tuple(move)] = total_score

    def get_unconnected_points(self):
        # 연결되지 않은 점들을 찾는 함수
        connected_points = set()
        for line in self.drawn_lines:
            connected_points.update(line)
        return [p for p in self.whole_points if p not in connected_points]

    def get_available_moves(self):
        return [list([point1, point2]) for (point1, point2) in combinations(self.whole_points, 2) if self.check_availability([point1, point2])]

    def minimax_with_alpha_beta(self, depth, alpha, beta, is_maximizing_player):
        if depth == 0:
            return None, -float('inf') if is_maximizing_player else float('inf')

        best_move = None
        if is_maximizing_player:
            best_score = -float('inf')
            for move in self.get_available_moves():
                move_as_tuple = tuple(move)  # 점수 조회를 위해 튜플로 변환
                move_score = self.heuristic_scores.get(move_as_tuple, -float('inf') if is_maximizing_player else float('inf'))
                if move_score > best_score:
                    best_score = move_score
                    best_move = move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else:
            best_score = float('inf')
            for move in self.get_available_moves():
                move_score = self.heuristic_scores.get(move, float('inf'))  # 변경된 부분
                if move_score < best_score:
                    best_score = move_score
                    best_move = move
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

        return best_move, best_score

    def find_best_selection(self):
        unconnected_points = [p for p in self.whole_points if p not in self.drawn_lines]

        # 남은 점이 1개 또는 0개일 경우 미니맥스 알고리즘 실행
        if len(unconnected_points) <= 1:
            self.calculate_all_heuristics()
            best_move, _ = self.minimax_with_alpha_beta(5, -float('inf'), float('inf'), True)
            if best_move:
                best_move = list(best_move)
            return best_move
        else:
            # 그렇지 않으면 랜덤한 움직임 선택
            can_move = [list([point1, point2]) for (point1, point2) in combinations(unconnected_points, 2) if self.check_availability2([point1, point2])]
            return random.choice(can_move) if can_move else None
    
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

    def check_availability2(self, line):
        line_string = LineString(line)

        if not all(point in self.get_unconnected_points() for point in line):
            return False

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
