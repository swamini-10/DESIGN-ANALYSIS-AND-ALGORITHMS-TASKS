import matplotlib.pyplot as plt
import numpy as np

N = int(input("Enter N (board size): "))

print("Enter blocked cells as row,col pairs (e.g., 0,2). Type 'done' to finish:")
blocked_cells = []
while True:
    inp = input()
    if inp.lower() == 'done':
        break
    try:
        r, c = map(int, inp.split(','))
        if 0 <= r < N and 0 <= c < N:
            blocked_cells.append((r, c))
        else:
            print("Invalid coordinate — must be between 0 and", N-1)
    except:
        print("Please enter in format row,col (e.g., 2,3)")

print(f"\nBlocked cells: {blocked_cells}\n")

# Board
board = [["." for _ in range(N)] for _ in range(N)]
for (r, c) in blocked_cells:
    board[r][c] = "X"

solutions = []

# Safety
def is_safe(row, col):
    if board[row][col] == "X":
        return False
    for i in range(row):
        if board[i][col] == "Q":
            return False
    # upper-left diagonal
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == "Q":
            return False
        i -= 1; j -= 1
    # upper-right diagonal
    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if board[i][j] == "Q":
            return False
        i -= 1; j += 1
    return True

# Backtracking
def solve(row=0):
    if row == N:
        solutions.append(["".join(r) for r in board])
        return
    for col in range(N):
        if is_safe(row, col):
            board[row][col] = "Q"
            solve(row + 1)
            board[row][col] = "."

solve()

#Visualization
def plot_board(sol, title=""):
    fig, ax = plt.subplots(figsize=(4,4))
    for r in range(N):
        for c in range(N):
            # color pattern chess
            color = "white" if (r + c) % 2 == 0 else "lightgray"
            ax.add_patch(plt.Rectangle((c, N-1-r), 1, 1, color=color, ec="black"))
            if (r, c) in blocked_cells:
                ax.add_patch(plt.Rectangle((c, N-1-r), 1, 1, color="black"))
            elif sol[r][c] == "Q":
                ax.text(c+0.5, N-1-r+0.5, "♛", ha="center", va="center", fontsize=22, color="red")
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.show()

#Solutions 
print(f"Total solutions found: {len(solutions)}")
if len(solutions) == 0:
    print("No valid arrangements found.")
else:
    for i, sol in enumerate(solutions, start=1):
        plot_board(sol, f"Solution {i}")